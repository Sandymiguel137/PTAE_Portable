"""Export a self-contained estimator and validate its fixed feeder contract."""
from dataclasses import asdict
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from dependencies.common import digest, read_json, write_json
from dependencies.experiment import Bundle, Runtime, TrainingConfig, checkpoint_report, core_hashes, matches_identity
from dependencies.model import SPEC
from dependencies.prepare import load_y

FORMAT = "ptae-deployment-v1"
ASSETS = (
    "best_model.pth", "normalization.npz", "sensor_order.json",
    "system/ynodeorder.json", "system/sensor_indices_0based.npy", "system/ybus_pu.npz",
)


def external_path(path):
    """Keep model bundles, datasets, and generated artifacts outside this repo."""
    path = Path(path).expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if path.is_relative_to(repository):
        raise ValueError(f"Keep model/data/output paths outside the repository: {path}")
    return path


def checked_path(root, relative):
    """Resolve manifest entries without accepting absolute paths or traversal."""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes its root: {relative}")
    return path


def empty_output(path):
    path = external_path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists; choose a fresh directory: {path}")
    return path


def export_bundle(prepared_dir, run_dir, output):
    """Export original checkpoint bytes, train-only moments, and fixed graph assets."""
    prepared, run = Path(prepared_dir).resolve(), Path(run_dir).resolve()
    output = empty_output(output)
    if output.is_relative_to(prepared) or output.is_relative_to(run):
        raise ValueError("Export outside the source dataset and training run")
    saved = read_json(run / "training_config.json")
    cfg = TrainingConfig(**{**saved, "prepared_dir": str(prepared), "run_dir": str(run), "device": "cpu"})
    # Export needs only fixed model assets, not all training trajectories.
    bundle = Bundle(prepared, verify=False)
    if bundle.native:
        raise ValueError("This deployment format supports fixed-Y prepared datasets")
    report = checkpoint_report(cfg, bundle)
    for relative in ASSETS[1:]:
        if relative == "sensor_order.json":
            continue
        expected = bundle.manifest["hashes"].get(relative)
        if expected is None or digest(prepared / relative) != expected:
            raise ValueError(f"Prepared model asset changed: {relative}")
    output.mkdir(parents=True)
    try:
        shutil.copy2(run / "best_model.pth", output / "best_model.pth")
        for relative in ASSETS[1:]:
            if relative == "sensor_order.json":
                continue
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prepared / relative, destination)
        write_json(output / "sensor_order.json", [bundle.nodes[int(i)] for i in bundle.sensors])
        checkpoint = torch.load(output / "best_model.pth", map_location="cpu", weights_only=True)
        config = asdict(cfg)
        for key in ("prepared_dir", "run_dir", "device", "threads"):
            config.pop(key)
        metadata = dict(
            format=FORMAT, nodes=len(bundle.nodes), sensors=len(bundle.sensors),
            channel_order=["Vm", "P", "Q"], power_base_va=bundle.manifest["power_base_va"],
            voltage_unit="pu", power_unit="pu", measurement_mode="fixed",
            training_config=config, architecture=SPEC, source_identity=checkpoint["identity"],
            best_epoch=report["best_epoch"],
            hashes={relative: digest(output / relative) for relative in ASSETS},
        )
        write_json(output / "bundle.json", metadata)
        Deployment(output)  # Includes strict state-dict loading before export succeeds.
    except Exception:
        shutil.rmtree(output)
        raise
    return metadata


class Deployment:
    def __init__(self, root, device="cpu", batch_size=16, threads=4):
        if batch_size < 1 or threads < 1:
            raise ValueError("batch-size and threads must be positive")
        self.root = Path(root).expanduser().resolve()
        self.metadata = read_json(self.root / "bundle.json")
        meta = self.metadata
        if meta.get("format") != FORMAT or meta.get("architecture") != SPEC:
            raise ValueError("Unknown deployment format or architecture")
        if (meta.get("channel_order") != ["Vm", "P", "Q"] or meta.get("voltage_unit") != "pu"
                or meta.get("power_unit") != "pu" or meta.get("measurement_mode") != "fixed"
                or not np.isfinite(meta.get("power_base_va", 0)) or meta["power_base_va"] <= 0):
            raise ValueError("Invalid deployment measurement contract")
        if set(meta.get("hashes", {})) != set(ASSETS):
            raise ValueError("Deployment asset list differs from the required files")
        for relative, expected in meta["hashes"].items():
            if digest(checked_path(self.root, relative)) != expected:
                raise ValueError(f"Model asset checksum mismatch: {relative}")
        # Only the known folder-layout migration may differ from training provenance.
        current_identity = {**meta["source_identity"], "code_sha256": core_hashes()}
        if not matches_identity(meta["source_identity"], current_identity):
            raise ValueError("Core model/training source differs from the exported checkpoint")
        self.nodes = read_json(self.root / "system/ynodeorder.json")
        self.sensor_order = read_json(self.root / "sensor_order.json")
        self.sensors = np.load(self.root / "system/sensor_indices_0based.npy", allow_pickle=False)
        n, m = meta["nodes"], meta["sensors"]
        if (not isinstance(self.nodes, list) or len(self.nodes) != n
                or not all(isinstance(v, str) and v for v in self.nodes) or len(set(self.nodes)) != n
                or self.sensors.shape != (m,) or self.sensors.dtype.kind not in "iu"
                or len(np.unique(self.sensors)) != m or np.any(self.sensors < 0) or np.any(self.sensors >= n)):
            raise ValueError("Invalid ordered nodes or sensor indices")
        if self.sensor_order != [self.nodes[int(i)] for i in self.sensors]:
            raise ValueError("Sensor names and indices disagree")
        y = load_y(self.root / "system/ybus_pu.npz")
        if y.shape != (n, n):
            raise ValueError("Feeder matrix dimensions differ from node order")
        with np.load(self.root / "normalization.npz", allow_pickle=False) as z:
            stats = {k: z[k] for k in z.files}
        shapes = dict(vr_mean=(n,), vr_std=(n,), vi_mean=(n,), vi_std=(n,), ami_mean=(m, 3), ami_std=(m, 3))
        if set(stats) != set(shapes):
            raise ValueError("Normalization keys differ from the required fields")
        for key, shape in shapes.items():
            value = stats[key]
            if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                raise ValueError(f"Invalid normalization array: {key}")
            if key.endswith("std") and np.any(value <= 0):
                raise ValueError(f"Nonpositive normalization scale: {key}")
        settings = {**meta["training_config"], "device": device, "batch_size": batch_size, "threads": threads}
        cfg = TrainingConfig(prepared_dir="", run_dir="", **settings)
        if (cfg.voltage_std_floor <= 0 or cfg.measurement_std_floor <= 0
                or not np.isfinite([cfg.voltage_std_floor, cfg.measurement_std_floor]).all()
                or (cfg.input_normalization_clip is not None and
                    (not np.isfinite(cfg.input_normalization_clip) or cfg.input_normalization_clip <= 0))):
            raise ValueError("Invalid normalization settings")
        torch.set_num_threads(threads)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu")
        self.runtime = Runtime(cfg, SimpleNamespace(nodes=self.nodes, sensors=self.sensors, y=y, stats=stats, native=False))
        checkpoint = torch.load(self.root / "best_model.pth", map_location="cpu", weights_only=True)
        if checkpoint.get("identity") != meta["source_identity"] or checkpoint.get("architecture") != SPEC:
            raise ValueError("Checkpoint identity or architecture differs from bundle metadata")
        if checkpoint.get("epoch") != meta["best_epoch"]:
            raise ValueError("Checkpoint epoch differs from bundle metadata")
        expected_config = dict(meta["training_config"])
        if expected_config.get("input_normalization_clip") == 50.:
            expected_config.pop("input_normalization_clip")
        if expected_config != checkpoint["identity"]["config"]:
            raise ValueError("Training configuration differs from the checkpoint")
        self.model = self.runtime.model()
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

    def summary(self):
        return dict(verified=True, format=FORMAT, nodes=len(self.nodes), sensors=len(self.sensors),
                    best_epoch=self.metadata["best_epoch"], device=str(self.runtime.device),
                    checkpoint_sha256=self.metadata["hashes"]["best_model.pth"])
