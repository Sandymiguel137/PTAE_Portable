"""Batched inference, optional reference scoring, and the frozen test partition."""
from pathlib import Path
import platform
import shutil

import numpy as np
import scipy
import torch

from dependencies.common import digest, read_json, write_json
from dependencies.experiment import Bundle, predict
from dependencies.metrics import new_accumulator, update_metrics, finalize_metrics
from .deployment import checked_path, empty_output


def load_array(path):
    if Path(path).suffix != ".npy":
        raise ValueError(f"Expected a .npy array: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise ValueError(f"Expected one numerical array: {path}")
    return array


def validate_arrays(deployment, raw, available=None, truth=None, indices=None):
    m, n = len(deployment.sensors), len(deployment.nodes)
    if raw.ndim != 3 or raw.shape[1:] != (m, 3) or not len(raw) or raw.dtype.kind != "f":
        raise ValueError(f"Measurements must be real floating point [T,{m},3] with T > 0")
    if available is not None and (available.shape != raw.shape[:2] or available.dtype.kind not in "bifu"):
        raise ValueError("Availability must be a binary numerical [T,M] array")
    if truth is not None and (truth.shape != (len(raw), n) or truth.dtype.kind != "c"):
        raise ValueError(f"Truth must be complex voltage [T,{n}] in pu")
    if indices is None:
        indices = np.arange(len(raw))
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices)
            or np.any(indices < 0) or np.any(indices >= len(raw)) or np.any(indices[1:] <= indices[:-1])):
        raise ValueError("Sample indices must be nonempty, strictly increasing integer timestamps in range")
    # Validate before writing predictions; memory stays bounded for large trajectories.
    for start in range(0, len(raw), 256):
        stop = start + 256
        if not np.isfinite(raw[start:stop]).all():
            raise ValueError("Measurements contain NaN or infinity; use finite placeholders with availability=0")
        if available is not None and not np.isin(available[start:stop], [0, 1]).all():
            raise ValueError("Availability must contain only zero and one")
        if truth is not None and not np.isfinite(truth[start:stop]).all():
            raise ValueError("Truth contains NaN or infinity")
    return indices


def infer(deployment, raw, available=None, truth=None, indices=None, prediction_path=None):
    """Truth is used only after each prediction for metrics; never passed to the model."""
    indices = validate_arrays(deployment, raw, available, truth, indices)
    output = None
    if prediction_path is not None:
        output = np.lib.format.open_memmap(prediction_path, mode="w+", dtype=np.complex128,
                                          shape=(len(raw), len(deployment.nodes)))
    accumulator = new_accumulator()
    # Preserve original contiguous batching when saving full traces. Metrics-only
    # mode evaluates eligible timestamps, as the research evaluator does.
    times = np.arange(len(raw)) if output is not None else indices
    try:
        for start in range(0, len(times), deployment.runtime.cfg.batch_size):
            selected = times[start:start + deployment.runtime.cfg.batch_size]
            av = np.ones((len(selected), len(deployment.sensors)), np.uint8) if available is None else available[selected]
            prediction = predict(deployment.runtime, deployment.model, raw[selected], av)
            if output is not None:
                output[selected] = prediction
            if truth is not None:
                keep = np.isin(selected, indices, assume_unique=True)
                if keep.any():
                    update_metrics(accumulator, prediction[keep], truth[selected[keep]])
        if output is not None:
            output.flush()
    finally:
        if output is not None:
            del output
    return dict(snapshots=len(raw), eligible_snapshots=len(indices),
                metrics=finalize_metrics(accumulator) if truth is not None else None)


def environment(deployment):
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                torch=torch.__version__, device=str(deployment.runtime.device),
                threads=torch.get_num_threads(), batch_size=deployment.runtime.cfg.batch_size)


def predict_files(deployment, measurements, sensor_order, output, availability=None,
                  truth=None, truth_node_order=None, sample_indices=None):
    if read_json(sensor_order) != deployment.sensor_order:
        raise ValueError("Input sensor order differs from the model; reorder measurements and availability explicitly")
    if truth is not None:
        if truth_node_order is None or read_json(truth_node_order) != deployment.nodes:
            raise ValueError("Scoring requires --truth-node-order matching the model's exact node order")
    elif truth_node_order is not None or sample_indices is not None:
        raise ValueError("truth-node-order and sample-indices require --truth")
    raw = load_array(measurements)
    av = None if availability is None else load_array(availability)
    reference = None if truth is None else load_array(truth)
    indices = None if sample_indices is None else load_array(sample_indices)
    validate_arrays(deployment, raw, av, reference, indices)
    output = empty_output(output)
    if output.is_relative_to(deployment.root):
        raise ValueError("Write predictions outside the model bundle")
    output.mkdir(parents=True)
    try:
        report = infer(deployment, raw, av, reference, indices, output / "predictions.npy")
        report.update(format="ptae-prediction-v1", checkpoint_sha256=deployment.metadata["hashes"]["best_model.pth"],
                      input_sha256={key: digest(path) for key, path in dict(
                          measurements=measurements, sensor_order=sensor_order, availability=availability,
                          truth=truth, truth_node_order=truth_node_order, sample_indices=sample_indices).items()
                          if path is not None}, environment=environment(deployment),
                      prediction_layout="time_nodes", voltage_unit="pu", channel_order=["Vm", "P", "Q"])
        write_json(output / "node_order.json", deployment.nodes)
        write_json(output / "report.json", report)
    except Exception:
        shutil.rmtree(output)
        raise
    return report


def test_prepared(deployment, prepared_dir, output, conditions=("clean", "noisy", "fdi"), save_predictions=False):
    """Score only the original held-out test trajectories, with frozen exclusions."""
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= {"clean", "noisy", "fdi"}:
        raise ValueError("Choose unique conditions from clean, noisy, fdi")
    root = Path(prepared_dir).expanduser().resolve()
    expected = deployment.metadata["source_identity"]["prepared_sha256"]
    if digest(root / "manifest.json") != expected:
        raise ValueError("Prepared manifest differs from the checkpoint's frozen dataset; use predict for new measurements")
    bundle = Bundle(root, verify=False)
    names = bundle.splits["test"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Test partition is empty or repeated")
    # Check every artifact used by scoring, without reading unused train/val/cal data.
    required = {"normalization.npz", "system/ynodeorder.json", "system/sensor_indices_0based.npy", "system/ybus_pu.npz"}
    for name in names:
        if Path(name).name != name or not name.endswith(".npy"):
            raise ValueError("Invalid test trajectory filename")
        required.update(f"{folder}/{name}" for folder in ("truth", "availability", *conditions))
        if bundle.manifest.get("sample_filter"):
            required.add(f"sample_indices/{name}")
    for relative in sorted(required):
        checksum = bundle.manifest["hashes"].get(relative)
        if checksum is None or digest(checked_path(root, relative)) != checksum:
            raise ValueError(f"Prepared test asset changed: {relative}")
    output = empty_output(output)
    if output.is_relative_to(root) or output.is_relative_to(deployment.root):
        raise ValueError("Write test outputs outside the dataset and model bundle")
    output.mkdir(parents=True)
    reports = {}
    try:
        for condition in conditions:
            accumulator = new_accumulator()
            individual = []
            if save_predictions:
                (output / condition).mkdir()
            for i, name in enumerate(names, 1):
                raw, av, truth = [bundle.array(folder, name) for folder in (condition, "availability", "truth")]
                indices = validate_arrays(deployment, raw, av, truth, bundle.indices(name))
                one = new_accumulator()
                dest = None
                if save_predictions:
                    dest = np.lib.format.open_memmap(output / condition / name, mode="w+", dtype=np.complex128, shape=truth.shape)
                try:
                    times = np.arange(len(raw)) if dest is not None else indices
                    for start in range(0, len(times), deployment.runtime.cfg.batch_size):
                        selected = times[start:start + deployment.runtime.cfg.batch_size]
                        prediction = predict(deployment.runtime, deployment.model, raw[selected], av[selected])
                        if dest is not None:
                            dest[selected] = prediction
                        keep = np.isin(selected, indices, assume_unique=True)
                        if keep.any():
                            update_metrics(one, prediction[keep], truth[selected[keep]])
                            update_metrics(accumulator, prediction[keep], truth[selected[keep]])
                    if dest is not None:
                        dest.flush()
                finally:
                    if dest is not None:
                        del dest
                individual.append(dict(file=name, eligible_snapshots=len(indices), metrics=finalize_metrics(one)))
                print(f"{condition}: {i}/{len(names)} test trajectories", flush=True)
            reports[condition] = dict(partition="test", overall=finalize_metrics(accumulator),
                                      eligible_snapshots=sum(v["eligible_snapshots"] for v in individual),
                                      per_trajectory=individual)
        result = dict(format="ptae-test-v1", prepared_sha256=expected,
                      checkpoint_sha256=deployment.metadata["hashes"]["best_model.pth"],
                      environment=environment(deployment), conditions=reports)
        write_json(output / "node_order.json", deployment.nodes)
        write_json(output / "report.json", result)
    except Exception:
        shutil.rmtree(output)
        raise
    return result
