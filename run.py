"""Main runner: verify and test an external trained model, or optionally retrain."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from dependencies.common import read_json
from dependencies.experiment import TrainingConfig, train, calibrate, evaluate
from dependencies.prepare import PreparationConfig, prepare_dataset
from dependencies.deployment import Deployment, export_bundle, external_path
from dependencies.inference import predict_files, test_prepared
from dependencies.mixed_preparation import prepare as prepare_mixed


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (("verify", "Verify model assets and load the checkpoint"),
                            ("predict", "Estimate all-node complex voltages from measurement arrays"),
                            ("test", "Evaluate the original frozen held-out test partition")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--model-dir", required=True, help="Deployment directory containing bundle.json")
        sub.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
        sub.add_argument("--batch-size", type=int, default=16)
        sub.add_argument("--threads", type=int, default=4)
        if name == "predict":
            sub.add_argument("--measurements", required=True)
            sub.add_argument("--sensor-order", required=True, help="JSON list describing the actual input sensor axis")
            sub.add_argument("--availability")
            sub.add_argument("--truth")
            sub.add_argument("--truth-node-order")
            sub.add_argument("--sample-indices")
            sub.add_argument("--output", required=True, help="New output directory")
        if name == "test":
            sub.add_argument("--prepared-dir", required=True)
            sub.add_argument("--conditions", nargs="+", choices=("clean", "noisy", "fdi"), default=["clean", "noisy", "fdi"])
            sub.add_argument("--save-predictions", action="store_true")
            sub.add_argument("--output", required=True, help="New output directory")
    sub = commands.add_parser("export", help="Export a validation-selected model from a training run")
    sub.add_argument("--prepared-dir", required=True)
    sub.add_argument("--run-dir", required=True)
    sub.add_argument("--output", required=True)
    sub = commands.add_parser("prepare", help="Prepare external complex-voltage trajectories for optional retraining")
    sub.add_argument("--config", required=True, help="JSON fields for PreparationConfig; paths are relative to the working directory")
    sub = commands.add_parser("prepare-mixed", help="Rebuild the original mixed-393 preparation from external native exports")
    sub.add_argument("--source", required=True)
    sub.add_argument("--source-batch", required=True)
    sub.add_argument("--phase-mapping", required=True)
    sub.add_argument("--output", required=True)
    sub.add_argument("--seed", type=int, default=20261020)
    sub = commands.add_parser("train", help="Optional training, calibration, and evaluation on prepared data")
    sub.add_argument("--prepared-dir", required=True)
    sub.add_argument("--run-dir", required=True)
    sub.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    sub.add_argument("--condition", default="fdi", choices=("clean", "noisy", "fdi"))
    sub.add_argument("--epochs", type=int, default=50)
    sub.add_argument("--patience", type=int, default=10)
    sub.add_argument("--batch-size", type=int, default=16)
    sub.add_argument("--threads", type=int, default=4)
    sub.add_argument("--stage", default="train", choices=("train", "calibrate", "test", "all"))
    sub.add_argument("--save-predictions", action="store_true")
    return result


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        for name in ('model_dir', 'prepared_dir', 'run_dir', 'source', 'source_batch',
                     'phase_mapping', 'output', 'measurements', 'sensor_order', 'availability',
                     'truth', 'truth_node_order', 'sample_indices'):
            value = getattr(args, name, None)
            if value is not None:
                external_path(value)
        if args.command in ("verify", "predict", "test"):
            deployment = Deployment(args.model_dir, args.device, args.batch_size, args.threads)
            if args.command == "verify":
                report = deployment.summary()
            elif args.command == "predict":
                report = predict_files(deployment, args.measurements, args.sensor_order, args.output,
                                       args.availability, args.truth, args.truth_node_order, args.sample_indices)
            else:
                report = test_prepared(deployment, args.prepared_dir, args.output, args.conditions, args.save_predictions)
        elif args.command == "export":
            meta = export_bundle(args.prepared_dir, args.run_dir, args.output)
            report = dict(exported=True, nodes=meta["nodes"], sensors=meta["sensors"], best_epoch=meta["best_epoch"])
        elif args.command == "prepare":
            config = PreparationConfig(**read_json(args.config))
            for name in ('voltage_dir', 'output_dir', 'node_order_path', 'system_dir', 'physics_dir',
                         'sensor_indices_path', 'group_file', 'split_file'):
                if getattr(config, name):
                    external_path(getattr(config, name))
            manifest = prepare_dataset(config)
            report = dict(prepared=True, nodes=manifest["nodes"], sensors=manifest["sensors"],
                          trajectories={key: len(names) for key, names in manifest["splits"].items()})
        elif args.command == "prepare-mixed":
            manifest = prepare_mixed(args.source, args.output, args.seed,
                                     source_batch=args.source_batch, phase_mapping=args.phase_mapping)
            report = dict(prepared=True, nodes=manifest["nodes"], sensors=manifest["sensors"],
                          trajectories={key: len(names) for key, names in manifest["splits"].items()})
        else:
            prepared, run = Path(args.prepared_dir).expanduser().resolve(), Path(args.run_dir).expanduser().resolve()
            if run == prepared or run.is_relative_to(prepared):
                raise ValueError("Keep the training run outside the prepared dataset")
            settings = dict(condition=args.condition, device=args.device, epochs=args.epochs, patience=args.patience,
                            batch_size=args.batch_size, threads=args.threads)
            if run.exists() and any(run.iterdir()) and not (run / "training_config.json").exists():
                raise FileExistsError("Nonempty run directory has no saved training configuration; choose a fresh run")
            if (run / "training_config.json").exists():
                saved = read_json(run / "training_config.json")
                settings = {**saved, "device": args.device, "threads": args.threads}
                settings.pop("prepared_dir"); settings.pop("run_dir")
                print("Resuming with the saved numerical settings; dataset identity is verified.", flush=True)
            config = TrainingConfig(str(prepared), str(run), **settings)
            report = {}
            if args.stage in ("train", "all"):
                report["training"] = train(config)
            if args.stage in ("calibrate", "all"):
                report["calibration"] = calibrate(config)
            if args.stage in ("test", "all"):
                report["test"] = evaluate(config, save_predictions=args.save_predictions)
        print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        command_parser.exit(2, f"ptae: {exc}\n")


if __name__ == "__main__":
    main()
