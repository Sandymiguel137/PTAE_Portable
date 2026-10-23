# Optional preparation and retraining

The main workflow is testing the separately supplied model. Retraining requires separately supplied voltage trajectories, matching feeder assets, and enough storage/compute. Use new prepared/run/output directories for changed experiments. The numerical model/training code is retained in `dependencies/`; provenance checks recognize the known folder consolidation while keeping the original checkpoint identity intact.

## Use an existing prepared dataset

The original 393-run preparation is directly compatible:

```bash
python run.py train \
  --prepared-dir ../PTAE_DOE_assets/prepared/mixed_injection_393 \
  --run-dir ../PTAE_DOE_assets/runs/retrained393 \
  --condition fdi --epochs 50 --patience 10 --batch-size 16 \
  --device cuda --stage all
```

Use `--device cpu` when a CUDA GPU is unavailable. `--stage all` trains, selects the validation checkpoint, calibrates on `cal`, and evaluates `test`. The default stage is only `train`. Calibration/testing can run separately with the same dataset and run:

```bash
python run.py train --prepared-dir ../PTAE_DOE_assets/prepared/mixed_injection_393 --run-dir ../PTAE_DOE_assets/runs/retrained393 --stage calibrate --device cuda
python run.py train --prepared-dir ../PTAE_DOE_assets/prepared/mixed_injection_393 --run-dir ../PTAE_DOE_assets/runs/retrained393 --stage test --device cuda
```

The training evaluator reports overall accuracy, physical regimes when metadata is present, FDI-active/family subsets, and empirical coverage. Add `--save-predictions` to save full test traces. Calibration may temporarily require substantial disk space for pooled per-node errors.

Restarting a run resumes matching epoch checkpoints, optimizer/scheduler states, and Torch RNG states. Saved numerical settings take precedence on resume, with an explicit CLI notice; paths, execution device, and thread count can change. Data/source/config identities remain strict. Resume uses the existing normalization and observations. To change settings or compare clean/noisy/FDI training, use a fresh run directory.

## Reproduce the mixed-393 preparation recipe

This uses the same split, noise, FDI, deduplication, and seed recipe as the reported experiment. It requires the original native AC phasor export, the native batch's completion records/plan, and verified bus-to-primary-phase mapping. These inputs and the physical simulator are not distributed here.

```bash
python run.py prepare-mixed \
  --source ../PTAE_DOE_assets/data/mixed_injection_400 \
  --source-batch ../PTAE_DOE_assets/data/native_mixed_batch_400 \
  --phase-mapping ../PTAE_DOE_assets/data/pv_mapping.csv \
  --output ../PTAE_DOE_assets/prepared/mixed393_rebuilt \
  --seed 20261020
```

Required native export structure:

```text
mixed_injection_400/
  manifest.json                       pymagic-sensor-npy-v2, accepted export records
  voltages/V<number>.npy              Complex (1200, N) voltage in pu
  voltages/V<number>.json             Source id, attack category, physical events
  sensor_clean/V<number>.npy          Clean float64 (1200, M, 3), [Vm,P,Q] in pu
  system/
    ynodeorder.json
    sensor_indices_0based.npy
    node_voltage_base_volts.npy
    ybus_pu.npz
    network_physics.json
native_mixed_batch_400/
  plan.json                          Planned trajectories
  <source_id>/complete.json           Successful status, convergence, voltage hash
pv_mapping.csv                       At least bus and primary_phase columns
```

The recipe accepts exactly 393 completed, converged runs and checks their voltage/network hashes. It checks clean observations against `V · conj(YV)` throughout each trajectory, freezes splits, excludes duplicate states, fits training statistics, and generates deterministic Gaussian/FDI observations. Original truth/clean arrays may be hard-linked to avoid copying; treat them as immutable. All derived noisy/FDI arrays are separate. The source directory is left untouched.

Rebuilt manifests contain provenance paths and file hashes, so a rebuild at a new location need not have the original manifest digest even when numerical arrays match. Train/export a new run for that rebuilt dataset. The supplied model's `test` command deliberately requires the original frozen manifest; its `predict` command can evaluate explicitly supplied arrays.

## Prepare other voltage trajectories

`dependencies.prepare` also supports generic fixed-Y or native time-varying measurement generation. The supplied generic example uses Gaussian noise plus configurable bias FDI. **It is a different experiment from the mixed-393 recipe.** Its P/Q noise uses eligible-training standard deviation, rather than the mixed-393 RMS recipe; it does not reproduce the five-family mixture.

Edit [prepare.example.json](prepare.example.json) to describe external files, then run from the repository root:

```bash
python run.py prepare --config prepare.example.json
python run.py train --prepared-dir ../PTAE_DOE_assets/prepared/custom --run-dir ../PTAE_DOE_assets/runs/custom --stage all --device cpu
```

Paths in that configuration resolve relative to the current working directory and must point outside the repository. The runner enforces this for model, data, and output paths. Supply complex `V1.npy`, `V2.npy`, … arrays, an exact ordered node-name list, a matching Y matrix, sensor indices, and voltage bases if converting volts to per unit. The example specifies `(time, nodes)` voltage layout; change it if needed. Magnitude-only truth is rejected.

Supply `groups.json` mapping every trajectory filename to an independent scenario/day/profile group; all related trajectories must share a group. At least four groups are required for nonempty train/validation/calibration/test partitions. A trajectory split alone cannot protect shared dates/windows that were not declared. The training subset alone fits normalization and corruption scales. Use a matching system for a new feeder and train a new model; the supplied checkpoint is tied to its original graph and sensor identities.

## Export a retrained model

```bash
python run.py export \
  --prepared-dir ../PTAE_DOE_assets/prepared/mixed_injection_393 \
  --run-dir ../PTAE_DOE_assets/runs/retrained393 \
  --output ../PTAE_DOE_assets/retrained_model
python run.py verify --model-dir ../PTAE_DOE_assets/retrained_model
```

Export copies only the best checkpoint, training normalization, ordered node/sensor maps, and fixed feeder matrix. It validates the checkpoint's original data/code/settings identity and required asset checksums. No training arrays, optimizer state, prediction results, or calibration arrays are exported. The current deployment format supports fixed-Y preparations; native time-varying training datasets can still use the original training/calibration/evaluation runtime.
