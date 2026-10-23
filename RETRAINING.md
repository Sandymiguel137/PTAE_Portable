# Optional preparation and retraining

Run commands from the `PTAE_Portable` repository root with `ptae-venv` activated; use the installation instructions in [README.md](README.md). The main workflow is testing the separately supplied model. Retraining requires separately supplied voltage trajectories, matching feeder assets, and enough storage/compute. Use new prepared/run/output directories for changed experiments. The numerical model/training code is retained in `dependencies/`; provenance checks recognize the known folder consolidation while keeping the original checkpoint identity intact.

## Use an existing prepared dataset

The original 393-run preparation is directly compatible:

```bash
python run.py train \
  --prepared-dir ../PTAE_Portable_assets/prepared/mixed_injection_393 \
  --run-dir ../PTAE_Portable_assets/runs/retrained393 \
  --condition fdi --epochs 50 --patience 10 --batch-size 16 \
  --device cuda --stage all
```

Use `--device cpu` when a CUDA GPU is unavailable. `--stage all` trains, selects the validation checkpoint, calibrates on `cal`, and evaluates `test`. The default stage is only `train`. Calibration/testing can run separately with the same dataset and run:

```bash
python run.py train --prepared-dir ../PTAE_Portable_assets/prepared/mixed_injection_393 --run-dir ../PTAE_Portable_assets/runs/retrained393 --stage calibrate --device cuda
python run.py train --prepared-dir ../PTAE_Portable_assets/prepared/mixed_injection_393 --run-dir ../PTAE_Portable_assets/runs/retrained393 --stage test --device cuda
```

The training evaluator reports overall accuracy, physical regimes when metadata is present, FDI-active/family subsets, and empirical coverage. Add `--save-predictions` to save full test traces. Calibration may temporarily require substantial disk space for pooled per-node errors.

Restarting a run resumes matching epoch checkpoints, optimizer/scheduler states, and Torch RNG states. Saved numerical settings take precedence on resume, with an explicit CLI notice; paths, execution device, and thread count can change. Data/source/config identities remain strict. Resume uses the existing normalization and observations. To change settings or compare clean/noisy/FDI training, use a fresh run directory.

## Reproduce the mixed-393 preparation recipe

This uses the same split, noise, FDI, deduplication, and seed recipe as the reported experiment. It requires the original native AC phasor export, the native batch's completion records/plan, and verified bus-to-primary-phase mapping. These inputs and the physical simulator are not distributed here.

```bash
python run.py prepare-mixed \
  --source ../PTAE_Portable_assets/data/mixed_injection_400 \
  --source-batch ../PTAE_Portable_assets/data/native_mixed_batch_400 \
  --phase-mapping ../PTAE_Portable_assets/data/pv_mapping.csv \
  --output ../PTAE_Portable_assets/prepared/mixed393_rebuilt \
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
python run.py train --prepared-dir ../PTAE_Portable_assets/prepared/custom --run-dir ../PTAE_Portable_assets/runs/custom --stage all --device cpu
```

Paths in that configuration resolve relative to the current working directory and must point outside the repository. The runner enforces this for model, data, and output paths. Supply complex `V1.npy`, `V2.npy`, … arrays, an exact ordered node-name list, a matching Y matrix, sensor indices, and voltage bases if converting volts to per unit. The example specifies `(time, nodes)` voltage layout; change it if needed. Magnitude-only truth is rejected.

Supply `groups.json` mapping every trajectory filename to an independent scenario/day/profile group; all related trajectories must share a group. At least four groups are required for nonempty train/validation/calibration/test partitions. A trajectory split alone cannot protect shared dates/windows that were not declared. The training subset alone fits normalization and corruption scales. Use a matching system for a new feeder and train a new model; the supplied checkpoint is tied to its original graph and sensor identities.

## Export a retrained model

```bash
python run.py export \
  --prepared-dir ../PTAE_Portable_assets/prepared/mixed_injection_393 \
  --run-dir ../PTAE_Portable_assets/runs/retrained393 \
  --output ../PTAE_Portable_assets/retrained_model
python run.py verify --model-dir ../PTAE_Portable_assets/retrained_model
```

Export copies only the best checkpoint, training normalization, ordered node/sensor maps, and fixed feeder matrix. It validates the checkpoint's original data/code/settings identity and required asset checksums. No training arrays, optimizer state, prediction results, or calibration arrays are exported. The current deployment format supports fixed-Y preparations; native time-varying training datasets can still use the original training/calibration/evaluation runtime.

## The 393-trajectory experiment

The source batch planned 400 trajectories. **393 validated, completed trajectories were accepted; seven failed/incomplete trajectories were excluded.** Each accepted trajectory has 1,200 snapshots at 1 Hz. Physical scenarios include oscillation, imbalance, guaranteed overlap, and mixed attacks. Clean, noisy, and FDI variants of a trajectory stay in the same partition.

Partitions were frozen before fitting statistics or generating measurement corruptions, stratified by physical category. All-node complex states were rounded to 10 decimal places and hashed; each state was retained once, processing train → validation → calibration → test. Repeated timestamps remain in stored traces but are excluded from fitting/scoring through `sample_indices/*.npy`.

| Partition | Trajectories | Eligible snapshots | Role |
| --- | ---: | ---: | --- |
| Train | 275 | 259,055 | Fit normalization/noise scales and weights |
| Validation | 39 | 36,683 | Select checkpoint |
| Calibration | 39 | 36,523 | Fit empirical per-node error radii |
| Test | 40 | 37,195 | Final evaluation |

Gaussian noise is independent across time, sensors, and channels: `Vm σ = 0.001 pu`; P/Q σ is 1% of each sensor/channel's eligible-training RMS, floored at `1e-7 pu`. Measurement FDI starts from the exact same noisy realization. The approximate trajectory allocation per partition is 20% no FDI, 30% bias, 30% ramp, 10% delayed packets, 5% fake oscillation, and 5% fake imbalance, rounded to integer counts.

Each attacked trajectory has one 120–240 second episode on approximately 5–10% of sensors. Bias/ramp/fake-oscillation Vm peaks are 0.001–0.003 pu; P/Q peaks are 1–3% of training RMS with a floor. Delay replays packets from 1–5 seconds earlier, including their original noise. Fake oscillation periods are 12–40 seconds. Fake imbalance changes Vm with opposite offsets on verified primary A/B phases; conductor suffixes are not treated as phase labels. FDI timing is separate from physical attack schedules. Targets never receive added measurement noise or FDI.

Training used FDI inputs, seed 42, batch size 16, AdamW with learning rate `1e-4` and weight decay `1e-5`, gradient clipping at `1.0`, a validation-loss plateau scheduler, up to 50 epochs, and early-stop patience 10. All 50 epochs completed. Validation selected **epoch 49**, with normalized real/imaginary state MSE **0.0009586120**. That normalized validation loss is distinct from the physical-unit [recorded test errors](README.md#results).
