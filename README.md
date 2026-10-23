# PTAE: portable model evaluation

Code-only evaluation repository for a trained **physics transformer autoencoder (PTAE)** that estimates clean, all-node complex voltages from noisy sensor measurements with optional false data injection (FDI). The repository is prepared for technical evaluation and sharing with the U.S. Department of Energy (DOE).

**Start with the separately supplied model. Training is optional.** Download the model bundle separately and place it outside this repository. The code repository contains no trained weights, normalization arrays, feeder arrays, datasets, or generated results. Historical results for the 393-trajectory experiment are documented below.

## Quick start

Place the separately downloaded model bundle at `../PTAE_DOE_assets/model/` (a sibling of the repository), or supply its actual external path with `--model-dir`. This folder must contain `bundle.json`, `best_model.pth`, `normalization.npz`, `sensor_order.json`, and `system/`.

Python **3.11** is recommended. From the repository root on Linux or macOS:

```bash
python3.11 -m venv ../ptae-venv
source ../ptae-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python run.py verify --model-dir ../PTAE_DOE_assets/model
```

On Windows, create the environment with `py -3.11 -m venv ..\ptae-venv` and activate it with `..\ptae-venv\Scripts\Activate.ps1`. The remaining Python commands are the same. For CUDA, install the PyTorch build appropriate for the local driver instead of the CPU wheel. CPU is the default; `--device cuda` explicitly requests a GPU.

`verify` checks every model asset checksum, original core source hashes, architecture, normalization, and sensor/node ordering, then loads the checkpoint. Expected dimensions are **3,913 bus-phase nodes**, **1,173 sensor locations**, and **epoch 49**. No dataset is required for this check. `python run.py --help` lists all commands. Package installation is not required. The runner prevents model/data/output paths inside the repository.

Run the software tests without a private dataset:

```bash
python -B -m unittest discover -s tests -v
```

Tests generate small numerical fixtures in temporary directories. They check physics, split exclusions, training/resumption, calibration, export, inference parity, relocation, input validation, and artifact integrity. Synthetic fixture accuracy is not a scientific performance result.

See [VALIDATION.md](VALIDATION.md) for the executed clean-install checks and full 40-trajectory test rerun against the historical results.

## Test the trained model on supplied measurements

Supply measurement arrays and a JSON list describing their **actual sensor order**. Paths below are examples to replace with the separately supplied evaluation data:

```bash
python run.py predict \
  --model-dir ../PTAE_DOE_assets/model \
  --measurements ../PTAE_DOE_assets/data/evaluation/measurements.npy \
  --sensor-order ../PTAE_DOE_assets/data/evaluation/sensor_order.json \
  --output ../PTAE_DOE_assets/outputs/evaluation
```

To score against clean voltage truth, add the reference arrays and their node order:

```bash
python run.py predict \
  --model-dir ../PTAE_DOE_assets/model \
  --measurements ../PTAE_DOE_assets/data/evaluation/measurements.npy \
  --sensor-order ../PTAE_DOE_assets/data/evaluation/sensor_order.json \
  --availability ../PTAE_DOE_assets/data/evaluation/availability.npy \
  --truth ../PTAE_DOE_assets/data/evaluation/truth.npy \
  --truth-node-order ../PTAE_DOE_assets/data/evaluation/node_order.json \
  --output ../PTAE_DOE_assets/outputs/scored_evaluation
```

| Input | Contract |
| --- | --- |
| Measurements | `.npy`, real floating point, shape `(T, 1173, 3)`, `T > 0`; channels **`[Vm, P, Q]`**, all in per unit. |
| Sensor order | JSON list of 1,173 bus-phase names in the measurement sensor-axis order. Must exactly match `../PTAE_DOE_assets/model/sensor_order.json`. Reorder measurements and availability together when needed. |
| Availability, optional | `.npy`, shape `(T, 1173)`, binary `0/1`, shared by all three channels at that sensor. Omitted means every sensor is available. Missing measurements require finite placeholders and availability `0`. |
| Clean truth, optional | `.npy`, complex64 or complex128, shape `(T, 3913)`, complex voltage in per unit. Used only for scoring after prediction. |
| Truth node order | Required with truth; JSON list matching `../PTAE_DOE_assets/model/system/ynodeorder.json` exactly. |
| Scoring timestamps, optional | `--sample-indices FILE.npy`: a nonempty, strictly increasing 1-D integer array of original zero-based timestamps. Requires truth; limits metrics to those timestamps. Predictions still cover every input timestamp. |

The model requires the **same feeder, node identities, reference-angle convention, sensor installation, and measurement definition** used for training. Matching dimensions alone is insufficient. The order files declare the supplied arrays' order; the program cannot infer their physical semantics from their values. Do not copy the model's order list onto differently ordered data.

`Vm` is node-to-ground magnitude. `P + jQ = V · conj(Y_network V)` is positive net injection into the passive network, using a **1 MVA** power base. It is not PV-only output or raw inverter telemetry. Convert kW/kvar to per unit by dividing by **1,000**. Voltage angle is estimated as part of the output and is not a measured input. No noise or FDI is regenerated during prediction.

Each output directory must be new. A successful prediction creates:

- `predictions.npy`: complex128 voltage estimates `(T, 3913)` in per unit; magnitude is `abs(V)`, angle is `numpy.angle(V)`.
- `node_order.json`: ordered output node names.
- `report.json`: optional metrics, input/checkpoint checksums, dimensions, and runtime versions/settings.

Inference is batched; `--batch-size 16 --threads 4` are the defaults. No target, attack label, physical-event schedule, or future observation is provided to the estimator.

## Re-evaluate the original held-out test partition

Obtain the original prepared dataset separately, then run:

```bash
python run.py test \
  --model-dir ../PTAE_DOE_assets/model \
  --prepared-dir ../PTAE_DOE_assets/prepared/mixed_injection_393 \
  --conditions clean noisy fdi \
  --output ../PTAE_DOE_assets/outputs/mixed393_test
```

This command uses only the frozen **40 test trajectories** and their eligible timestamps. It checks the original prepared manifest identity and checksums of every array used for scoring, then computes metrics independently for the three input conditions. It does not fit normalization, train weights, select a checkpoint, or recalibrate. Its default output is a JSON report; add `--save-predictions` for full traces. Saving predictions changes batching around excluded timestamps, so very small floating point differences are possible.

The prepared dataset must retain `manifest.json`, `normalization.npz`, `system/ynodeorder.json`, `system/sensor_indices_0based.npy`, `system/ybus_pu.npz`, and the `truth/`, `availability/`, `sample_indices/`, and requested measurement-condition arrays for every test trajectory. The full original preparation also works. A new or modified dataset should use `predict` with an explicit input contract instead.

Historical calibration coverage and event/family breakdowns are described as applicable below; the standalone `test` command reports voltage accuracy overall and per trajectory. Optional training-stage evaluation provides the original calibrated and event/family breakdowns.

## Pipeline and model

```mermaid
flowchart LR
  A[Physical feeder simulation with cyberphysical attacks] --> B[Clean complex voltage truth]
  B --> C[Clean Vm / P / Q from matching network Y]
  C --> D[Gaussian measurement noise]
  D --> E[Measurement FDI]
  E --> F[Frozen training normalization and PTAE]
  F --> G[Estimated clean all-node complex voltage]
  B --> H[Supervision during training / reference during scoring]
  G --> H
```

**Clean means free of added measurement corruption.** Physical oscillation and imbalance attacks change the feeder state and remain in the clean voltage target. The estimator removes the effect of measurement noise/FDI while reconstructing the physical state that actually occurred. It is a state estimator, not a model that reverses physical attacks or classifies attacks.

The original model and loss are preserved byte for byte in `dependencies/model.py`. The training runtime is unchanged numerically; its provenance checks explicitly recognize the known folder-layout migration and still enforce dataset, numerical-setting, and remaining core-source identities. The model is a spatial, per-snapshot transformer autoencoder with graph propagation, learned node embeddings, 48 compressed attention tokens, four transformer layers, width 128, and four attention heads. It has **5,362,327 trainable parameters** and no temporal memory. The supervised output is normalized real/imaginary voltage at every node. Training also reconstructs clean measurements and applies a differentiable AC measurement loss. Loss weights are `1` for voltage-state MSE, `0.05` for measurement reconstruction, and `0.001` for the log-squared AC residual term. Training randomly masks 10% of sensor locations; inference uses the supplied availability mask.

The AC map uses the passive-network admittance assembled from the native simulation's PDE element matrices, excluding PC Norton equivalents. A static graph derived from that network supplies the model's physical coupling structure. The learned measurement-reconstruction head supports training; the public prediction output is complex voltage.

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

Training used FDI inputs, seed 42, batch size 16, AdamW with learning rate `1e-4` and weight decay `1e-5`, gradient clipping at `1.0`, a validation-loss plateau scheduler, up to 50 epochs, and early-stop patience 10. All 50 epochs completed. Validation selected **epoch 49**, with normalized real/imaginary state MSE **0.0009586120**. That normalized validation loss is distinct from the physical-unit test errors below.

## Recorded test results

Recorded on **2026-10-20**, using the same FDI-trained epoch-49 checkpoint for every condition. Each row pools **145,544,035 node-time values** from 37,195 eligible snapshots across 40 held-out trajectories. These are historical experiment results, transcribed from the original evaluation records; the repository does not bundle those records or the evaluation arrays.

| Input condition | Complex voltage RMSE (pu) | Magnitude RMSE (pu) | Wrapped angle MAE (degrees) | Maximum complex error (pu) |
| --- | ---: | ---: | ---: | ---: |
| Clean measurements | 9.425467e-5 | 7.420936e-5 | 0.001917673 | 0.007055078 |
| Gaussian noise | 1.052836e-4 | 8.434254e-5 | 0.002071860 | 0.007233010 |
| Gaussian noise + FDI | **1.117436e-4** | **9.022551e-5** | **0.002125034** | **0.011015962** |

For Gaussian-plus-FDI input, the physical-regime breakdown is:

| Physical regime | Eligible snapshots | Complex voltage RMSE (pu) | Magnitude RMSE (pu) |
| --- | ---: | ---: | ---: |
| No physical command active | 21,202 | 4.684536e-5 | 3.606879e-5 |
| Oscillation only | 6,451 | 1.487444e-4 | 1.212013e-4 |
| Imbalance only | 7,354 | 1.624904e-4 | 1.313797e-4 |
| Concurrent oscillation + imbalance | 2,188 | 1.924254e-4 | 1.563847e-4 |

The FDI-active subset contains 5,590 eligible snapshots, pooling all nodes at times when at least one measurement channel is tampered with. Its complex RMSE is **1.383930e-4 pu**, magnitude RMSE **1.147554e-4 pu**, and angle MAE **0.002295019°**. Delayed-packet episodes had the largest FDI-family complex RMSE (**2.939141e-4 pu**) and produced the overall maximum complex error (**0.011015962 pu**). Small pooled errors should be read alongside these worst-case errors.

Complex RMSE is `sqrt(mean(abs(V_est − V_true)^2))`; magnitude RMSE is `sqrt(mean((abs(V_est) − abs(V_true))^2))`. Angle MAE uses the absolute wrapped phase difference, excluding references with magnitude ≤ `1e-6 pu`. The real/imaginary-component RMSE is complex RMSE divided by `sqrt(2)`. Metrics weight eligible node-time values equally; they are not averages of trajectory RMSEs.

The original 95th-percentile calibration yielded pooled test node-time coverage of **94.2673% / 94.2508% / 93.9417%** for clean/noisy/FDI inputs. These are empirical marginal error-radius results, not simultaneous or distribution-free guarantees. Calibration arrays are excluded from this testing bundle; standalone prediction does not issue uncertainty intervals.

**Evaluation scope:** all trajectories share the same feeder and load/solar operating window. The test covers held-out physical attack schedules and measurement corruption realizations within that window. Duplicate-state exclusion does not establish statistical independence or generalization to unseen dates, feeders, topologies, or operating conditions. The injected FDI is not claimed to be stealthy or AC-consistent. Exact cross-device or cross-PyTorch bitwise equality is not promised.

## Optional retraining

See [RETRAINING.md](RETRAINING.md) for exact mixed-393 preparation, generic external-voltage preparation, training, calibration, testing, and model export. The simulation engine and private simulation inputs are external prerequisites for regenerating physical trajectories; inference and retraining on prepared arrays do not require OpenDSS or the original workstation.

## Repository contents and provenance

```text
PTAE_DOE/
  run.py                 Main runner: verify, predict, test, prepare, train, export
  dependencies/          Model, physics, training, preparation, and inference helpers
  tests/                 Software checks using temporary fixtures
  .github/               Automated test workflow
  README.md              Setup, input contract, pipeline, and 393-run results
  RETRAINING.md           Optional retraining instructions
  VALIDATION.md           Executed validation and reproduction limits
  prepare.example.json   Editable configuration example; no data
  requirements.txt       Python dependencies
  constraints-tested.txt Tested numerical-library versions

PTAE_DOE_assets/          Separate sibling directory; upload/distribute separately
  model/                 Checkpoint, normalization, feeder/order assets, hashes
  data/                  Evaluation arrays, when supplied separately
  prepared/              Prepared datasets, when supplied separately
  outputs/               Generated prediction/test outputs
  runs/                  Optional retraining outputs
```

Model SHA-256: `a060e07d99ffc072c41174412a6b182ef92a82cfcb2b493e5629bb3506c9449b`.

Original prepared-manifest SHA-256: `0ea13d78b62ea7fc6dd64d38db3b3553e7ccb2ace8ba18ad5fd3afb4b0d5845f`.

`../PTAE_DOE_assets/model/bundle.json` records exact asset/core-code hashes, architecture, and original numerical settings without workstation paths. The separately distributed checkpoint is about 21 MB. The repository has exactly three top-level support directories: `dependencies/`, `tests/`, and `.github/` (plus Git’s internal `.git/` metadata). Model assets, datasets, virtual environments, and generated artifacts belong outside the repository. The original research directory is independent of this code-only repository.
