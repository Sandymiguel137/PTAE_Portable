# Validation and compatibility

This record distinguishes executed software checks from historical experiment results. Validation was performed on **2026-10-23**. No datasets, model arrays, generated predictions, or execution logs are committed to the code repository.

## Environment

A fresh external `ptae-venv` installed the existing tested numerical pins: **Python 3.11.11**, **NumPy 2.3.5**, **SciPy 1.16.3**, and **PyTorch 2.10.0+cpu**. Scientific evaluation used the available **PyTorch 2.10.0+cu128** environment. No numerical dependency pins were changed. `pip check` reported no broken requirements.

```bash
python3.11 -m venv ../ptae-venv
source ../ptae-venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

The environment was created with the available Python 3.11.11 interpreter. Local execution was on Linux. Windows/macOS instructions, Python 3.12, and hosted GitHub Actions were not executed locally. The workflow is configured to run CPU tests on Python 3.11/3.12 after publication.

## Automated tests

From the repository root with `ptae-venv` activated:

```bash
python -B -m unittest discover -s tests -v
```

**Observed: 29 tests run; 29 passed; 0 failures; 0 errors; 0 skips.** The unmodified starting suite also passed all 23 tests before changes.

The affected deployment tests were run separately before the complete suite:

```bash
python -B -m unittest discover -s tests -p test_deployment.py -v
```

**Observed: 15 tests run; 15 passed; 0 failures; 0 errors; 0 skips.** That targeted run used the existing CPU-only validation environment with the same numerical versions; the complete run used the fresh `ptae-venv`.

Existing assertions were preserved. Six added tests cover pre-installation help and missing dependencies, malformed bundle metadata, truncated arrays and failed-inference cleanup, evaluation from another working directory, changed-normalization export rejection, and invalid training/configuration inputs.

The full suite also exercises AC measurements/gradients, grouped splits, duplicate-state exclusions, all mixed FDI families, original-timestamp scoring, train-only statistics, training/calibration/evaluation, exact epoch resumption, stale-calibration rejection, export, relocation, order validation, missing sensors, and matching prediction outputs. These tests use temporary small fixtures; their prediction accuracy is not scientific performance evidence.

## CLI workflows

| Command | Executed validation |
| --- | --- |
| `verify` | Original external epoch-49 bundle loaded on CPU; metadata, asset hashes, source compatibility, order maps, normalization, and state dictionary passed. |
| `predict` | Full real 1,200-snapshot V5 trajectory scored on CPU with original availability/truth/eligible indices; invalid order/shape/dtype/truncated inputs and inference-failure cleanup checked. |
| `test` | All original test trajectories evaluated for clean/noisy/FDI input on CUDA; CLI also tested from an external working directory containing spaces. |
| `prepare` | Generic fixture preparation executed through the CLI; invalid JSON rejected with status 2. |
| `train` | Fixture training → calibration → evaluation → relocation → resume executed through the CLI. Saved numerical settings took precedence; invalid settings and unrelated output directories were rejected. |
| `export` | Fixture export and verification executed; changed normalization rejected before creating a model output directory. |
| `prepare-mixed` | Help, source recipe, component tests, and portable explicit arguments inspected. Full 393-trajectory preparation was not rerun. |

Help was executed with site packages disabled (`python -S run.py --help`): exit status 0. A command requiring numerical dependencies in the same restricted environment returned status 2 with installation guidance. Successful workflows return 0; tested validation errors return 2 without a traceback.

The real model verification command is:

```bash
python run.py verify --model-dir ../PTAE_Portable_assets/model
```

Its output reported `verified: true`, epoch 49, 3,913 nodes, 1,173 sensors, and checkpoint SHA-256 `a060e07d99ffc072c41174412a6b182ef92a82cfcb2b493e5629bb3506c9449b`.

## Scientific evaluation and prediction parity

The existing original dataset and saved experiment reports were available locally outside the repository. The actual evaluation command used the preserved research archive as its prepared-data source:

```bash
python run.py test \
  --model-dir ../PTAE_Portable_assets/model \
  --prepared-dir ../PTAE_Portable_OG/prepared/mixed_injection_393 \
  --conditions clean noisy fdi \
  --output ../PTAE_Portable_assets/outputs/audit393 \
  --device cuda
```

This local archive path records the executed audit command; users should use the separately supplied prepared dataset path shown in [README.md](README.md#reproducing-evaluation). The CUDA environment was activated for this command.

Each condition evaluated **40 held-out trajectories**, **37,195 eligible snapshots**, and **145,544,035 node-time values**. Comparing the newly computed overall metrics to the original 2026-10-20 report gave a maximum relative difference of **2.5626728686922634e-10**. Maximum complex errors and value counts matched exactly for all three conditions. Small aggregate differences follow batch-wise summation. The README retains the original recorded values and precision; rerun metrics did not replace the historical results.

For the complete V5 trajectory, all **1,200 × 3,913** CPU predictions were **bitwise identical** to the pre-audit output from the same CPU numerical build. Separately, 32 real V5 snapshots produced bitwise-identical predictions between the original research runtime and the portable runtime in the fresh CPU environment.

CPU versus historical CUDA predictions for those 32 snapshots differed by maximum complex voltage approximately **4.32e-6 pu** and RMS approximately **2.64e-7 pu**. This is a cross-device/build numerical comparison, not a guarantee of bitwise equality across hardware or library versions.

## Integrity and compatibility

- SHA-256 comparisons before/after this audit confirmed **all seven core source files** (`__init__.py`, `common.py`, `corruptions.py`, `experiment.py`, `metrics.py`, `model.py`, `prepare.py`) remained byte for byte unchanged during this audit.
- All seven files in the external original model bundle, including `bundle.json`, checkpoint, normalization, and feeder/order assets, remained byte for byte unchanged.
- Architecture, forward computations, physics, normalization values, loss weights, calibration, training numerical behavior, original checkpoint identities, and dataset partition identities were preserved. No new source-compatibility exception was added.
- The earlier source-layout migration already present in `experiment.py` remains authoritative. Six core files match the original research version exactly; the pre-existing runtime migration changes only identity enumeration/compatibility. This audit did not modify it.
- Model bundles work after relocation; absolute script paths and filesystem paths containing spaces are covered. Standalone inference does not need the original training directory.
- Model/test checksums, node/sensor ordering, external-path restrictions, and existing-output protection remain enforced. No fingerprint was edited to bypass a check.

## Evidence and limits

Local evidence is retained separately under `../PTAE_Portable_assets/validation/2026-10-23/`: installation logs, baseline/targeted/full test logs, CPU environment listing, verification JSON, before/after integrity records, prediction-parity record, and the full-test comparison. Numerical outputs remain under the external `outputs/` directory. These local evidence files are not part of the public code repository.

The full native simulation, 393-trajectory preparation, and 50-epoch training were not repeated. Their records, recipes, source, and compatibility were inspected; component/fixture tests and the complete held-out scientific evaluation were executed as stated above. The model does not establish unseen-feeder, unseen-date, or unseen-operating-window generalization. Minimum hardware requirements were not established.

A public asset download location and repository license remain owner decisions. No license text or permissions were invented. Markdown parsing, local links/anchors, Bash syntax, and source-control hygiene were checked locally; hosted GitHub rendering and CI remain unexecuted until publication.
