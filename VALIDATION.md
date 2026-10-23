# Validation performed

Validated on 2026-10-23 with Python 3.11.11, NumPy 2.3.5, SciPy 1.16.3, and PyTorch 2.10.0. A fresh virtual environment installed the CPU-only PyTorch wheel and numerical dependencies using `requirements.txt` / `constraints-tested.txt`; GPU checks used the installed CUDA build. No validation data or generated outputs are tracked in this repository.

## Executed checks

- **23 software tests passed** in the fresh CPU-only environment. Fixtures cover AC measurements and gradients, training-only statistics, group/snapshot exclusions, all mixed FDI families, original-time scoring, training/calibration/evaluation, exact interrupted-epoch resume, stale-calibration rejection, deployment integrity, invalid inputs/order rejection, finite missing-sensor behavior, export/relocation, and CLI paths containing spaces.
- **The optional CLI workflow passed end to end:** generic preparation → one-epoch fixture training → calibration → test → export → verification → relocation → resume with saved numerical settings. Existing unrelated output directories were rejected.
- **The actual epoch-49 model loaded in the clean CPU environment**, with all fixed model assets and original core source hashes verified. The main runner can be invoked using its absolute script path from outside the repository. `pip check` found no dependency conflicts.
- **Real feeder inference matched the original runtime:** deployment and original-runtime CPU predictions were bitwise identical for 32 snapshots from held-out trajectory V5. Deployment CUDA predictions on those snapshots were bitwise identical to the historical saved CUDA predictions.
- **All 40 original held-out trajectories were reevaluated** on CUDA for clean, Gaussian, and Gaussian-plus-FDI conditions. Each condition scored 37,195 eligible snapshots / 145,544,035 node-time values. Every overall numerical metric agreed with the historical report to better than `3e-10` relative difference; all three maximum complex errors and value counts matched exactly. Minor summation differences arise from accumulating metrics in batches.
- **A complete 1,200-snapshot real V5 trajectory was predicted and scored with the CPU-only installation**, using its frozen eligible timestamps and externally supplied truth/order/availability arrays.
- **The architecture and loss remain byte for byte identical** to the research version. Six original core files are unchanged. Only source-identity enumeration/migration in the training runtime was adjusted to consolidate helper files in one dependency folder. The model checkpoint and original training identity were preserved without rewriting their bytes.

## Numerical and reproduction scope

For the 32-snapshot cross-device check, CPU versus historical CUDA predictions differed by at most approximately `4.32e-6 pu` in complex voltage, with RMS difference approximately `2.64e-7 pu`. This is a device/kernel numerical difference; the deployment and original runtime agree exactly when run on the same CPU configuration. Recorded scientific metrics in the README remain the original CUDA experiment values.

The full 393-trajectory preparation and 50-epoch training were not repeated for this packaging task. Their existing manifests, checkpoint, completed-training summary, split/sample counts, and saved test reports were inspected; the complete held-out test was rerun as described above. Repeating physical simulation requires the separately supplied simulator and its inputs. Repeating training requires the separately supplied prepared dataset.

GitHub Actions is configured for CPU software tests on Python 3.11 and 3.12. The hosted workflow has not been run until this repository is pushed; local execution was on Python 3.11.

## Code-only repository layout

The model bundle was moved to the external sibling `PTAE_DOE_assets/model/`. The repository uses one `run.py` entrypoint, one `dependencies/` implementation directory, `tests/`, and `.github/`. Documentation and the configuration example live at the root. Packaging output and Python caches were removed. The runner rejects model/data/output paths inside the repository.

After consolidation, all 23 tests passed again using the main runner, and a full 1,200-snapshot real V5 CPU prediction was bitwise identical to the earlier layout. The original external epoch-49 bundle verifies successfully without modifying its checkpoint or metadata.
