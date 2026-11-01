# CD8 representation and target-aware ablations

Final experiment: **7 HGB models + 5 neural input groups × 3 seeds**, followed by correctness_001. Computation and visual QA are complete; the joint weak-positive/false-positive criterion was **not met**. Human adjudication of 31 real cells remains pending. No training or data access occurred during repository publication.

- [中文完整报告](reports/REPORT_zh.md) · [English report](reports/REPORT_en.md)
- [External concise results](reports/CONCISE_EXTERNAL_RESULTS.csv) · [All external operating points](results/EXTERNAL_MAIN.csv)
- [Tables](TABLE_INDEX.csv) · [Claim-to-evidence index](CONCLUSION_INDEX.csv) · [Versions](VERSION_INDEX.csv)
- [Review scope](review/README.md) · [Publication changes](PUBLICATION_NOTES.md)

`ENS_*` means averaging three seed probabilities **before** calculating metrics. Seed metric means/SD are separate tables. H1 remains the original frozen internal selection; tree-node counts and neural scalar-weight counts are not comparable complexity units. The dtype repair changed **zero** real prediction decisions. AP improvement is limited to the controlled lightweight input ablation; it is not evidence of general superiority to Nimbus or E2.

## Reproduce without raw data or GPU

From the repository root, use Python 3.11 with NumPy, pandas and SciPy:

```bash
python analyses/cd8_representation_targetaware_20261031_v1/scripts/test_correctness.py
python -m unittest discover -s analyses/cd8_representation_targetaware_20261031_v1/tests
python tools/check_publication.py
python -m unittest discover -s tests
```

These checks cover synthetic descriptors, atomic output validation, dtype ties, budget/retry state, publication provenance and saved summary consistency. They do not rerun training or independently recreate patient-level performance. Runtime packages used in the original two environments are recorded in [feature environment](provenance/ENVIRONMENT_features_HGB.json) and [neural environment](provenance/ENVIRONMENT_neural_report.json); these historical inventories are not an interchangeable installation lockfile.

## Relocated full reproduction (Linux; separately licensed inputs required)

Use a **new** experiment directory with copied scripts/configuration. The code supports `MIF_PROJECT_ROOT` (historical upstream data/cache layout), `MIF_RUN_ROOT` (new experiment directory), `MIF_SOURCE_ROOT` (this repository checkout), `MIF_FEATURE_PYTHON` and `MIF_NEURAL_PYTHON`. Defaults point to the checkout/current interpreter; they do not download data. Stage preparation still requires the original PhenoBIC extraction audit, images/masks, bounding boxes, E2 features/calibration/model, robust bounds and frozen risk definitions. Published code alone does not contain these inputs. The existing `mif_features.py` in this repository remains the legacy pixel-feature dependency.

```bash
export MIF_SOURCE_ROOT=/path/to/this/repository
export MIF_PROJECT_ROOT=/path/to/authorized/project
export MIF_RUN_ROOT=/path/to/new/experiment
export MIF_FEATURE_PYTHON=/path/to/features/python
export MIF_NEURAL_PYTHON=/path/to/neural/python
# Copy this directory into the new experiment; MIF_SOURCE_ROOT locates sibling source modules.
# Validate upstream input hashes and freeze a NEW run manifest before execution.
# Original stage order is listed in reports/REPRODUCE.md; no stages were rerun for publication.
```

The pipeline uses Linux process/resource semantics. The read-only/synthetic publication tests also run on Windows. Set interpreter variables explicitly to preserve the original two-environment separation. Paths recorded in historical JSON/hash manifests remain immutable provenance, not portable executable configuration. `revision_runner.py` is specific to the completed frozen run and asserts its exact selection hash; it is **not** a generic new-experiment launcher. Do not copy completed markers or use publication source hashes to validate old runtime caches.

Large weights, raw data, per-cell scores, calibration-reserved IDs and actual review images are intentionally absent. Pan-M remains sealed. [Data policy](../../docs/DATA_POLICY.md).
