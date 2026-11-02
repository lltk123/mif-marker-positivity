# CD8 technical proxies and calibration: audited results

**Completed, with explicit negative results.** Stage0 evaluated all 539,132 internal cells under 16 technical proxy scenarios using three existing A seeds (48 changed seed predictions). Spatial zero-label calibration missed Gate1 because AP=0.965875528<0.966. Boundary-active calibration with 20 labels per core met Gate2 by the three-repeat mean, with only 2/3 repeats passing. Physics continuation failed the full internal gate; its synthetic follow-up and external predictions were not run.

- [中文结果入口](ANALYSIS_REPORT.md) · [中文完整报告](reports/EXECUTION_REPORT_ZH.md) · [English full report](reports/EXECUTION_REPORT_EN.md)
- [Table index](TABLE_INDEX.csv) · [Claim-to-evidence index](CONCLUSION_INDEX.csv) · [Source/publication versions](VERSION_INDEX.csv)
- [Execution ledger](EXPERIMENT_LEDGER.md) · [Publication scope](PUBLICATION_NOTES.md) · [Human review remains pending](review/README.md)

The external benchmark was previously inspected and is exploratory. It is not a new blind test. ENS means an average of three seed probabilities before metric computation; seed means/SD are separate. Calibration repeats vary the reserved label pools, not model seeds or independent patients. Core IDs refer to published dataset units; TMA1/2 share samples.

## Read-only and synthetic checks

Use Python 3.11 with NumPy, pandas, SciPy, scikit-learn and PyTorch. These commands do not train or access image data:

```bash
python -B -m unittest discover -s analyses/cd8_physics_autocal_20261102_v1/tests -p test_published_results.py -v
# With a working PyTorch installation, run the original17synthetic checks too:
python -B -m unittest discover -s analyses/cd8_physics_autocal_20261102_v1/tests -v
python -B analyses/cd8_physics_autocal_20261102_v1/scripts/check_synthetic_gate.py
python -B -m unittest discover -s tests -v
python -B tools/check_publication.py
```

The integration check uses mocked predictions and prints its result; it does not overwrite the historical audit JSON. The original 14 inherited +17 new +1 integration checks are historical evidence; publication-time reruns are reported separately.

## Reproduce calculations with authorized inputs

No model stage was rerun during publication. Full reproduction requires the prior condition experiment's A/CV_A checkpoints, frozen scores, thresholds, reserve pools and strata, plus the earlier representation feature/crop cache and PhenoBIC inputs. The condition experiment is a historical dependency; its complete data/weights are not supplied by cloning this repository. Its exact three immediate runtime modules are supplied under `scripts/prior_runtime/`.

Set `MIF_PROJECT_ROOT` to the authorized project layout, `MIF_CONDITION_ROOT` to its `analyses/cd8_condition_targetaware_20261102_v1` directory, and `MIF_RUN_ROOT` to a **new** experiment directory. Copy only scripts and frozen protocol/amendment files to the new run; do not copy completed models/results/markers and do not use publication code hashes to resume original frozen artifacts. Linux and the original recorded environment are required for the full GPU workflow.

Original order: `stage0_run.py --phase pilot`, `--phase full`, `stage0_audit.py`, `zero_label.py --phase dev`, conditional `continuation_run.py --phase cv` and `--phase final`; `active_calibration.py`; then only development-passed external branches and independent audits. Inspect all gates and budget before each new stage. The 28,800-second cap includes new neural inference and training; recorded usage was 5,397.58 seconds. The original pipeline is complete: these commands are documentation, not instructions to rerun it.

`make_figures.py` and `make_final_figures.py` regenerate aggregate scientific figures from the published CSVs. Set `MIF_CJK_FONT` to an installed CJK font if needed. Historical hashes refer to executed server code; publication hashes refer to formatted, relocated source. See [publication notes](PUBLICATION_NOTES.md).

Publication environment limitation: the local PyTorch DLL failed to initialize (WinError1114), so the full17synthetic rerun and integration rerun were not claimed as passed. Original execution passed those checks. Five publication-boundary and five read-only summary checks run locally; the scientific-function AST and all45summaryCSV hashes are independently checked. No additional server transfer was performed.
