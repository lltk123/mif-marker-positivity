"""Exercise prediction-file IO using explicit synthetic scores, never a paper model."""
from pathlib import Path
import argparse, json, subprocess, sys
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    keys = ["dataset_id", "image_id", "cell_id", "marker"]
    cells = pd.DataFrame(
        dict(
            dataset_id=["SYNTHETIC"] * 4,
            image_id=["WSI_fixture"] * 4,
            cell_id=[11, 37, 93, 101],
            marker=["CD8"] * 4,
            split=["development_fixture"] * 4,
            slide_id=["slide_fixture"] * 4,
            ground_truth_binary=[0.0, 1.0, 0.0, np.nan],
        )
    )
    cells.to_parquet(a.output / "synthetic_cells.parquet", index=False)
    pred = cells[keys].iloc[[2, 0, 3, 1]].copy()
    pred["score"] = [0.1, 0.2, np.nan, 0.9]
    pred.to_parquet(a.output / "synthetic_scores.parquet", index=False)
    spec = dict(
        status="frozen",
        model_name="SYNTHETIC_IO_TEST_NOT_MODEL",
        model_version="fixture_v1",
        paper_reference="SYNTHETIC_TEST_NOT_A_PAPER",
        official_repository="synthetic_fixture",
        code_revision="synthetic_fixture",
        weights_sha256="0" * 64,
        threshold=0.5,
        threshold_operator="gt",
        threshold_source="official_fixed",
        frozen_at="synthetic_test_only",
    )
    (a.output / "synthetic_spec.json").write_text(json.dumps(spec))

    def run(scorefile, out):
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("evaluate_predictions.py")),
            "--manifest",
            str(a.output / "synthetic_cells.parquet"),
            "--predictions",
            str(scorefile),
            "--model-spec",
            str(a.output / "synthetic_spec.json"),
            "--split",
            "development_fixture",
            "--output",
            str(out),
            "--export-cell-csv",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout + "\n" + result.stderr)

    run(a.output / "synthetic_scores.parquet", a.output / "synthetic_complete")
    data = pd.read_parquet(
        a.output / "synthetic_complete/cell_predictions.parquet"
    ).set_index("cell_id")
    metrics = pd.read_csv(a.output / "synthetic_complete/metrics.csv")
    checks = [
        ("shuffled_prediction_rows_align_by_id", data.loc[37, "score"] == 0.9),
        (
            "unlabelled_cell_retained",
            len(data) == 4 and pd.isna(data.loc[101, "prediction"]),
        ),
        ("labelled_metrics_only", (metrics.n_total == 3).all()),
        (
            "pooled_slide_image_metrics",
            set(metrics.level) == {"split", "slide", "image"},
        ),
        ("known_synthetic_F1", (metrics.F1 == 1).all()),
        (
            "cell_csv_exported",
            (a.output / "synthetic_complete/cell_predictions.csv").exists(),
        ),
    ]
    pred.loc[pred.cell_id == 37, "score"] = np.nan
    pred.to_parquet(a.output / "synthetic_failed_score.parquet", index=False)
    run(a.output / "synthetic_failed_score.parquet", a.output / "synthetic_incomplete")
    failed = pd.read_csv(a.output / "synthetic_incomplete/metrics.csv")
    checks.append(
        (
            "failed_labelled_score_kept_and_full_metrics_undefined",
            (failed.n_failed == 1).all()
            and failed.F1.isna().all()
            and np.allclose(failed.prediction_coverage, 2 / 3),
        )
    )
    result = pd.DataFrame(
        [
            dict(
                check=k, passed=bool(v), scope="synthetic_IO_only_not_model_performance"
            )
            for k, v in checks
        ]
    )
    result.to_csv(a.output / "synthetic_cli_checks.csv", index=False)
    if not result.passed.all():
        raise AssertionError(result)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
