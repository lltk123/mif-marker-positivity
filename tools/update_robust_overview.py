"""Append the completed robust-calibration results to the historical overview.

Reads summary CSVs only. Keeps all historical rows, validates the fixed external
population and threshold, and writes a source-linked ranking without refitting.
Run from any directory: python tools/update_robust_overview.py
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "analyses/cd8_robust_comparison_20261030/" "report_tables/MAIN_FIXED_0_5.csv"
OVERVIEW = ROOT / "docs/results_overview.csv"
RANKING = ROOT / "docs/results_rankings_20261031.csv"
SPLIT = "external_val_TMA_cores"
METHOD = "E1_outer_P1_P99"
NEW_PREFIXES = ("TabM_outerP1P99_", "HGB_outerP1P99_", "Nimbus_outerP1P99_")


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    fields, previous = read_rows(OVERVIEW)
    historical = [row for row in previous if not row["model"].startswith(NEW_PREFIXES)]
    _, source = read_rows(ROOT / SOURCE)
    current = [
        row
        for row in source
        if row["variant"] == "new"
        and row["split"] == SPLIT
        and row["level"] == "split"
        and row["threshold_mode"] == "fixed_0.5"
    ]
    expected = {
        (family, features)
        for family, features in [
            ("TabM", "A"),
            ("TabM", "B"),
            ("TabM", "C"),
            ("HGB", "A"),
            ("HGB", "B"),
            ("HGB", "C"),
            ("Nimbus", "pretrained"),
            ("Nimbus", "finetuned"),
        ]
    }
    if (
        len(current) != 8
        or {(row["model_family"], row["feature_set"]) for row in current} != expected
    ):
        raise ValueError("Expected exactly eight completed model/feature groups")

    mapping = {
        "precision": "precision_mean",
        "recall": "recall_mean",
        "F1": "f1_mean",
        "ROC_AUC": "roc_auc_mean",
        "AP": "average_precision_mean",
        "TP": "tp_mean",
        "FP": "fp_mean",
        "FN": "fn_mean",
        "TN": "tn_mean",
        "F1_seed_sd": "f1_std",
        "AP_seed_sd": "average_precision_std",
    }
    additions = []
    for source_row in sorted(
        current, key=lambda r: (r["model_family"], r["feature_set"])
    ):
        family, features = source_row["model_family"], source_row["feature_set"]
        seeds = 1 if family == "Nimbus" else 3
        if not (
            source_row["normalization"] == METHOD
            and float(source_row["n_cells_mean"]) == 437480
            and float(source_row["threshold_mean"]) == 0.5
            and int(source_row["n_seeds"]) == seeds
            and source_row["all_seeds_complete"] == "True"
            and source_row["all_calibration_valid"] == "True"
            and float(source_row["prediction_coverage_mean"]) == 1.0
        ):
            raise ValueError(f"Incomplete or incompatible results: {family}/{features}")
        name = f"{family}_outerP1P99_{features}"
        if seeds == 3:
            name += "_mean_3_seeds"
        row = dict.fromkeys(fields, "")
        row.update({key: source_row[value] for key, value in mapping.items()})
        row.update(
            model=name,
            split=SPLIT,
            threshold="0.5",
            n_labelled="437480",
            positive_rule="score > threshold"
            + ("; metric mean over 3 seeds" if seeds == 3 else ""),
            n_model_runs=str(seeds),
            source_csv=SOURCE,
            source_filter=(
                f"variant=new;split={SPLIT};level=split;threshold_mode=fixed_0.5;"
                f"model_family={family};feature_set={features};normalization={METHOD};"
                "TP/FP/FN/TN are mean counts when n_model_runs=3"
            ),
        )
        additions.append(row)

    rows = historical + additions
    if len({row["model"] for row in rows}) != len(rows):
        raise ValueError("Duplicate model names")
    write_rows(OVERVIEW, fields, rows)
    ranked = sorted(rows, key=lambda row: (-float(row["F1"]), row["model"]))
    write_rows(
        RANKING,
        ["rank_by_F1", *fields],
        [dict(rank_by_F1=i, **row) for i, row in enumerate(ranked, 1)],
    )
    print(f"Preserved {len(historical)} historical rows; added {len(additions)} rows.")
    print(f"Ranking: {RANKING}")


if __name__ == "__main__":
    main()
