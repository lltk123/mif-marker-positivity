"""Evaluate externally generated predictions by explicit keys. Does not train."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from baseline_contract import (
    KEYS,
    exact_join,
    binary_metrics,
    predictions_from_scores,
    key_hash,
)


def validate_spec(spec):
    required = [
        "model_name",
        "model_version",
        "paper_reference",
        "official_repository",
        "code_revision",
        "weights_sha256",
        "threshold",
        "threshold_operator",
        "threshold_source",
        "frozen_at",
    ]
    if spec.get("status") != "frozen" or any(
        spec.get(x) in [None, "", "pending"] for x in required
    ):
        raise ValueError(
            "Paper, official implementation, weights and threshold must be specified and frozen first"
        )
    if spec["threshold_source"] not in [
        "official_fixed",
        "training_grouped_oof",
        "grouped_development_validation",
    ]:
        raise ValueError("Threshold source must exclude held-out evaluation labels")
    if len(spec["weights_sha256"]) != 64 or any(
        c not in "0123456789abcdef" for c in spec["weights_sha256"]
    ):
        raise ValueError("Invalid weights SHA256")
    predictions_from_scores([], float(spec["threshold"]), spec["threshold_operator"])


def read_table(path):
    return (
        pd.read_parquet(path)
        if str(path).endswith(".parquet")
        else pd.read_csv(path, float_precision="round_trip")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--model-spec", type=Path, required=True)
    p.add_argument("--split", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--export-cell-csv", action="store_true")
    a = p.parse_args()
    spec = json.loads(a.model_spec.read_text())
    validate_spec(spec)
    manifest = read_table(a.manifest)
    if set(a.split) - set(manifest.split):
        raise ValueError("Unknown split requested")
    manifest = manifest[manifest.split.isin(a.split)].copy()
    pred = read_table(a.predictions)
    if set(pred.columns) != set(KEYS + ["score"]):
        raise ValueError("Predictions must contain exactly the four keys and score")
    joined = exact_join(manifest, pred)
    joined["score"] = pd.to_numeric(joined.score, errors="raise")
    joined["prediction"] = predictions_from_scores(
        joined.score, spec["threshold"], spec["threshold_operator"]
    )
    joined["model_version"] = spec["model_version"]
    joined["model_name"] = spec["model_name"]
    joined["threshold"] = spec["threshold"]
    joined["threshold_operator"] = spec["threshold_operator"]
    joined["score_finite"] = np.isfinite(joined.score)
    labelled = joined[joined.ground_truth_binary.isin([0, 1])]
    rows = []
    for split, g in labelled.groupby("split", sort=True):
        groups = [("split", split, g)]
        groups += [("slide", str(k), v) for k, v in g.groupby("slide_id", sort=True)]
        groups += [("image", str(k), v) for k, v in g.groupby("image_id", sort=True)]
        for level, group, values in groups:
            m = binary_metrics(
                values.ground_truth_binary,
                values.score,
                spec["threshold"],
                spec["threshold_operator"],
            )
            m.update(
                split=split,
                level=level,
                group=group,
                model_version=spec["model_version"],
                threshold=spec["threshold"],
                threshold_operator=spec["threshold_operator"],
            )
            rows.append(m)
    a.output.mkdir(parents=True, exist_ok=False)
    joined.to_parquet(
        a.output / "cell_predictions.parquet", index=False, compression="zstd"
    )
    if a.export_cell_csv:
        joined.to_csv(a.output / "cell_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(a.output / "metrics.csv", index=False)
    (a.output / "EVALUATION_CONFIG.json").write_text(
        json.dumps(
            dict(
                model_spec=spec,
                model_spec_sha256=hashlib.sha256(a.model_spec.read_bytes()).hexdigest(),
                evaluation_key_sha256=key_hash(manifest),
                splits=a.split,
                no_positive_policy="Precision/recall/F1 use zero_division=0; AP and ROC-AUC are NaN without positives; ROC-AUC also NaN without negatives",
                failed_scores_policy="Retain row, report coverage; full-population performance undefined if any labelled row cannot score",
                scope="Overall within each official split, per slide and per image; unknown labels retained but excluded from metrics",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
