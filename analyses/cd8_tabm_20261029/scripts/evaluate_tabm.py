#!/usr/bin/env python3
"""Evaluate frozen TabM/HGB CD8 runs without fitting or discarding failed rows.

Usage: python scripts/evaluate_tabm.py --run-dir /path/to/run
All joins use the explicit four-column cell key. Internal validation is a
selection/development set; external TMA cores are held-out test observations.
Submodel disagreement is descriptive, not calibrated uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

KEYS = ["dataset_id", "image_id", "cell_id", "marker"]
TRAIN, VALID, TEST = "training_WSIs", "internal_val_WSIs", "external_val_TMA_cores"
METRIC_COLUMNS = ["tp", "fp", "fn", "tn", "precision", "recall", "f1", "specificity", "roc_auc", "average_precision", "predicted_positive_fraction", "positive_fraction_absolute_error"]
SUMMARY_COLUMNS = METRIC_COLUMNS + ["true_positive_fraction", "prediction_coverage"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def fit_risk_strata(cells: pd.DataFrame) -> dict[str, Any]:
    """Fit audit-only strata BEFORE test unlock, using labelled training cells."""
    train = cells.loc[(cells["split"] == TRAIN) & cells["ground_truth_binary"].isin([0, 1])]
    contact = train["neighbor_edge_area_fraction"].gt(0) if "neighbor_edge_area_fraction" in train else pd.Series(False, index=train.index)
    definitions = {
        "weak_positive_whole_mean_q25": ("whole_mean", 0.25, train["ground_truth_binary"].eq(1)),
        "contact_bright_diff_q75": ("neighbor_edge_mean_diff", 0.75, contact),
        "local_reference_q75": ("local_reference_median", 0.75, pd.Series(True, index=train.index)),
        "sector_cv_q75": ("sector_signal_cv", 0.75, pd.Series(True, index=train.index)),
    }
    output, counts = {}, {}
    for name, (column, quantile, eligible) in definitions.items():
        if column not in train.columns:
            output[name], counts[name] = None, 0
            continue
        values = pd.to_numeric(train.loc[eligible, column], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        output[name] = float(np.quantile(values, quantile)) if len(values) else None
        counts[name] = int(len(values))
    output["metadata"] = {
        "fit_split": TRAIN, "fit_population": "labelled training cells with finite defining feature", "fit_counts": counts,
        "weak_positive_rule": "true class 1 and whole_mean <= training-positive q25",
        "contact_rule": "neighbor_edge_area_fraction > 0 and neighbor_edge_mean_diff > training-contact q75",
        "interpretation": "Feature-based audit proxies, not verified contamination or pathology labels.",
    }
    return output


def apply_risk_strata(cells: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Apply frozen thresholds; unavailable defining features remain unavailable."""
    label = cells["ground_truth_binary"].to_numpy()
    finite_label = np.isin(label, [0, 1])
    output = {"all_labelled": finite_label}
    definitions = [
        ("weak_positive", "whole_mean", "weak_positive_whole_mean_q25", "le"),
        ("contact_bright_proxy", "neighbor_edge_mean_diff", "contact_bright_diff_q75", "gt"),
        ("high_local_reference_proxy", "local_reference_median", "local_reference_q75", "gt"),
        ("directional_signal_proxy", "sector_signal_cv", "sector_cv_q75", "gt"),
    ]
    for name, column, parameter, operator in definitions:
        threshold = config.get(parameter)
        if threshold is None or column not in cells:
            continue
        values = pd.to_numeric(cells[column], errors="coerce").to_numpy(float)
        eligible = finite_label & np.isfinite(values)
        if name == "weak_positive":
            eligible &= label == 1
        if name == "contact_bright_proxy":
            fraction = pd.to_numeric(cells["neighbor_edge_area_fraction"], errors="coerce").to_numpy(float)
            eligible &= np.isfinite(fraction) & (fraction > 0)
        output[name] = eligible & (values <= threshold if operator == "le" else values > threshold)
        if name != "weak_positive":
            output[name + "_negative"] = output[name] & (label == 0)
            output[name + "_positive"] = output[name] & (label == 1)
    return output


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, Any]:
    """Complete-set metrics: any failed score makes complete-set metrics NaN.

    No true positives: recall/AP NaN. No predicted positives: precision 0.
    Neither true nor predicted positives: F1 NaN. ROC requires both classes.
    """
    y, score = np.asarray(y, dtype=np.int8), np.asarray(score, dtype=float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("classification_metrics received an unlabelled row")
    finite, n = np.isfinite(score), len(y)
    positive = int(np.count_nonzero(y == 1))
    result = {
        "n_cells": n, "n_true_positive": positive, "n_true_negative": n - positive,
        "n_score_finite": int(finite.sum()), "n_score_failed": int((~finite).sum()),
        "prediction_coverage": float(finite.mean()) if n else np.nan,
        "true_positive_fraction": positive / n if n else np.nan,
        "complete_set_metrics_valid": bool(n and finite.all()),
    }
    if not n or not finite.all():
        result.update({column: np.nan for column in METRIC_COLUMNS})
        return result
    predicted = score > threshold
    tp = int(np.count_nonzero(predicted & (y == 1)))
    fp = int(np.count_nonzero(predicted & (y == 0)))
    fn, tn = positive - tp, n - positive - fp
    result.update({
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positive if positive else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if 0 < positive < n else np.nan,
        "average_precision": float(average_precision_score(y, score)) if positive else np.nan,
        "predicted_positive_fraction": (tp + fp) / n,
        "positive_fraction_absolute_error": abs((tp + fp - positive) / n),
    })
    return result


def build_groups(cells: pd.DataFrame) -> list[tuple[dict[str, Any], np.ndarray]]:
    """image_id already identifies each deduplicated WSI or external core."""
    labelled = cells["ground_truth_binary"].isin([0, 1])
    groups = []
    for split in (TRAIN, VALID, TEST):
        indices = np.flatnonzero((cells["split"].eq(split) & labelled).to_numpy())
        groups.append(({"split": split, "group_type": "split", "group_id": split, "context": "all"}, indices))
        part = cells.iloc[indices]
        for (context, unit), frame in part.groupby(["context", "image_id"], sort=True, dropna=False):
            groups.append(({"split": split, "group_type": "image", "group_id": str(unit), "context": str(context)}, frame.index.to_numpy()))
        if split == TEST:
            for context, frame in part.groupby("context", sort=True, dropna=False):
                groups.append(({"split": split, "group_type": "context", "group_id": str(context), "context": str(context)}, frame.index.to_numpy()))
    return groups


def aligned_scores(cells: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    """One-to-one key alignment rejects missing, unexpected and duplicate IDs."""
    if prediction.duplicated(KEYS).any() or prediction[KEYS].isna().any().any():
        raise ValueError("Prediction keys must be complete and unique")
    columns = KEYS + [c for c in prediction if c == "score" or c.startswith("score_") or c.startswith("submodel_")]
    check = cells[KEYS].merge(prediction[KEYS], how="outer", on=KEYS, validate="one_to_one", indicator=True, sort=False)
    if not check["_merge"].eq("both").all() or len(check) != len(cells):
        raise ValueError(f"Prediction/cell key mismatch: {check['_merge'].value_counts().to_dict()}")
    return cells[KEYS].merge(prediction[columns], on=KEYS, how="left", validate="one_to_one", sort=False)


def save_curves(path: Path, y: np.ndarray, score: np.ndarray) -> list[dict[str, Any]]:
    """Save every ROC/PR point and threshold; the PR endpoint has no threshold."""
    if not len(y) or not np.isfinite(score).all():
        return [{"curve": "both", "status": "no_cells_or_failed_scores", "path": ""}]
    rows = []
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(np.unique(y)) == 2:
        fpr, tpr, thresholds = roc_curve(y, score, drop_intermediate=False)
        target = path.with_name(path.name + "_roc.csv")
        pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds}).to_csv(target, index=False)
        rows.append({"curve": "roc", "status": "complete", "path": str(target), "n_points": len(fpr)})
    else:
        rows.append({"curve": "roc", "status": "requires_both_classes", "path": ""})
    if np.any(y == 1):
        precision, recall, thresholds = precision_recall_curve(y, score)
        target = path.with_name(path.name + "_pr.csv")
        pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.r_[thresholds, np.nan], "has_threshold": np.r_[np.ones(len(thresholds), bool), False]}).to_csv(target, index=False)
        rows.append({"curve": "pr", "status": "complete", "path": str(target), "n_points": len(precision)})
    else:
        rows.append({"curve": "pr", "status": "no_true_positive", "path": ""})
    return rows


def descriptive_seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["model_family", "feature_set", "threshold_mode", "split", "group_type", "group_id", "context"]
    rows = []
    for group, frame in metrics.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, group))
        for column in SUMMARY_COLUMNS:
            values = pd.to_numeric(frame[column], errors="coerce")
            rows.append({**base, "metric": column, "n_seeds": int(frame["seed"].nunique()), "n_finite": int(values.notna().sum()), "mean": values.mean(), "std_between_seeds": values.std(ddof=1), "minimum": values.min(), "maximum": values.max()})
    return pd.DataFrame(rows)


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Within-seed feature ablations and same-feature architecture comparisons."""
    comparisons = []
    for family in ("TabM", "HGB"):
        for second, first in (("B", "A"), ("C", "B"), ("C", "A")):
            comparisons.append((f"{family}_{second}_minus_{first}", family, second, family, first))
    for feature_set in ("A", "B", "C"):
        comparisons.append((f"TabM_minus_HGB_{feature_set}", "TabM", feature_set, "HGB", feature_set))
        comparisons.append((f"TabM_{feature_set}_minus_GMM", "TabM", feature_set, "GMM", "GMM"))
    keys = ["seed", "threshold_mode", "split", "group_type", "group_id", "context"]
    if "stratum" in metrics.columns:
        keys.append("stratum")
    rows = []
    for name, family_new, set_new, family_old, set_old in comparisons:
        new = metrics.loc[metrics["model_family"].eq(family_new) & metrics["feature_set"].eq(set_new)]
        old = metrics.loc[metrics["model_family"].eq(family_old) & metrics["feature_set"].eq(set_old)].copy()
        if len(new) == 0 or len(old) == 0:
            continue
        if family_old == "GMM":
            old = old.loc[old["threshold_mode"].eq("fixed_0.5")]
            old = pd.concat([old.assign(seed=seed, threshold_mode=mode) for seed in new["seed"].unique() for mode in new["threshold_mode"].unique()], ignore_index=True)
        merged = new.merge(old, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
        for _, record in merged.iterrows():
            if record["n_cells_new"] != record["n_cells_old"]:
                raise ValueError("Paired comparison cell count mismatch")
            row = {"comparison": name, **{k: record[k] for k in keys}, "n_cells": record["n_cells_new"], "threshold_new": record["threshold_new"], "threshold_old": record["threshold_old"], "reference_threshold_mode": "fixed_0.5" if family_old == "GMM" else record["threshold_mode"]}
            for column in SUMMARY_COLUMNS:
                row[column + "_new"] = record[column + "_new"]
                row[column + "_old"] = record[column + "_old"]
                row["delta_" + column] = record[column + "_new"] - record[column + "_old"]
            rows.append(row)
    return pd.DataFrame(rows)


def missingness_tables(cells: pd.DataFrame, features: list[str], output: Path) -> None:
    """Audit missingness before train-fitted imputation, with unlabelled retained."""
    rows = []
    for (split, unit, label), indices in cells.groupby(["split", "image_id", "ground_truth_binary"], dropna=False, sort=False).groups.items():
        values = cells.loc[indices, features].to_numpy(dtype=float)
        for j, feature in enumerate(features):
            finite = np.isfinite(values[:, j])
            rows.append({"split": split, "unit_id": unit, "true_class": label, "feature": feature, "n_cells": len(values), "n_finite": int(finite.sum()), "n_missing": int((~finite).sum()), "missing_fraction": float((~finite).mean()), "n_zero_finite": int(np.count_nonzero(finite & (values[:, j] == 0)))})
    pd.DataFrame(rows).to_csv(output / "feature_missingness_by_unit_class.csv", index=False)
    qc_columns = [c for c in cells if c.startswith("qc_") or c in ("qc_flags", "context_clipped", "reference_tissue_unknown")]
    qc_rows = []
    for (split, label), frame in cells.groupby(["split", "ground_truth_binary"], dropna=False):
        for column in qc_columns:
            for value, count in frame[column].astype(str).value_counts(dropna=False).items():
                qc_rows.append({"split": split, "true_class": label, "qc_column": column, "qc_value": value, "n_cells": len(frame), "count": int(count), "fraction": count / len(frame)})
    pd.DataFrame(qc_rows, columns=["split", "true_class", "qc_column", "qc_value", "n_cells", "count", "fraction"]).to_csv(output / "qc_by_split_class.csv", index=False)


def evaluate_run(run_dir: Path, skip_curves: bool = False) -> None:
    protocol, risk_path = run_dir / "FROZEN_TEST_PROTOCOL.json", run_dir / "data" / "risk_strata.json"
    if not protocol.is_file() or not risk_path.is_file():
        raise FileNotFoundError("FROZEN_TEST_PROTOCOL.json and risk_strata.json must exist before evaluation")
    frozen = json.loads(protocol.read_text(encoding="utf-8"))
    protocol_hash = sha256(protocol)
    cells_path = run_dir / "data" / "cells.parquet"
    if sha256(cells_path) != frozen["signature"]["data"]:
        raise ValueError("Cell table differs from frozen training/inference data")
    output = run_dir / "results"
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.read_parquet(run_dir / "data" / "cells.parquet").reset_index(drop=True)
    if cells.duplicated(KEYS).any() or cells[KEYS].isna().any().any():
        raise ValueError("Cell keys must be complete and unique")
    groups = build_groups(cells)
    strata = apply_risk_strata(cells, json.loads(risk_path.read_text(encoding="utf-8")))
    y_all = cells["ground_truth_binary"].to_numpy()
    metric_rows, risk_rows, disagreement_rows, curve_rows, source_rows = [], [], [], [], []
    all_features = set()
    prediction_files = sorted((run_dir / "predictions").glob("*.parquet"))
    expected_models = {f"{family}_{feature_set}_s{seed}" for family in ("TabM", "HGB") for feature_set in ("A", "B", "C") for seed in range(3)}
    observed_models = {path.stem for path in prediction_files}
    if observed_models != expected_models:
        raise ValueError(f"Formal evaluation requires all 18 frozen models. Missing={sorted(expected_models - observed_models)}; unexpected={sorted(observed_models - expected_models)}")
    if set(frozen["models"]) != expected_models:
        raise ValueError("Frozen protocol does not contain exactly the 18 expected models")
    models = []
    prediction_hashes = {}
    for path in prediction_files:
        metadata_path = run_dir / "models" / path.stem / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = frozen["models"][path.stem]
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError(f"Metadata differs from frozen protocol: {path.stem}")
        done = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        prediction_hashes[path.stem] = sha256(path)
        if done["frozen_sha256"] != protocol_hash or done["sha256"] != prediction_hashes[path.stem]:
            raise ValueError(f"Prediction is not compatible with the frozen protocol: {path.stem}")
        models.append((path.stem, metadata, path))
        all_features.update(metadata.get("features", []))
    if "GMM_score" not in cells:
        raise ValueError("Frozen GMM_score baseline missing; do not silently omit it")
    models.append(("GMM", {"model_family": "GMM", "feature_set": "GMM", "seed": -1, "threshold_validation": 0.5}, None))
    for model_id, metadata, prediction_path in models:
        print(f"Evaluating {model_id}", flush=True)
        identity = {"model_id": model_id, "model_family": metadata["model_family"], "feature_set": metadata["feature_set"], "seed": int(metadata["seed"])}
        if prediction_path is None:
            prediction = cells[KEYS].copy()
            prediction["score"] = cells["GMM_score"].to_numpy(float)
            if "GMM_prediction" in cells:
                original = pd.to_numeric(cells["GMM_prediction"], errors="coerce")
                mismatch = np.isfinite(prediction["score"]) & original.notna() & original.ne(prediction["score"].gt(0.5).astype(int))
                if mismatch.any():
                    raise ValueError(f"Original GMM strict-threshold mismatch in {int(mismatch.sum())} cells")
            modes = [("fixed_0.5", 0.5)]
        else:
            prediction = aligned_scores(cells, pd.read_parquet(prediction_path))
            threshold = float(metadata["threshold_validation"])
            if not np.isfinite(threshold):
                raise ValueError(f"Invalid frozen threshold for {model_id}")
            modes = [("fixed_0.5", 0.5), ("validation_selected", threshold)]
            source_rows.append({"model_id": model_id, "prediction_sha256": prediction_hashes[model_id], "metadata_sha256": sha256(run_dir / "models" / model_id / "metadata.json")})
        score = prediction["score"].to_numpy(float)
        finite_scores = score[np.isfinite(score)]
        if np.any((finite_scores < 0) | (finite_scores > 1)):
            raise ValueError(f"{model_id}: expected probabilities/scores in [0, 1]")
        members = [c for c in prediction if c.startswith("submodel_")]
        if identity["model_family"] == "TabM" and len(members) != 32:
            raise ValueError(f"{model_id}: expected 32 saved submodel probabilities")
        if members:
            member_score = prediction[members].to_numpy(float)
            valid_members = np.isfinite(member_score).all(axis=1)
            if not np.allclose(score[valid_members], member_score[valid_members].mean(axis=1), rtol=2e-6, atol=2e-7):
                raise ValueError(f"{model_id}: score differs from mean submodel probability")
            disagreement, member_range = np.std(member_score, axis=1, ddof=0), np.ptp(member_score, axis=1)
        else:
            disagreement = member_range = None
        for group, indices in groups:
            y, group_score = y_all[indices].astype(np.int8), score[indices]
            for mode, threshold in modes:
                base = {**identity, **group, "threshold_mode": mode, "threshold": threshold, "decision_rule": "score > threshold", "evaluation_role": "training_fit" if group["split"] == TRAIN else "development_validation" if group["split"] == VALID else "held_out_test"}
                metric_rows.append({**base, **classification_metrics(y, group_score, threshold)})
                for stratum, selector in strata.items():
                    if stratum == "all_labelled":
                        continue
                    chosen = indices[selector[indices]]
                    risk_rows.append({**base, "stratum": stratum, **classification_metrics(y_all[chosen].astype(np.int8), score[chosen], threshold)})
            if disagreement is not None:
                for true_class in ("all", 0, 1):
                    chosen = indices if true_class == "all" else indices[y == true_class]
                    values = disagreement[chosen]
                    valid = np.isfinite(values)
                    summary = {"n_cells": len(chosen), "n_finite": int(valid.sum()), "std_mean": np.mean(values[valid]) if valid.any() else np.nan, "std_median": np.median(values[valid]) if valid.any() else np.nan, "std_p95": np.quantile(values[valid], 0.95) if valid.any() else np.nan, "range_mean": np.mean(member_range[chosen][valid]) if valid.any() else np.nan}
                    disagreement_rows.append({**identity, **group, "true_class": true_class, "n_submodels": len(members), **summary})
            if not skip_curves and group["group_type"] in ("split", "context"):
                stem = run_dir / "curves" / model_id / (group["split"] + "__" + group["group_id"])
                for record in save_curves(stem, y, group_score):
                    if record.get("path"):
                        record["path"] = str(Path(record["path"]).relative_to(run_dir))
                    curve_rows.append({**identity, **group, **record})
        del prediction
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "metrics_all.csv", index=False)
    for mode in ("fixed_0.5", "validation_selected"):
        metrics.loc[metrics["threshold_mode"].eq(mode)].to_csv(output / f"metrics_{mode}.csv", index=False)
    descriptive_seed_summary(metrics).to_csv(output / "metrics_seed_summary.csv", index=False)
    ranked = metrics.loc[metrics.group_type.eq("image")].copy()
    ranked["ascending_f1_rank"] = ranked.groupby(["model_id", "threshold_mode", "split"])["f1"].rank(method="min", ascending=True)
    ranked.sort_values(["model_id", "threshold_mode", "split", "ascending_f1_rank"]).to_csv(output / "unit_performance_ranked.csv", index=False)
    paired_comparisons(metrics).to_csv(output / "paired_comparisons.csv", index=False)
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(output / "risk_strata_metrics.csv", index=False)
    paired_comparisons(risk).to_csv(output / "risk_strata_paired_comparisons.csv", index=False)
    pd.DataFrame(disagreement_rows).to_csv(output / "submodel_disagreement.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "curve_manifest.csv", index=False)
    pd.DataFrame(source_rows).to_csv(output / "prediction_input_hashes.csv", index=False)
    macro_rows = []
    macro_keys = ["model_id", "model_family", "feature_set", "seed", "threshold_mode", "split"]
    for key, frame in metrics.loc[metrics["group_type"].eq("image")].groupby(macro_keys, sort=False):
        base = dict(zip(macro_keys, key))
        for metric in SUMMARY_COLUMNS:
            values = frame[metric].dropna()
            macro_rows.append({**base, "metric": metric, "n_units": len(frame), "n_finite_units": len(values), "unit_equal_mean": values.mean(), "unit_median": values.median(), "unit_minimum": values.min(), "unit_maximum": values.max(), "n_units_no_positive_predictions": int(frame["predicted_positive_fraction"].eq(0).sum()), "interpretation": "descriptive WSI/core equal weighting; cores are not independent patients"})
    pd.DataFrame(macro_rows).to_csv(output / "unit_equal_summary.csv", index=False)
    absent = sorted(all_features.difference(cells.columns))
    if absent:
        raise ValueError(f"Model metadata refers to absent raw features: {absent}")
    missingness_tables(cells, sorted(all_features), output)
    write_json(output / "EVALUATION_MANIFEST.json", {
        "status": "complete", "protocol_sha256": protocol_hash, "risk_strata_sha256": sha256(risk_path),
        "n_all_cells": len(cells), "n_models": len(models), "n_metric_rows": len(metrics),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "curves": "skipped_by_explicit_flag" if skip_curves else "full ROC/PR arrays for splits and TMA contexts",
        "curve_threshold_convention": "sklearn ROC/PR curves use score >= threshold; formal classification tables use the separately recorded strict > rule.",
        "missing_score_policy": "All labelled rows retained; full-set metrics NaN if any score nonfinite.",
        "uncertainty_warning": "Member disagreement is descriptive, not calibrated uncertainty.",
        "statistical_policy": "Seed SD is descriptive, not patient-level confidence intervals; no cell bootstrap or significance test.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--skip-curves", action="store_true", help="Smoke test only; formal delivery requires full curves.")
    args = parser.parse_args()
    evaluate_run(args.run_dir.resolve(), args.skip_curves)


if __name__ == "__main__":
    main()
