"""Compare completed runs, preserving full populations and provenance.

This module reads saved metrics and completion records only. It never loads
images, fits models, changes thresholds, or accesses Pan-M.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

METHOD = "E1_outer_P1_P99"
EXPECTED = {
    "training_WSIs": 101057,
    "internal_val_WSIs": 539132,
    "external_val_TMA_cores": 437480,
}
IMAGE_COUNTS = {
    "training_WSIs": 3,
    "internal_val_WSIs": 3,
    "external_val_TMA_cores": 42,
}
MODES = {"fixed_0.5", "validation_selected"}
METRICS = [
    "precision",
    "recall",
    "f1",
    "specificity",
    "roc_auc",
    "average_precision",
    "tp",
    "fp",
    "fn",
    "tn",
    "true_positive_fraction",
    "predicted_positive_fraction",
    "positive_fraction_absolute_error",
    "prediction_coverage",
    "n_score_finite",
    "n_score_failed",
    "n_calibration_invalid",
    "calibration_coverage",
    "threshold",
    "n_cells",
]
FORMAL = [
    "precision",
    "recall",
    "f1",
    "specificity",
    "roc_auc",
    "average_precision",
    "predicted_positive_fraction",
    "positive_fraction_absolute_error",
]
GROUPS = [
    "variant",
    "model_family",
    "feature_set",
    "normalization",
    "split",
    "level",
    "unit",
    "context",
    "threshold_mode",
]
UNIT_KEYS = ["split", "level", "unit", "context"]
PAIR_KEYS = ["model_family", "feature_set", "seed"] + UNIT_KEYS + ["threshold_mode"]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError("Required completed-run record missing: " + str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def key_set(frame: pd.DataFrame, columns: list[str]) -> set:
    return set(frame[columns].itertuples(index=False, name=None))


def add_calibration_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for name in ["n_calibration_invalid", "calibration_coverage"]:
        if name not in frame:
            frame[name] = np.nan
    return frame


def read_tabm(path: Path, variant: str) -> pd.DataFrame:
    frame = read_csv(path).rename(columns={"group_type": "level", "group_id": "unit"})
    frame["variant"] = variant
    frame["normalization"] = (
        METHOD if variant == "new" else "raw_train_quantile_transform"
    )
    frame["source_csv"] = str(path)
    if variant == "new":
        frame = frame.loc[frame.model_family.ne("GMM")].copy()
        if (
            "normalization_method" not in frame
            or not frame.normalization_method.eq(METHOD).all()
        ):
            raise ValueError(
                "New TabM/HGB metrics lack the prescribed normalization identity."
            )
        for column in ["n_calibration_invalid", "calibration_coverage"]:
            if column not in frame:
                raise ValueError("New TabM/HGB calibration coverage missing: " + column)
    else:
        frame.loc[frame.model_family.eq("GMM"), "normalization"] = "original_GMM"
        frame.loc[frame.model_family.eq("GMM"), "variant"] = "baseline"
    return add_calibration_columns(frame)


def read_nimbus(path: Path, variant: str) -> pd.DataFrame:
    frame = read_csv(path)
    frame = frame.loc[frame.model.ne("GMM")].copy()
    frame = frame.rename(
        columns={
            "model": "model_id",
            "n": "n_cells",
            "n_scored": "n_score_finite",
            "n_failed": "n_score_failed",
            "coverage": "prediction_coverage",
            "TP": "tp",
            "FP": "fp",
            "FN": "fn",
            "TN": "tn",
            "F1": "f1",
            "ROC_AUC": "roc_auc",
            "AP": "average_precision",
        }
    )
    if variant == "new":
        coverage_path = path.parent / "calibration_coverage.csv"
        coverage = read_csv(coverage_path)
        keys = ["split", "level", "context", "image_id"]
        needed = keys + ["n_cells", "n_calibration_invalid", "calibration_coverage"]
        if not set(needed).issubset(coverage):
            raise ValueError("Nimbus calibration coverage schema is incomplete.")
        if coverage.duplicated(keys).any():
            raise ValueError("Duplicate Nimbus calibration coverage unit.")
        frame = frame.drop(
            columns=["n_calibration_invalid", "calibration_coverage"], errors="ignore"
        )
        frame = frame.merge(
            coverage[needed],
            on=keys,
            how="left",
            validate="many_to_one",
            suffixes=("", "_calibration"),
            indicator=True,
        )
        if not frame["_merge"].eq("both").all():
            raise ValueError("Nimbus metric unit lacks calibration audit.")
        if (
            not pd.to_numeric(frame.n_cells)
            .eq(pd.to_numeric(frame.n_cells_calibration))
            .all()
        ):
            raise ValueError(
                "Nimbus labelled calibration and metric populations differ."
            )
        frame = frame.drop(columns=["_merge", "n_cells_calibration"])
    frame["variant"], frame["model_family"] = variant, "Nimbus"
    frame["feature_set"] = np.where(
        frame.model_id.str.contains("finetuned"), "finetuned", "pretrained"
    )
    frame["seed"] = 0
    frame["normalization"] = (
        METHOD if variant == "new" else "training_factor_17.2643260510762"
    )
    frame["level"] = frame.level.replace({"slide": "context"})
    frame["unit"] = np.where(
        frame.level.eq("split"),
        frame.split,
        np.where(frame.level.eq("context"), frame.context, frame.image_id),
    )
    frame["threshold_mode"] = frame.threshold_mode.replace(
        {
            "official_default": "fixed_0.5",
            "validation_threshold": "validation_selected",
        }
    )
    frame["source_csv"] = str(path)
    return add_calibration_columns(frame)


def expected_model_grid() -> list[tuple]:
    grid = []
    for variant in ["old", "new"]:
        for family in ["TabM", "HGB"]:
            for feature_set in "ABC":
                for seed in [0, 1, 2]:
                    for mode in sorted(MODES):
                        grid.append((variant, family, feature_set, seed, mode))
        for feature_set in ["pretrained", "finetuned"]:
            for mode in sorted(MODES):
                grid.append((variant, "Nimbus", feature_set, 0, mode))
    grid.append(("baseline", "GMM", "GMM", -1, "fixed_0.5"))
    return grid


def assert_complete(frame: pd.DataFrame) -> None:
    """Validate every seed/unit/mode, not only a global union of seed numbers."""
    identity = ["variant", "model_family", "feature_set", "seed", "threshold_mode"]
    keys = identity + UNIT_KEYS
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate metric keys across combined sources.")
    for field in METRICS + ["seed"]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if key_set(frame, identity) != set(expected_model_grid()):
        raise ValueError(
            "Missing or unexpected model/feature/seed/threshold combinations, including GMM."
        )
    if not set(frame.split).issubset(EXPECTED) or not set(frame.level).issubset(
        {"split", "image", "context"}
    ):
        raise ValueError("Unexpected split or metric level.")
    for name in ["seed", "n_cells", "n_score_finite", "n_score_failed"]:
        values = frame[name].to_numpy(float)
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError("Non-integer or missing count/seed: " + name)
    if (
        not frame.n_cells.gt(0).all()
        or not frame.n_score_finite.ge(0).all()
        or not frame.n_score_failed.ge(0).all()
    ):
        raise ValueError("Invalid population or failed/scored counts.")

    baseline = frame.loc[frame.model_family.eq("GMM")]
    reference_images = key_set(
        baseline.loc[baseline.level.eq("image")], ["split", "unit"]
    )
    reference_tma = key_set(
        baseline.loc[baseline.level.eq("context")], ["split", "unit"]
    )
    for model, part in frame.groupby(identity, dropna=False, sort=False):
        family = model[1]
        if set(part.split) != set(EXPECTED):
            raise ValueError("Missing split for model/seed/mode: " + str(model))
        for split, expected_n in EXPECTED.items():
            unit_rows = part.loc[part.split.eq(split)]
            profile = unit_rows.groupby("level").size().to_dict()
            wanted = {"split": 1, "image": IMAGE_COUNTS[split]}
            if family == "Nimbus" or split == "external_val_TMA_cores":
                wanted["context"] = 3
            if profile != wanted:
                raise ValueError(
                    "Incomplete unit grid: " + str((model, split, profile, wanted))
                )
            total = unit_rows.loc[unit_rows.level.eq("split")]
            if not total.n_cells.eq(expected_n).all() or not total.unit.eq(split).all():
                raise ValueError(
                    "Incorrect complete split population: " + str((model, split))
                )
            for level in ["image", "context"]:
                selected = unit_rows.loc[unit_rows.level.eq(level)]
                if len(selected) and selected.n_cells.sum() != expected_n:
                    raise ValueError(
                        "Unit populations do not partition the split: "
                        + str((model, split, level))
                    )
        if (
            key_set(part.loc[part.level.eq("image")], ["split", "unit"])
            != reference_images
        ):
            raise ValueError("Image-unit keys differ across models or seeds.")
        ext = part.loc[
            part.level.eq("context") & part.split.eq("external_val_TMA_cores")
        ]
        if key_set(ext, ["split", "unit"]) != reference_tma:
            raise ValueError("External TMA context keys differ across models.")
        if (
            part.threshold.nunique(dropna=False) != 1
            or not part.threshold.between(0, 1, inclusive="neither").all()
        ):
            raise ValueError("Expected one valid global threshold per model/seed/mode.")
        if model[-1] == "fixed_0.5" and not part.threshold.eq(0.5).all():
            raise ValueError("Fixed threshold was changed.")
    newer = frame.loc[frame.variant.eq("new")]
    older = frame.loc[frame.variant.eq("old")]
    if key_set(newer, PAIR_KEYS) != key_set(older, PAIR_KEYS):
        raise ValueError(
            "Old and new unit/seed/mode key sets differ in either direction."
        )
    pair = newer.merge(
        older, on=PAIR_KEYS, suffixes=("_new", "_old"), validate="one_to_one"
    )
    if not pair.n_cells_new.eq(pair.n_cells_old).all():
        raise ValueError("Old and new paired populations differ.")

    if not (frame.n_score_finite + frame.n_score_failed).eq(frame.n_cells).all():
        raise ValueError("Successful and failed counts do not cover the population.")
    if not np.allclose(
        frame.prediction_coverage,
        frame.n_score_finite / frame.n_cells,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("Coverage does not agree with the scored population.")
    calibration = frame.variant.eq("new")
    for field in ["n_calibration_invalid", "calibration_coverage"]:
        if not np.isfinite(frame.loc[calibration, field]).all():
            raise ValueError(
                "New calibrated runs require explicit finite calibration audit: "
                + field
            )
    invalid = frame.loc[calibration, "n_calibration_invalid"]
    population = frame.loc[calibration, "n_cells"]
    if not (
        (invalid >= 0) & (invalid <= population) & (invalid == np.floor(invalid))
    ).all():
        raise ValueError("Invalid calibration failure counts.")
    if not np.allclose(
        frame.loc[calibration, "calibration_coverage"],
        1 - invalid / population,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError(
            "Calibration coverage does not match its own labelled population."
        )
    complete = frame.n_score_failed.eq(0)
    counts = frame[["tp", "fp", "fn", "tn"]].sum(axis=1, min_count=4)
    if not counts[complete].eq(frame.loc[complete, "n_cells"]).all():
        raise ValueError(
            "Confusion counts do not cover the stated complete population."
        )
    for field in ["f1", "precision", "recall", "roc_auc", "average_precision"]:
        if not np.isfinite(frame.loc[complete & frame.level.eq("split"), field]).all():
            raise ValueError("Missing primary complete-split metric: " + field)
    frame["confusion_count_scope"] = np.where(
        complete, "complete_population", "scored_subset_or_unavailable"
    )
    frame["calibration_audit_status"] = np.where(
        calibration, "explicit_outer_calibration", "outer_calibration_not_applicable"
    )
    # A finite HGB/TabM output remains a formal score even if image calibration
    # failed. Keep the separate calibration audit rather than dropping that row.
    frame.loc[~complete, FORMAL] = np.nan


def verify_tabm_completion(
    run: Path, variant: str, require_delivery: bool
) -> list[Path]:
    files = [
        run / "PREDICTIONS_COMPLETE.json",
        run / "results/EVALUATION_MANIFEST.json",
        run / "FROZEN_TEST_PROTOCOL.json",
    ]
    pred, evaluation, frozen = [read_json(p) for p in files]
    frozen_sha = sha(files[2])
    if (
        pred.get("models") != 18
        or pred.get("n_per_model") != 1817783
        or pred.get("frozen_sha256") != frozen_sha
    ):
        raise ValueError("TabM predictions are not complete or frozen: " + str(run))
    expected_ids = {
        f"{family}_{group}_s{seed}"
        for family in ["TabM", "HGB"]
        for group in "ABC"
        for seed in range(3)
    }
    if set(frozen.get("models", {})) != expected_ids:
        raise ValueError("TabM frozen models are incomplete.")
    if (
        evaluation.get("status") != "complete"
        or evaluation.get("n_models") != 19
        or evaluation.get("n_all_cells") != 1817783
        or evaluation.get("n_metric_rows") != 1998
        or evaluation.get("protocol_sha256") != frozen_sha
        or not evaluation.get("curves", "").startswith("full ROC/PR")
    ):
        raise ValueError("TabM full evaluation has not completed: " + str(run))
    if variant == "new" and any(
        d.get("normalization_method") != METHOD for d in [evaluation, frozen]
    ):
        raise ValueError("New TabM completion records have another normalization.")
    if require_delivery:
        delivery_path = run / "DELIVERY_VALIDATION.json"
        delivery = read_json(delivery_path)
        if (
            delivery.get("status") != "complete"
            or delivery.get("n_models") != 18
            or delivery.get("n_cells_per_model") != 1817783
        ):
            raise ValueError("TabM delivery validation is incomplete.")
        files.append(delivery_path)
    return files


def assert_same_metric_rows(
    combined: pd.DataFrame, individual: pd.DataFrame, model: str
) -> None:
    keys = ["model", "split", "level", "context", "image_id", "threshold_mode"]
    chosen = combined.loc[combined.model.eq(model)]
    if len(chosen) != len(individual) or not set(individual.columns).issubset(chosen):
        raise ValueError(
            "Nimbus concatenated metrics omit rows or columns for " + model
        )
    if chosen.duplicated(keys).any() or individual.duplicated(keys).any():
        raise ValueError("Nimbus duplicate metric row.")
    a = chosen[individual.columns].sort_values(keys).reset_index(drop=True)
    b = individual.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        a, b, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14
    )


def verify_nimbus_delivery(run: Path, variant: str) -> list[Path]:
    """Accept the documented legacy completion contract only for the old run."""
    marker = run / "provenance/delivery_complete.json"
    delivery = read_json(marker)
    if variant == "new":
        if delivery.get("status") not in {"complete", "complete_with_unscorable_cells"}:
            raise ValueError("New Nimbus delivery is incomplete.")
        return [marker]
    if variant != "old":
        raise ValueError("Unknown Nimbus run variant.")
    completed = delivery.get("completed")
    if not isinstance(completed, str) or "T" not in completed:
        raise ValueError("Legacy Nimbus delivery lacks its completion timestamp.")
    try:
        datetime.fromisoformat(completed)
    except ValueError as error:
        raise ValueError("Invalid legacy Nimbus completion timestamp.") from error
    required = {
        "numeric_checks_passed": True,
        "visual_review_complete": True,
        "PanM_accessed": False,
        "model_count_new": 2,
        "n_cells_per_model": 1817783,
        "n_units": 48,
    }
    for key, value in required.items():
        actual = delivery.get(key)
        if actual != value or (isinstance(value, bool) and actual is not value):
            raise ValueError("Legacy Nimbus delivery is incomplete: " + key)
    evidence = [
        "results/final_delivery_checks.csv",
        "results/GMM_reuse_audit.csv",
        "results/prediction_integrity.csv",
        "figures/FIGURE_QA.json",
    ]
    names = delivery.get("evidence")
    if (
        not isinstance(names, list)
        or len(names) != len(evidence)
        or set(names) != set(evidence)
    ):
        raise ValueError("Legacy Nimbus delivery evidence list is incomplete.")
    paths = [run / name for name in evidence]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                "Legacy Nimbus delivery evidence missing: " + str(path)
            )
    # Bind the actual legacy evidence files into the comparison signature.
    return [marker] + paths


def verify_nimbus_completion(
    run: Path, variant: str, require_delivery: bool
) -> list[Path]:
    prefix = "Nimbus_outerP1P99_" if variant == "new" else "Nimbus_"
    combined = read_csv(run / "results/all_metrics.csv")
    files = [run / "NIMBUS_PROTOCOL.json"]
    protocol_hash = sha(files[0])
    allowed = {"GMM"}
    for kind in ["pretrained", "finetuned"]:
        model = prefix + kind
        allowed.add(model)
        marker = run / "provenance" / (model + "_evaluation_complete.json")
        record = read_json(marker)
        signature = record.get("signature", {})
        if (
            record.get("all_cells") != 1817783
            or signature.get("model") != model
            or signature.get("protocol_sha256") != protocol_hash
        ):
            raise ValueError(
                "Nimbus per-model evaluation is incomplete or incompatible: " + model
            )
        metric_file = run / "results" / (model + "_metrics.csv")
        hashes = [
            value
            for name, value in record.get("output_sha256", {}).items()
            if PurePosixPath(name.replace("\\", "/")).name == metric_file.name
        ]
        if len(hashes) != 1 or hashes[0] != sha(metric_file):
            raise ValueError("Nimbus authoritative metric hash mismatch: " + model)
        assert_same_metric_rows(combined, read_csv(metric_file), model)
        files.extend([marker, metric_file])
    if not set(combined.model).issubset(allowed):
        raise ValueError("Unexpected Nimbus model in concatenated metrics.")
    if variant == "new":
        marker = run / "provenance/calibration_coverage_complete.json"
        coverage_record = read_json(marker)
        table = run / "results/calibration_coverage.csv"
        if (
            coverage_record.get("status") != "complete"
            or coverage_record.get("n_all_cells") != 1817783
            or coverage_record.get("table_sha256") != sha(table)
        ):
            raise ValueError(
                "Nimbus calibration coverage audit is incomplete or changed."
            )
        files.extend([marker, table])
    if require_delivery:
        files.extend(verify_nimbus_delivery(run, variant))
    return files


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    agg = frame.groupby(GROUPS, dropna=False, sort=True)[METRICS].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    summary = agg.reset_index()
    sizes = (
        frame.groupby(GROUPS, dropna=False).seed.nunique().reset_index(name="n_seeds")
    )
    summary = summary.merge(sizes, on=GROUPS, validate="one_to_one")
    summary["expected_seeds"] = np.where(
        summary.model_family.isin(["TabM", "HGB"]), 3, 1
    )
    summary["seed_matrix_complete"] = summary.n_seeds.eq(summary.expected_seeds)
    summary[
        "all_seeds_complete"
    ] = summary.seed_matrix_complete & summary.prediction_coverage_mean.eq(1)
    summary["all_calibration_valid"] = np.where(
        summary.variant.eq("new"), summary.calibration_coverage_mean.eq(1), None
    )
    for metric in FORMAL:
        summary.loc[
            ~summary.all_seeds_complete, [metric + "_mean", metric + "_std"]
        ] = np.nan
    return summary


def cache_compatible(previous: dict, signature: dict) -> bool:
    return all(previous.get(key) == value for key, value in signature.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "RETRAIN_PROTOCOL.json",
    )
    parser.add_argument(
        "--require-delivery",
        action="store_true",
        help="Also require completed per-run reports/delivery checks.",
    )
    args = parser.parse_args()
    base = args.project_root / "analyses"
    sources = [
        ("tabm", "old", base / "cd8_tabm_20261029/results/metrics_all.csv"),
        ("tabm", "new", base / "cd8_tabm_robust_20261030/results/metrics_all.csv"),
        ("nimbus", "old", base / "cd8_nimbus_20261024/results/all_metrics.csv"),
        ("nimbus", "new", base / "cd8_nimbus_robust_20261030/results/all_metrics.csv"),
    ]
    missing = [str(p) for _, _, p in sources if not p.is_file()]
    if missing:
        raise SystemExit(
            "New experiments are not complete. Missing:\n" + "\n".join(missing)
        )
    manifests = []
    for family, variant, path in sources:
        verify = (
            verify_tabm_completion if family == "tabm" else verify_nimbus_completion
        )
        manifests.extend(verify(path.parent.parent, variant, args.require_delivery))
    read_json(args.protocol)
    signature = {
        "source_sha256": {str(p): sha(p) for _, _, p in sources},
        "completion_manifest_sha256": {str(p): sha(p) for p in manifests},
        "comparison_code_sha256": sha(Path(__file__)),
        "protocol_sha256": sha(args.protocol),
        "require_delivery": args.require_delivery,
    }
    frame = pd.concat(
        [
            (read_tabm if kind == "tabm" else read_nimbus)(path, variant)
            for kind, variant, path in sources
        ],
        ignore_index=True,
    )
    assert_complete(frame)
    output = args.output_dir
    marker = output / "COMPARISON_COMPLETE.json"
    if marker.exists():
        previous = read_json(marker)
        if not cache_compatible(previous, signature):
            raise RuntimeError(
                "Changed source/code/protocol/completion records; use a fresh comparison output."
            )
        for name, expected_hash in previous["output_sha256"].items():
            if sha(output / name) != expected_hash:
                raise RuntimeError("Existing comparison output changed: " + name)
        print("Compatible completed comparison retained")
        return
    names = [
        "MODEL_COMPARISON.csv",
        "SUMMARY_METRICS.csv",
        "PAIRED_CHANGES.csv",
        "RANKINGS_FIXED_0.5.csv",
    ]
    if any((output / name).exists() for name in names):
        raise RuntimeError(
            "Partial comparison output exists; use a new output directory."
        )
    summary = build_summary(frame)
    match = ["model_family", "feature_set"] + UNIT_KEYS + ["threshold_mode"]
    newer, older = (
        summary.loc[summary.variant.eq("new")],
        summary.loc[summary.variant.eq("old")],
    )
    if key_set(newer, match) != key_set(older, match):
        raise ValueError("Summary old/new keys differ in either direction.")
    paired = newer.merge(
        older, on=match, suffixes=("_new", "_old"), validate="one_to_one"
    )
    for metric in METRICS:
        paired[metric + "_delta"] = (
            paired[metric + "_mean_new"] - paired[metric + "_mean_old"]
        )
    ranked = summary.loc[
        summary.threshold_mode.eq("fixed_0.5")
        & summary.split.isin(["internal_val_WSIs", "external_val_TMA_cores"])
        & summary.level.isin(["split", "context"])
    ].copy()
    ranked["rank_eligible"] = ranked.all_seeds_complete & ranked.f1_mean.notna()
    ranked["rank_f1"] = np.nan
    eligible = ranked.loc[ranked.rank_eligible]
    ranked.loc[eligible.index, "rank_f1"] = eligible.groupby(
        ["split", "level", "unit"]
    ).f1_mean.rank(ascending=False, method="min")
    output.mkdir(parents=True, exist_ok=True)
    frame[
        GROUPS
        + ["seed", "model_id"]
        + METRICS
        + ["confusion_count_scope", "calibration_audit_status", "source_csv"]
    ].to_csv(output / names[0], index=False)
    summary.to_csv(output / names[1], index=False)
    paired.to_csv(output / names[2], index=False)
    ranked.sort_values(["split", "level", "unit", "rank_f1"]).to_csv(
        output / names[3], index=False
    )
    marker.write_text(
        json.dumps(
            {
                "status": "complete"
                if frame.n_score_failed.eq(0).all()
                else "completed_with_scoring_failures",
                **signature,
                "output_sha256": {name: sha(output / name) for name in names},
                "n_metric_rows": len(frame),
                "n_summary_rows": len(summary),
                "n_paired_rows": len(paired),
                "count_interpretation": "Summary counts are seed means, not pooled cells. Failed-row confusion counts are scored-subset or unavailable; formal rates are suppressed.",
                "calibration_interpretation": "Finite prediction coverage and image calibration coverage are distinct; calibration-failed cells are retained.",
                "new_models": "TabM/HGB A/B/C three seeds; Nimbus pretrained and finetuned one seed",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metric_rows": len(frame), "paired_rows": len(paired)}))


if __name__ == "__main__":
    main()
