"""Unified locked evaluation and isolated target-label calibration simulation."""
import os

os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import json, hashlib, time
import numpy as np, pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
)
from experiment_common import *
from execution_guards import native_threshold

GROUPS = [
    "image",
    "image_target",
    "image_target_other",
    "tabular_only",
    "image_target_other_tabular",
]


def record_metrics(y, s, t, rank=None, probability=True):
    y = np.asarray(y, int)
    s = np.asarray(s)
    assert np.isfinite(s).all()
    n = len(y)
    p = s > np.asarray(t, dtype=s.dtype)
    tp = int(np.sum((y == 1) & p))
    fp = int(np.sum((y == 0) & p))
    fn = int(np.sum((y == 1) & ~p))
    tn = int(np.sum((y == 0) & ~p))
    positive = tp + fn
    negative = tn + fp
    z = {
        "n_cells": n,
        "n_score_finite": n,
        "n_score_failed": 0,
        "prediction_coverage": 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / positive if positive else np.nan,
        "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        "FPR": fp / negative if negative else np.nan,
        "true_positive_fraction": positive / n,
        "predicted_positive_fraction": float(p.mean()),
        "positive_fraction_absolute_error": abs(float(p.mean()) - float(y.mean())),
    }
    if rank is None:
        rank = {
            "AP": float(average_precision_score(y, s)) if positive else np.nan,
            "ROC_AUC": float(roc_auc_score(y, s)) if positive and negative else np.nan,
            "Brier": float(brier_score_loss(y, s)) if probability else np.nan,
        }
    z.update(rank)
    return z


def model_names():
    return (
        ["H0", "H1", "H2", "H3", "H4", "H5", "H6"]
        + ["N_" + g + "_s" + str(seed) for seed in [0, 1, 2] for g in GROUPS]
        + ["ENS_" + g for g in GROUPS]
        + ["GMM"]
    )


def meta_for(name):
    if name == "GMM":
        return {
            "model": name,
            "family": "baseline",
            "thresholds": {"fixed_0.5": 0.5},
            "deployment_candidate": False,
        }
    return json.loads((RUN / "models" / (name + ".json")).read_text())


def metrics_main(d):
    all_records = []
    risk_records = []
    diagnostics = []
    curve_manifest = []
    names = model_names()
    y = d.ground_truth_binary.to_numpy(int)
    risk = risks(d)
    risk.update(
        {
            "zero_reference_signal": d.repair_reference_no_nonzero_signal.to_numpy(
                bool
            ),
            "reference_support_insufficient": d.repair_reference_n_observed.to_numpy(
                int
            )
            < 20,
            "legacy_reference_MAD_zero": d.qc_reference_mad_zero.to_numpy(bool),
            "conditional_intensity_missing": d[
                ["positive_mean_" + x for x in ["whole", "edge", "inner", "reference"]]
            ]
            .isna()
            .any(axis=1)
            .to_numpy(),
        }
    )
    groups = []
    for split, sg in d.groupby("split", sort=True):
        groups.append((split, "split", split, "all", sg.index.to_numpy()))
        for context, g in sg.groupby("context", sort=True):
            groups.append((split, "context", context, context, g.index.to_numpy()))
        for uid, g in sg.groupby("image_id", sort=True):
            groups.append(
                (split, "image", uid, str(g.context.iloc[0]), g.index.to_numpy())
            )
    for name in names:
        meta = meta_for(name)
        s = (
            d.GMM_score.to_numpy(float)
            if name == "GMM"
            else np.load(RUN / "predictions" / (name + "_all.npy"))
        )
        assert len(s) == len(d) and np.isfinite(s).all()
        modes = meta["thresholds"]
        prob = name != "GMM"
        output = d[
            [
                "dataset_id",
                "image_id",
                "cell_id",
                "marker",
                "split",
                "context",
                "ground_truth_original",
                "ground_truth_binary",
            ]
        ].copy()
        output["score"] = s
        for mode, t in modes.items():
            output["prediction_" + mode] = s > np.asarray(t, dtype=s.dtype)
        output.to_parquet(
            RUN / "predictions" / (name + "_all_cells.parquet"), index=False
        )
        for split, level, unit, context, ix in groups:
            yy = y[ix]
            ss = s[ix]
            base = record_metrics(yy, ss, 0.5, probability=prob)
            rank = {k: base[k] for k in ["AP", "ROC_AUC", "Brier"]}
            for mode, t in modes.items():
                rec = {
                    "model": name,
                    "family": meta["family"],
                    "split": split,
                    "level": level,
                    "unit": unit,
                    "context": context,
                    "threshold_mode": mode,
                    "threshold": t,
                }
                rec.update(record_metrics(yy, ss, t, rank, prob))
                all_records.append(rec)
            if level != "image":
                for stratum, mask in risk.items():
                    if stratum == "all":
                        continue
                    jx = ix[mask[ix]]
                    if not len(jx):
                        continue
                    yy = y[jx]
                    ss = s[jx]
                    base = record_metrics(yy, ss, 0.5, probability=prob)
                    rank = {k: base[k] for k in ["AP", "ROC_AUC", "Brier"]}
                    for mode, t in modes.items():
                        rec = {
                            "model": name,
                            "family": meta["family"],
                            "split": split,
                            "level": level,
                            "unit": unit,
                            "context": context,
                            "stratum": stratum,
                            "threshold_mode": mode,
                            "threshold": t,
                        }
                        rec.update(record_metrics(yy, ss, t, rank, prob))
                        risk_records.append(rec)
        # Test-only threshold requirements are diagnostic; deployment thresholds above stay fixed.
        for context, g in d[d.split == TEST].groupby("context", sort=True):
            ix = g.index.to_numpy()
            yy = y[ix]
            ss = s[ix]
            grid = np.arange(1, 1000) / 1000.0
            order = np.argsort(ss)
            sv = ss[order]
            cum = np.r_[0, np.cumsum(yy[order])]
            j = np.searchsorted(sv, grid.astype(sv.dtype), side="right")
            tp = yy.sum() - cum[j]
            recall = tp / yy.sum()
            for goal in [0.9, 0.95]:
                ok = np.where(recall >= goal)[0]
                t = float(grid[ok[-1]]) if len(ok) else 0.0
                rec = {
                    "model": name,
                    "context": context,
                    "requested_recall": goal,
                    "test_only_threshold": t,
                    "diagnostic_only_not_deployable": True,
                    "deployed_R90_threshold": modes.get("dev_recall_90", np.nan),
                }
                rec.update(record_metrics(yy, ss, t, probability=prob))
                diagnostics.append(rec)
        testidx = d.index[d.split == TEST].to_numpy()
        precision, recall, threshold = precision_recall_curve(y[testidx], s[testidx])
        np.savez_compressed(
            RUN / "results/curves" / (name + "_TEST_DIAGNOSTIC_PR.npz"),
            precision=precision,
            recall=recall,
            threshold=threshold,
        )
        curve_manifest.append(
            {
                "model": name,
                "path": "results/curves/" + name + "_TEST_DIAGNOSTIC_PR.npz",
                "scope": "test diagnostic only; no model or deployment-threshold selection",
            }
        )
        print("METRICS_DONE", name, flush=True)
    table = pd.DataFrame(all_records)
    risk_table = pd.DataFrame(risk_records)
    table.to_csv(RUN / "results/METRICS_BY_UNIT.csv", index=False)
    risk_table.to_csv(RUN / "results/RISK_STRATA.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(
        RUN / "results/TEST_ONLY_THRESHOLD_DIAGNOSTIC.csv", index=False
    )
    pd.DataFrame(curve_manifest).to_csv(RUN / "results/CURVE_MANIFEST.csv", index=False)
    table[(table.split == TEST) & (table.level == "split")].to_csv(
        RUN / "results/EXTERNAL_MAIN.csv", index=False
    )
    table[(table.split == TEST) & (table.level == "context")].to_csv(
        RUN / "results/EXTERNAL_BY_TMA.csv", index=False
    )
    neural = table[table.family == "neural"].copy()
    neural["group"] = neural.model.str.replace(r"^N_", "", regex=True).str.replace(
        r"_s[012]$", "", regex=True
    )
    numerical = [
        "precision",
        "recall",
        "F1",
        "AP",
        "ROC_AUC",
        "FPR",
        "tp",
        "fp",
        "fn",
        "tn",
        "positive_fraction_absolute_error",
    ]
    summary = neural.groupby(
        ["group", "split", "level", "unit", "context", "threshold_mode"]
    )[numerical].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(RUN / "results/NEURAL_SEED_SUMMARY.csv", index=False)
    pairs = [
        ("H1", "H0", "normalization_only"),
        ("H2", "H1", "representation_only"),
        ("H5", "H2", "imputation_only"),
        ("H3", "H5", "missing_indicators"),
        ("H4", "H3", "quantile_transform"),
    ]
    for seed in [0, 1, 2]:
        pairs += [
            (
                "N_image_target_s" + str(seed),
                "N_image_s" + str(seed),
                "target_mask_seed" + str(seed),
            ),
            (
                "N_image_target_other_s" + str(seed),
                "N_image_target_s" + str(seed),
                "other_mask_seed" + str(seed),
            ),
            (
                "N_image_target_other_tabular_s" + str(seed),
                "N_image_target_other_s" + str(seed),
                "fusion_vs_image_seed" + str(seed),
            ),
            (
                "N_image_target_other_tabular_s" + str(seed),
                "N_tabular_only_s" + str(seed),
                "fusion_vs_features_seed" + str(seed),
            ),
        ]
    comparisons = []
    join = ["split", "level", "unit", "context", "threshold_mode"]
    for a, b, label in pairs:
        joined = table[table.model == a].merge(
            table[table.model == b],
            on=join,
            suffixes=("_A", "_B"),
            validate="one_to_one",
        )
        for row in joined.itertuples(index=False):
            rec = {k: getattr(row, k) for k in join}
            rec.update(comparison=label, model_A=a, model_B=b)
            for k in numerical:
                rec["delta_" + k] = getattr(row, k + "_A") - getattr(row, k + "_B")
            comparisons.append(rec)
    pd.DataFrame(comparisons).to_csv(RUN / "results/PAIRED_ABLATIONS.csv", index=False)
    return table, risk_table


def calibration(d):
    risk = risks(d)
    y = d.ground_truth_binary.to_numpy(int)
    test = d[d.split == TEST]
    indices = []
    records = []
    parameters = []
    dtype_audit = []
    eps = np.finfo(np.float64).eps
    for replicate in [101, 102, 103]:
        reserved = {}
        holdout = np.zeros(len(d), bool)
        holdout[test.index] = True
        for uid, g in test.groupby("image_id", sort=True):
            ids = g.index.to_numpy()
            keys = [
                int.from_bytes(
                    hashlib.blake2b(
                        (str(replicate) + "|" + str(uid) + "|" + str(int(c))).encode(),
                        digest_size=8,
                    ).digest(),
                    "little",
                )
                for c in g.cell_id
            ]
            chosen = ids[np.argsort(np.asarray(keys, np.uint64))[:100]]
            assert len(chosen) == 100 and len(g) > 100
            reserved[uid] = chosen
            holdout[chosen] = False
            for rank, i in enumerate(chosen, 1):
                indices.append(
                    {
                        "replicate": replicate,
                        "image_id": uid,
                        "row_id": int(i),
                        "cell_id": int(d.loc[i, "cell_id"]),
                        "sampling_rank": rank,
                        "used_for_evaluation": False,
                        "label_used_for_sampling": False,
                    }
                )
        assert (
            np.intersect1d(
                np.flatnonzero(holdout), np.concatenate(list(reserved.values()))
            ).size
            == 0
        )
        for name in [x for x in model_names() if x != "GMM"]:
            meta = meta_for(name)
            native = np.load(RUN / "predictions" / (name + "_all.npy"))
            s = native.astype(np.float64)
            rawlogit = logit(np.clip(s, eps, 1 - eps))
            original_threshold = meta["thresholds"]["dev_recall_90"]
            threshold = native_threshold(original_threshold, native.dtype)
            for budget in [0, 20, 50, 100]:
                adjusted = s.copy()
                unchanged = np.zeros(len(d), bool)
                for uid, g in test.groupby("image_id", sort=True):
                    ix = g.index.to_numpy()
                    cal = reserved[uid][:budget]
                    fallback = budget > 0 and len(np.unique(y[cal])) < 2
                    b = 0.0
                    if budget and not fallback:
                        z = rawlogit[cal]
                        yy = y[cal]
                        b = float(
                            brentq(
                                lambda v: float(np.sum(expit(z + v) - yy)) + v,
                                -64.0,
                                64.0,
                            )
                        )
                    if budget and not fallback:
                        adjusted[ix] = expit(rawlogit[ix] + b)
                    else:
                        unchanged[ix] = True
                    parameters.append(
                        {
                            "replicate": replicate,
                            "model": name,
                            "image_id": uid,
                            "budget": budget,
                            "n_calibration_positive": int(y[cal].sum())
                            if budget
                            else 0,
                            "intercept": b,
                            "one_class_fallback_to_global": fallback,
                            "calibration_instances_excluded_from_all_comparisons": 100,
                        }
                    )
                check = holdout & unchanged
                assert np.array_equal(
                    adjusted[check] > threshold,
                    native[check] > np.asarray(original_threshold, dtype=native.dtype),
                )
                dtype_audit.append(
                    {
                        "model": name,
                        "replicate": replicate,
                        "budget": budget,
                        "dtype": str(native.dtype),
                        "original_threshold": original_threshold,
                        "effective_native_threshold": threshold,
                        "evaluation_n": int(holdout.sum()),
                        "old_implementation_changed_decisions": int(
                            np.sum(
                                (adjusted[holdout] > original_threshold)
                                != (adjusted[holdout] > threshold)
                            )
                        ),
                        "unchanged_mapping_n": int(check.sum()),
                        "old_unchanged_mapping_changed_decisions": int(
                            np.sum(
                                (adjusted[check] > original_threshold)
                                != (adjusted[check] > threshold)
                            )
                        ),
                        "native_mapping_exact": True,
                    }
                )
                for level, unit, ix in (
                    [("split", TEST, test.index.to_numpy())]
                    + [
                        ("context", context, g.index.to_numpy())
                        for context, g in test.groupby("context", sort=True)
                    ]
                    + [
                        ("image", uid, g.index.to_numpy())
                        for uid, g in test.groupby("image_id", sort=True)
                    ]
                ):
                    ix = ix[holdout[ix]]
                    if not len(ix):
                        continue
                    rec = {
                        "simulation": "target_core_supervised_calibration",
                        "replicate": replicate,
                        "model": name,
                        "budget_per_core": budget,
                        "level": level,
                        "unit": unit,
                        "threshold_mode": "frozen_dev_recall_90",
                        "threshold": threshold,
                        "same_evaluation_population_across_budgets": True,
                    }
                    rec.update(record_metrics(y[ix], adjusted[ix], threshold))
                    records.append(rec)
                    if level != "image":
                        for stratum in [
                            "weak_positive",
                            "contact_bright_proxy_negative",
                            "zero_target_signal",
                            "boundary_abnormal",
                        ]:
                            jx = ix[risk[stratum][ix]]
                            if not len(jx):
                                continue
                            rr = rec.copy()
                            rr["stratum"] = stratum
                            rr.update(record_metrics(y[jx], adjusted[jx], threshold))
                            records.append(rr)
            print("CALIBRATION_SIMULATION_DONE", replicate, name, flush=True)
    pd.DataFrame(dtype_audit).to_csv(
        RUN / "results/CALIBRATION_NATIVE_THRESHOLD_AUDIT.csv", index=False
    )
    pd.DataFrame(indices).to_csv(
        RUN / "results/CALIBRATION_RESERVED_INSTANCES.csv", index=False
    )
    pd.DataFrame(parameters).to_csv(
        RUN / "results/CALIBRATION_PARAMETERS.csv", index=False
    )
    cal = pd.DataFrame(records)
    cal.to_csv(RUN / "results/TARGET_LABEL_CALIBRATION_SIMULATION.csv", index=False)
    savej(
        RUN / "results/CALIBRATION_COMPLETE.json",
        {
            "replicates": [101, 102, 103],
            "budgets_per_core": [20, 50, 100],
            "all_100_reserved_instances_excluded_from_every_budget_and_zero_budget_baseline": True,
            "fixed_full_external_main_unchanged": 437480,
            "holdout_n_per_replicate": int((test.shape[0] - 42 * 100)),
            "sampling_label_blind": True,
            "interpretation": "supervised within-core simulation, not zero-target-label generalization",
            "rows": len(cal),
        },
    )


def main():
    assert (RUN / "predictions/HGB_EXTERNAL_COMPLETE.json").exists() and (
        RUN / "predictions/NEURAL_EXTERNAL_COMPLETE.json"
    ).exists()
    (RUN / "results/curves").mkdir(exist_ok=True)
    d, man = load()
    start = time.time()
    quality = []
    for (split, context), g in d.groupby(["split", "context"], sort=True):
        for col in (
            man["feature_sets"]["H0"]
            + man["feature_sets"]["H1"]
            + [c for c in man["feature_sets"]["H2"] if not c.startswith("h1_")]
        ):
            a = g[col].to_numpy(float)
            quality.append(
                {
                    "split": split,
                    "context": context,
                    "feature": col,
                    "n_labelled_instances": len(g),
                    "n_missing": int(np.isnan(a).sum()),
                    "n_observed_zero": int(np.sum(a == 0)),
                    "missing_fraction": float(np.isnan(a).mean()),
                }
            )
    pd.DataFrame(quality).to_csv(
        RUN / "results/INPUT_QUALITY_LABELLED_POPULATION.csv", index=False
    )
    table, risk = metrics_main(d)
    calibration(d)
    savej(
        RUN / "results/EVALUATION_COMPLETE.json",
        {
            "metric_rows": len(table),
            "risk_rows": len(risk),
            "models": len(model_names()),
            "seconds": time.time() - start,
            "all_main_model_populations_aligned": True,
            "full_external_labelled": 437480,
            "full_internal_labelled": 539132,
            "full_train_labelled": 101057,
            "selection_frozen_sha256": sha(RUN / "models/MODEL_SELECTION_FROZEN.json"),
            "PanM_accessed": False,
            "code_sha256": sha(__file__),
        },
    )
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
