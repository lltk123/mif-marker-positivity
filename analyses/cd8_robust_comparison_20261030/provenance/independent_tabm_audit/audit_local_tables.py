"""Independent audits of completed TabM summary tables and frozen records."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

B = Path(r"F:/Codex/Projects/学习/mif-marker-positivity/analyses")
R = B / "cd8_tabm_robust_20261030"
C = B / "cd8_robust_comparison_20261030"
OUT = Path(__file__).parent
checks = []


def read(f):
    return pd.read_csv(f, float_precision="round_trip")


def js(f):
    return json.loads(f.read_text(encoding="utf-8-sig"))


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def check(name, value, detail=""):
    checks.append({"check": name, "passed": bool(value), "detail": detail})
    if not bool(value):
        raise AssertionError(name + ": " + detail)


def close(a, b):
    return np.allclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True)


D = read(R / "results/metrics_all.csv")
F = js(R / "FROZEN_TEST_PROTOCOL.json")
E = js(R / "results/EVALUATION_MANIFEST.json")
P = js(R / "PREDICTIONS_COMPLETE.json")
V = js(R / "DELIVERY_VALIDATION.json")
expected = {f"{f}_{g}_s{s}" for f in ["TabM", "HGB"] for g in "ABC" for s in range(3)}
check("18 canonical frozen models", set(F["models"]) == expected)
check(
    "19 evaluation models and 1998 rows",
    len(D) == 1998
    and set(D.model_id) == expected | {"GMM"}
    and E["n_models"] == 19
    and E["n_metric_rows"] == 1998,
)
check(
    "full completion markers",
    P["models"] == V["n_models"] == 18
    and P["n_per_model"] == V["n_cells_per_model"] == E["n_all_cells"] == 1817783
    and E["status"] == V["status"] == "complete",
)
check(
    "frozen signatures",
    sha(R / "FROZEN_TEST_PROTOCOL.json") == P["frozen_sha256"] == E["protocol_sha256"],
)
check(
    "no external label selection",
    F["test_labels_used_in_selection"] is False and V["PanM_accessed"] is False,
)
check(
    "risk audit earlier than freeze",
    js(R / "AUDIT_PROTOCOL.json")["frozen_unix_time"] < F["frozen_unix_time"],
)
report = js(R / "REPORT_GENERATION_MANIFEST.json")
for scope in ["inputs", "outputs"]:
    for name, digest in report[scope].items():
        check("report " + scope + " SHA " + name, sha(R / name) == digest)
check(
    "report generation script SHA",
    sha(R / "scripts/write_tabm_report.py") == report["script_sha256"],
)
keys = ["model_id", "threshold_mode", "split", "group_type", "group_id", "context"]
check("unique metric keys", not D.duplicated(keys).any())
check(
    "complete observation and calibration coverage",
    D.n_score_failed.eq(0).all()
    and D.prediction_coverage.eq(1).all()
    and D.n_calibration_invalid.eq(0).all()
    and D.calibration_coverage.eq(1).all(),
)
check(
    "confusion denominators",
    (D.tp + D.fp + D.fn + D.tn).eq(D.n_cells).all()
    and (D.tp + D.fn).eq(D.n_true_positive).all()
    and (D.tn + D.fp).eq(D.n_true_negative).all(),
)
for col, value in [
    ("precision", np.where(D.tp + D.fp > 0, D.tp / (D.tp + D.fp), 0)),
    ("recall", D.tp / (D.tp + D.fn)),
    ("f1", 2 * D.tp / (2 * D.tp + D.fp + D.fn)),
    ("specificity", D.tn / (D.tn + D.fp)),
    ("predicted_positive_fraction", (D.tp + D.fp) / D.n_cells),
    ("positive_fraction_absolute_error", abs(D.fp - D.fn) / D.n_cells),
]:
    check("recomputed " + col, close(D[col], value))
counts = {
    "training_WSIs": 101057,
    "internal_val_WSIs": 539132,
    "external_val_TMA_cores": 437480,
}
for (model, mode, split), g in D.groupby(["model_id", "threshold_mode", "split"]):
    top = g[g.group_type == "split"]
    units = g[g.group_type == "image"]
    cols = ["n_cells", "n_true_positive", "n_true_negative", "tp", "fp", "fn", "tn"]
    check(
        "group sums " + str((model, mode, split)),
        len(top) == 1
        and int(top.iloc[0].n_cells) == counts[split]
        and close(units[cols].sum(), top[cols].iloc[0]),
    )
    if split == "external_val_TMA_cores":
        check(
            "TMA sums " + model + mode,
            close(g[g.group_type == "context"][cols].sum(), top[cols].iloc[0]),
        )
for model in expected:
    meta = js(R / "models" / model / "metadata.json")
    check("frozen metadata " + model, meta == F["models"][model])
    g = D[D.model_id == model]
    check(
        "threshold application " + model,
        g[g.threshold_mode == "fixed_0.5"].threshold.eq(0.5).all()
        and g[g.threshold_mode == "validation_selected"]
        .threshold.eq(meta["threshold_validation"])
        .all(),
    )
    p = g[g.threshold_mode == "fixed_0.5"].sort_values(keys[2:])
    q = g[g.threshold_mode == "validation_selected"].sort_values(keys[2:])
    check(
        "ranking invariant across thresholds " + model,
        close(p.roc_auc, q.roc_auc) and close(p.average_precision, q.average_precision),
    )
    check(
        "feature count " + model,
        len(meta["features"]) == {"A": 8, "B": 27, "C": 90}[meta["feature_set"]]
        and meta["features"] == meta["requested_features"],
    )
    # Saved sidecars bind the canonical inference run to this freeze.
    side = js(R / "predictions" / (model + ".json"))
    check(
        "prediction sidecar " + model,
        side["n"] == 1817783
        and side["frozen_sha256"] == sha(R / "FROZEN_TEST_PROTOCOL.json"),
    )
histories = list((R / "models").glob("TabM_*/history.csv"))
check("12 actual TabM fits", len(histories) == 12)
fitcounts = {}
for f in histories:
    meta = js(f.with_name("metadata.json"))
    history = read(f)
    key = (meta["feature_set"], meta["seed"])
    fitcounts[key] = fitcounts.get(key, 0) + 1
    check(
        "training history " + f.parent.name,
        len(history) == meta["epochs_run"]
        and close(history.validation_macro_AP.max(), meta["validation_macro_AP"])
        and close(
            history.loc[
                history.epoch == meta["best_epoch"], "validation_macro_AP"
            ].iloc[0],
            meta["validation_macro_AP"],
        ),
    )
check(
    "6 seed0 candidates plus 6 extra seed fits",
    all(fitcounts[(g, s)] == (2 if s == 0 else 1) for g in "ABC" for s in range(3)),
)
check("9 HGB fits", len(list((R / "models").glob("HGB_*/metadata.json"))) == 9)
for g in "ABC":
    candidates = [
        js(R / "models" / f"TabM_{g}_s0_{name}" / "metadata.json")
        for name in ["official_default", "lower_lr"]
    ]
    canonical = F["models"][f"TabM_{g}_s0"]
    check(
        "validation-only candidate winner " + g,
        close(
            canonical["validation_macro_AP"],
            max(x["validation_macro_AP"] for x in candidates),
        ),
    )
    for seed in [1, 2]:
        check(
            "chosen config carried to seed " + g + str(seed),
            F["models"][f"TabM_{g}_s{seed}"]["candidate"] == canonical["candidate"],
        )
# Independent aggregation uses the original per-model table, not report pivots.
G = [
    "model_family",
    "feature_set",
    "threshold_mode",
    "split",
    "group_type",
    "group_id",
    "context",
]
lookup = {key: g for key, g in D.groupby(G, dropna=False)}
summary = read(R / "results/metrics_seed_summary.csv")
for _, row in summary.iterrows():
    g = lookup[tuple(row[k] for k in G)]
    v = g[row.metric]
    check(
        "seed summary " + str(row.name),
        int(row.n_seeds) == len(g)
        and int(row.n_finite) == int(v.notna().sum())
        and close(
            [row["mean"], row.std_between_seeds, row.minimum, row.maximum],
            [v.mean(), v.std(ddof=1), v.min(), v.max()],
        ),
    )
units = read(R / "results/unit_equal_summary.csv")
ulookup = {
    key: g
    for key, g in D[D.group_type == "image"].groupby(
        ["model_id", "threshold_mode", "split"]
    )
}
for _, row in units.iterrows():
    g = ulookup[(row.model_id, row.threshold_mode, row["split"])]
    v = g[row.metric]
    check(
        "unit equal " + str(row.name),
        len(g) == row.n_units
        and int(v.notna().sum()) == row.n_finite_units
        and close(
            [row.unit_equal_mean, row.unit_median, row.unit_minimum, row.unit_maximum],
            [v.mean(), v.median(), v.min(), v.max()],
        ),
    )
for filename in [
    "report_performance_fixed_0.5.csv",
    "report_performance_validation_selected.csv",
]:
    r = read(R / "results" / filename)
    for _, row in r.iterrows():
        g = lookup[tuple(row[k] for k in G)]
        for col in [x[:-5] for x in r.columns if x.endswith("_mean")]:
            check(
                "report pivot " + filename + str(row.name) + col,
                close(
                    [row[col + "_mean"], row[col + "_std"]],
                    [g[col].mean(), g[col].std(ddof=1)],
                ),
            )
paired = read(R / "results/paired_comparisons.csv")
source = {tuple(row[k] for k in keys): row for _, row in D.iterrows()}
for _, row in paired.iterrows():
    parts = row.comparison.split("_")
    seed = int(row.seed)
    if "_minus_HGB_" in row.comparison:
        left = f"TabM_{parts[-1]}_s{seed}"
        right = f"HGB_{parts[-1]}_s{seed}"
    elif row.comparison.endswith("_minus_GMM"):
        left = f"TabM_{parts[1]}_s{seed}"
        right = "GMM"
    else:
        left = f"{parts[0]}_{parts[1]}_s{seed}"
        right = f"{parts[0]}_{parts[-1]}_s{seed}"
    for suffix, model, mode in [
        ("_new", left, row.threshold_mode),
        ("_old", right, row.reference_threshold_mode),
    ]:
        actual = source[
            (model, mode, row["split"], row.group_type, row.group_id, row.context)
        ]
        for metric in [x[6:] for x in paired.columns if x.startswith("delta_")]:
            check(
                "paired source " + str(row.name) + suffix + metric,
                close(row[metric + suffix], actual[metric]),
            )
    for metric in [x[6:] for x in paired.columns if x.startswith("delta_")]:
        check(
            "paired delta " + str(row.name) + metric,
            close(row["delta_" + metric], row[metric + "_new"] - row[metric + "_old"]),
        )
rootpair = read(C / "results/PAIRED_CHANGES.csv")
for metric in [x[:-6] for x in rootpair if x.endswith("_delta")]:
    check(
        "root paired delta " + metric,
        close(
            rootpair[metric + "_delta"],
            rootpair[metric + "_mean_new"] - rootpair[metric + "_mean_old"],
        ),
    )
comparison = read(C / "results/MODEL_COMPARISON.csv")
new = comparison[
    (comparison.variant == "new") & comparison.model_family.isin(["TabM", "HGB"])
]
check("root contains1944 new rows", len(new) == 1944)
for _, row in new.iterrows():
    actual = source[
        (
            row.model_id,
            row.threshold_mode,
            row["split"],
            row.level,
            row.unit,
            row.context,
        )
    ]
    for col in [
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "threshold",
        "n_cells",
    ]:
        check("root new source " + str(row.name) + col, close(row[col], actual[col]))
# Old comparisons are verified against their actual legacy source table.
old = read(B / "cd8_tabm_20261029/results/metrics_all.csv")
oldsource = {tuple(row[k] for k in keys): row for _, row in old.iterrows()}
for _, row in comparison[
    (comparison.variant == "old") & comparison.model_family.isin(["TabM", "HGB"])
].iterrows():
    actual = oldsource[
        (
            row.model_id,
            row.threshold_mode,
            row["split"],
            row.level,
            row.unit,
            row.context,
        )
    ]
    for col in [
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "average_precision",
        "roc_auc",
        "threshold",
        "n_cells",
    ]:
        check("root old source " + str(row.name) + col, close(row[col], actual[col]))
result = {
    "status": "passed",
    "scope": "independent local summary/freeze/fit-budget audit; full-cell audit separately recorded",
    "n_checks": len(checks),
    "metrics_rows": len(D),
    "TabM_fits": 12,
    "HGB_fits": 9,
    "formal_models": 18,
    "frozen_sha256": sha(R / "FROZEN_TEST_PROTOCOL.json"),
    "metrics_sha256": sha(R / "results/metrics_all.csv"),
    "report_sha256": sha(R / "ANALYSIS_REPORT.md"),
    "script_sha256": sha(Path(__file__)),
    "PanM_accessed": False,
}
pd.DataFrame(checks).to_csv(OUT / "LOCAL_CHECKS.csv", index=False)
(OUT / "LOCAL_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
print(
    read(R / "results/report_performance_fixed_0.5.csv")
    .query("group_type=='split' and split=='external_val_TMA_cores'")[
        [
            "model_family",
            "feature_set",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "f1_std",
            "average_precision_mean",
            "tp_mean",
            "fp_mean",
            "fn_mean",
        ]
    ]
    .to_string(index=False)
)
