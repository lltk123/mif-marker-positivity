"""Research evaluate for the frozen PhenoBIC CD8 experiment."""

import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from common import *

MODELS = ["GMM", "intensity_8", "morphospatial_90"] + list(SETS)
PAIRS = [
    ("E1", "intensity_8"),
    ("E2", "E1"),
    ("E3", "E1"),
    ("E4", "E3"),
    ("E4", "E2"),
    ("E4", "GMM"),
    ("E4", "morphospatial_90"),
]


def evaluate():
    assert (A / "results/TRAINING_COMPLETE.json").exists() and (
        A / "THRESHOLDS_FROZEN.json"
    ).exists()
    records = []
    curveindex = []
    oofrows = []
    for mode in ["fixed_0.5", "OOF_threshold"]:
        d = pd.read_parquet(
            A / "predictions" / ("final_all_cells_" + mode + ".parquet")
        )
        d = d[d.ground_truth_binary.isin([0, 1])]
        assert len(d) == 1077669
        groups = (
            [("split", (split,), g) for split, g in d.groupby("split")]
            + [("context", key, g) for key, g in d.groupby(["split", "context"])]
            + [
                ("unit", (g.split.iloc[0], g.context.iloc[0], uid), g)
                for uid, g in d.groupby("image_id")
            ]
        )
        for level, key, g in groups:
            for model in MODELS:
                t = float(g[model + "_threshold"].iloc[0])
                m = metrics(g.ground_truth_binary, g[model + "_score"], t)
                m.update(
                    level=level,
                    split=key[0],
                    context=key[1] if len(key) > 1 else "",
                    image_id=key[2] if len(key) > 2 else "",
                    model=model,
                    threshold_mode=mode,
                    threshold=t,
                    threshold_source="original_baseline" if model not in SETS else mode,
                    n_units=g.image_id.nunique(),
                    training_resubstitution=key[0] == "training_WSIs"
                    and model != "GMM",
                )
                records.append(m)
                if level == "split" and mode == "fixed_0.5":
                    y = g.ground_truth_binary.to_numpy(int)
                    s = g[model + "_score"].to_numpy(float)
                    fpr, tpr, rt = roc_curve(y, s, drop_intermediate=False)
                    p, r, pt = precision_recall_curve(y, s)
                    fn = "%s_%s.npz" % (model, key[0])
                    np.savez_compressed(
                        A / "curves" / fn,
                        ROC_fpr=fpr,
                        ROC_tpr=tpr,
                        ROC_threshold=rt,
                        PR_precision=p,
                        PR_recall=r,
                        PR_threshold=pt,
                    )
                    curveindex.append(
                        dict(
                            model=model,
                            split=key[0],
                            file="curves/" + fn,
                            n=len(g),
                            ROC_points=len(rt),
                            PR_points=len(p),
                            PR_thresholds=len(pt),
                            note="PR endpoint has no threshold; threshold array is one shorter",
                        )
                    )
        oof = pd.read_parquet(A / "predictions" / ("training_OOF_" + mode + ".parquet"))
        for model in SETS:
            for uid, g in [("ALL", oof)] + list(oof.groupby("image_id")):
                m = metrics(
                    g.ground_truth_binary,
                    g[model + "_score"],
                    g[model + "_threshold"].iloc[0],
                )
                m.update(
                    model=model,
                    image_id=uid,
                    threshold_mode=mode,
                    threshold=g[model + "_threshold"].iloc[0],
                    interpretation="OOF threshold development fitting value"
                    if mode == "OOF_threshold"
                    else "OOF fixed threshold",
                )
                oofrows.append(m)
        print("EVALUATED", mode, flush=True)
    allm = pd.DataFrame(records)
    allm.to_csv(A / "results/metrics_all_levels.csv", index=False)
    for level in ["split", "context", "unit"]:
        allm[allm.level == level].to_csv(
            A / "results" / ("metrics_" + level + ".csv"), index=False
        )
    for mode in ["fixed_0.5", "OOF_threshold"]:
        allm[allm.threshold_mode == mode].to_csv(
            A / "results" / ("metrics_" + mode + ".csv"), index=False
        )
    pd.DataFrame(oofrows).to_csv(A / "results/OOF_metrics.csv", index=False)
    pd.DataFrame(curveindex).to_csv(A / "curves/curve_index.csv", index=False)
    macro = []
    low = []
    u = allm[allm.level == "unit"]
    for key, g in u.groupby(["split", "model", "threshold_mode"]):
        r = dict(zip(["split", "model", "threshold_mode"], key))
        r.update(
            n_units=len(g), units_no_positive_prediction=int(((g.TP + g.FP) == 0).sum())
        )
        for c in [
            "F1",
            "precision",
            "recall",
            "specificity",
            "ROC_AUC",
            "AP",
            "prevalence_absolute_error",
        ]:
            for stat, v in [
                ("mean", g[c].mean()),
                ("std", g[c].std()),
                ("min", g[c].min()),
                ("p10", g[c].quantile(0.1)),
                ("median", g[c].median()),
                ("max", g[c].max()),
            ]:
                r[c + "_" + stat] = v
        macro.append(r)
        low.append(g.sort_values("F1").head(10))
    pd.DataFrame(macro).to_csv(A / "results/metrics_macro_units.csv", index=False)
    pd.concat(low).to_csv(A / "results/lowest_units.csv", index=False)
    paired = []
    groupcols = ["level", "split", "context", "image_id", "threshold_mode"]
    nums = [
        "TP",
        "FP",
        "FN",
        "TN",
        "F1",
        "precision",
        "recall",
        "specificity",
        "ROC_AUC",
        "AP",
        "predicted_prevalence",
        "prevalence_absolute_error",
    ]
    for key, g in allm.fillna({"context": "", "image_id": ""}).groupby(groupcols):
        idx = g.set_index("model")
        for target, base in PAIRS:
            r = dict(zip(groupcols, key))
            r.update(comparison=target + " - " + base, target=target, reference=base)
            for c in nums:
                r[c + "_delta"] = idx.loc[target, c] - idx.loc[base, c]
            paired.append(r)
    pd.DataFrame(paired).to_csv(A / "results/paired_changes.csv", index=False)
    f = pd.read_parquet(A / "PhenoBIC_E1_E4_features.parquet")
    ann = pd.read_parquet(
        A / "audit/cells.parquet", columns=KEYS + ["ground_truth_binary"]
    )
    f = f.merge(ann, on=KEYS, validate="one_to_one")
    f["class_stratum"] = f.ground_truth_binary.map(
        {0.0: "negative", 1.0: "positive"}
    ).fillna("unlabelled")
    missing = []
    qcs = []
    for grouping in [["image_id"], ["split", "context", "class_stratum"]]:
        for key, g in f.groupby(grouping):
            key = (key,) if len(grouping) == 1 else key
            r0 = dict(zip(grouping, key))
            r0.update(
                level="unit" if len(grouping) == 1 else "true_class",
                split=g.split.iloc[0],
                context=g.context.iloc[0],
                n=len(g),
            )
            for col in I8 + S9:
                v = g[col]
                row = dict(
                    r0,
                    feature=col,
                    n_finite=int(np.isfinite(v).sum()),
                    n_missing=int(v.isna().sum()),
                    missing_rate=float(v.isna().mean()),
                    zero_fraction_all=float((v == 0).mean()),
                    zero_fraction_finite=float((v.dropna() == 0).mean()),
                )
                missing.append(row)
            for col in [c for c in f if c.startswith("qc_")]:
                qcs.append(
                    dict(
                        r0,
                        reason=col,
                        count=int(g[col].sum()),
                        fraction=float(g[col].mean()),
                    )
                )
    pd.DataFrame(missing).to_csv(A / "results/feature_availability.csv", index=False)
    pd.DataFrame(qcs).to_csv(A / "results/QC_reasons.csv", index=False)
    write_json(
        A / "results/EVALUATION_COMPLETE.json",
        dict(
            status="complete",
            metric_rows=len(allm),
            curve_files=len(curveindex),
            all_labeled=1077669,
            code_sha256=sha(__file__),
            threshold_freeze_sha256=sha(A / "THRESHOLDS_FROZEN.json"),
        ),
    )
    print("EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    evaluate()
