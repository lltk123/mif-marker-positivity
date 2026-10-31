"""Cell-level evaluation; validation-only threshold selection, frozen external test."""
from common import *
import argparse
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


def key_score_hash(frame):
    return hashlib.sha256(
        frame[KEYS + ["score"]]
        .sort_values(KEYS)
        .to_csv(index=False, lineterminator="\n")
        .encode()
    ).hexdigest()


def metrics(y, s, t):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    valid = np.isfinite(s)
    n = len(y)
    out = dict(
        n=n,
        n_scored=int(valid.sum()),
        n_failed=int((~valid).sum()),
        coverage=float(valid.mean()),
        n_positive=int(y.sum()),
        true_positive_fraction=float(y.mean()),
        threshold=float(t),
    )
    names = [
        "TP",
        "FP",
        "FN",
        "TN",
        "precision",
        "recall",
        "F1",
        "specificity",
        "ROC_AUC",
        "AP",
        "predicted_positive_fraction",
        "positive_fraction_absolute_error",
    ]
    if not valid.all():
        out.update({k: np.nan for k in names})
        return out
    p = s > t
    tp = int(((y == 1) & p).sum())
    fp = int(((y == 0) & p).sum())
    fn = int(((y == 1) & ~p).sum())
    tn = int(((y == 0) & ~p).sum())
    out.update(
        TP=tp,
        FP=fp,
        FN=fn,
        TN=tn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else np.nan,
        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        specificity=tn / (tn + fp) if tn + fp else np.nan,
        ROC_AUC=roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan,
        AP=average_precision_score(y, s) if y.sum() > 0 else np.nan,
        predicted_positive_fraction=float(p.mean()),
        positive_fraction_absolute_error=abs(float(p.mean() - y.mean())),
    )
    return out


def choose_threshold(df, model):
    rows = []
    for t in protocol()["thresholds"]["supplementary_grid"]:
        ff = []
        for i, g in df.groupby("image_id"):
            yy = g.ground_truth_binary.to_numpy(int)
            p = g.score.to_numpy() > t
            tp = ((yy == 1) & p).sum()
            fp = ((yy == 0) & p).sum()
            fn = ((yy == 1) & ~p).sum()
            den = 2 * tp + fp + fn
            f = 2 * tp / den if den else np.nan
            ff.append(f)
            rows.append(dict(model=model, threshold=t, image_id=i, F1=f))
    table = pd.DataFrame(rows)
    table.to_csv(
        R / "results" / (model + "_validation_threshold_grid.csv"), index=False
    )
    avg = table.groupby("threshold").F1.mean()
    best = sorted(avg.index, key=lambda t: (-avg[t], round(abs(t - 0.5), 12), -t))[0]
    return float(best), float(avg[best])


def assemble(model):
    if model == "GMM":
        old = pd.read_parquet(
            E / "audit/cells.parquet", columns=KEYS[:3] + ["GMM_score"]
        )
        old["marker"] = "CD8"
        old = old.rename(columns={"GMM_score": "score"})
        old["model_version"] = "original_frozen_GMM"
        return old
    paths = [
        p
        for p in (R / "predictions" / model).glob("*.parquet")
        if p.stem != "all_cells"
    ]
    x = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    exact_keys(x)
    return x


def evaluate(model):
    pred = assemble(model)
    cells = load_cells()
    exact_keys(cells)
    if "pixel_count" in pred:
        pred = pred.rename(columns={"pixel_count": "nimbus_pixel_count"})
    assert len(pred) == len(cells) and key_hash(pred) == key_hash(cells)
    drop = [c for c in pred.columns if c in cells.columns and c not in KEYS]
    pred = pred.drop(columns=drop)
    df = cells.merge(pred, on=KEYS, how="left", validate="one_to_one")
    assert len(df) == 1817783
    labelled = df.loc[df.ground_truth_binary.isin([0, 1])].copy()
    specpath = R / "models" / (model + "_thresholds.json")
    if model == "GMM":
        spec = dict(
            model=model,
            default=0.5,
            operator=">",
            validation_selected=None,
            selection="original GMM fixed 0.5; no new threshold fitting",
        )
        dump(specpath, spec)
        threshold = None
    elif specpath.exists():
        spec = json.loads(specpath.read_text())
        assert spec["protocol_sha256"] == sha(R / "NIMBUS_PROTOCOL.json")
        assert spec["prediction_score_sha256"] == key_score_hash(pred)
        threshold = spec["validation_selected"]
    else:
        val = labelled.query("split=='internal_val_WSIs'")
        if np.isfinite(val.score).all():
            threshold, value = choose_threshold(val, model)
        else:
            # Never choose a threshold on a silently reduced validation population.
            threshold, value = None, None
        spec = dict(
            model=model,
            default=0.5,
            operator=">",
            validation_selected=threshold,
            selection="equal WSI mean F1 on full labelled internal validation",
            validation_development_fitted_macro_F1=value,
            not_unbiased_validation=True,
            selection_status="complete"
            if threshold is not None
            else "unavailable_due_to_unscorable_validation",
            protocol_sha256=sha(R / "NIMBUS_PROTOCOL.json"),
            prediction_score_sha256=key_score_hash(pred),
        )
        dump(specpath, spec)
    modes = {"official_default": 0.5}
    if threshold is not None:
        modes["validation_selected"] = threshold
    for mode, t in modes.items():
        df["prediction_" + mode] = pd.array(
            np.where(np.isfinite(df.score), (df.score > t).astype(int), np.nan),
            dtype="Int8",
        )
        df["threshold_" + mode] = t
    if "prediction_validation_selected" not in df:
        df["prediction_validation_selected"] = pd.array([pd.NA] * len(df), dtype="Int8")
        df["threshold_validation_selected"] = np.nan
    df["model"] = model
    for col in ["experiment_role", "development_fold"]:
        if col in df:
            df = df.rename(columns={col: "previous_plan_" + col})
    df["experiment_role"] = df.split.map(
        {
            "training_WSIs": "training_resubstitution",
            "internal_val_WSIs": "validation_used_for_selection",
            "external_val_TMA_cores": "frozen_test",
        }
    )
    out = R / "predictions" / model
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "all_cells.parquet", index=False)
    # CSV is compressed to retain a complete editable table without multiplying storage.
    df.to_csv(out / "all_cells.csv.gz", index=False)
    rows = []
    for mode, t in modes.items():
        for level, cols in [
            ("split", ["split"]),
            ("slide", ["split", "context"]),
            ("image", ["split", "context", "image_id"]),
        ]:
            for keys, g in labelled.groupby(cols, dropna=False):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                r = dict(
                    model=model,
                    threshold_mode=mode,
                    level=level,
                    **dict(zip(cols, keys)),
                    **metrics(g.ground_truth_binary, g.score, t),
                )
                r["evaluation_role"] = (
                    "training_resubstitution"
                    if r["split"] == "training_WSIs"
                    else (
                        "validation_used_for_selection"
                        if r["split"] == "internal_val_WSIs"
                        else "frozen_test"
                    )
                )
                rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(R / "results" / (model + "_metrics.csv"), index=False)
    curve_dir = R / "results/curves"
    curve_dir.mkdir(exist_ok=True)
    for s, g in labelled.groupby("split"):
        if not np.isfinite(g.score).all():
            continue
        y = g.ground_truth_binary.to_numpy(int)
        v = g.score.to_numpy()
        fpr, tpr, rt = roc_curve(y, v, drop_intermediate=False)
        pp, rr, pt = precision_recall_curve(y, v, drop_intermediate=False)
        np.savez_compressed(
            curve_dir / (model + "_" + s + ".npz"),
            roc_fpr=fpr,
            roc_tpr=tpr,
            roc_thresholds=rt,
            pr_precision=pp,
            pr_recall=rr,
            pr_thresholds=pt,
        )
        pd.DataFrame(dict(FPR=fpr, TPR=tpr, threshold=rt)).to_csv(
            curve_dir / (model + "_" + s + "_ROC.csv.gz"), index=False
        )
        pd.DataFrame(dict(precision=pp, recall=rr, threshold=np.r_[pt, np.nan])).to_csv(
            curve_dir / (model + "_" + s + "_PR.csv.gz"), index=False
        )
    print(
        "EVALUATED",
        model,
        table.query("level=='split'")[
            ["split", "threshold_mode", "precision", "recall", "F1", "AP"]
        ].to_string(index=False),
        flush=True,
    )
    return table


def evaluate_cached(model):
    source_paths = (
        [E / "audit/cells.parquet"]
        if model == "GMM"
        else sorted(
            p
            for p in (R / "predictions" / model).glob("*.parquet")
            if p.stem != "all_cells"
        )
    )
    signature = dict(
        model=model,
        code_sha256=sha(__file__),
        protocol_sha256=sha(R / "NIMBUS_PROTOCOL.json"),
        cell_source_sha256=sha(PREP / "artifacts/cells.parquet"),
        prediction_sources={str(p): sha(p) for p in source_paths},
    )
    cache = R / "provenance" / (model + "_evaluation_complete.json")
    if cache.exists():
        state = json.loads(cache.read_text())
        assert (
            state["signature"] == signature
        ), "evaluation cache input/code/config mismatch"
        for path, digest in state["output_sha256"].items():
            assert sha(path) == digest, "evaluation output changed: " + path
        print("REUSE_VERIFIED_EVALUATION", model, flush=True)
        return pd.read_csv(R / "results" / (model + "_metrics.csv"))
    result = evaluate(model)
    outputs = [
        R / "results" / (model + "_metrics.csv"),
        R / "models" / (model + "_thresholds.json"),
        R / "predictions" / model / "all_cells.parquet",
        R / "predictions" / model / "all_cells.csv.gz",
    ]
    grid = R / "results" / (model + "_validation_threshold_grid.csv")
    if grid.exists():
        outputs.append(grid)
    outputs += sorted((R / "results/curves").glob(model + "_*"))
    dump(
        cache,
        dict(
            signature=signature,
            output_sha256={str(p): sha(p) for p in outputs},
            all_cells=1817783,
        ),
    )
    return result


def combine():
    files = [
        R / "results" / (m + "_metrics.csv")
        for m in ["GMM", "Nimbus_outerP1P99_pretrained", "Nimbus_outerP1P99_finetuned"]
    ]
    tables = [pd.read_csv(p) for p in files if p.exists()]
    t = pd.concat(tables, ignore_index=True)
    t.to_csv(R / "results/all_metrics.csv", index=False)
    micro = t.query("level=='split'")
    micro.to_csv(R / "results/overall_metrics.csv", index=False)
    macro = (
        t.query("level=='image'")
        .groupby(["model", "threshold_mode", "split"])
        .agg(
            n_units=("image_id", "size"),
            n_F1_defined=("F1", "count"),
            n_AUC_defined=("ROC_AUC", "count"),
            n_AP_defined=("AP", "count"),
            F1_mean=("F1", "mean"),
            F1_median=("F1", "median"),
            F1_min=("F1", "min"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            AP_mean=("AP", "mean"),
            n_no_positive_predictions=(
                "predicted_positive_fraction",
                lambda x: int((x == 0).sum()),
            ),
        )
    )
    macro.to_csv(R / "results/unit_equal_weight_summary.csv")
    base = t.query("model=='GMM'").drop(columns=["model"])
    diffs = []
    for model in ["Nimbus_outerP1P99_pretrained", "Nimbus_outerP1P99_finetuned"]:
        a = t.query("model==@model").drop(columns=["model"])
        by = ["level", "split", "context", "image_id", "threshold_mode"]
        m = a.merge(base, on=by, suffixes=("_Nimbus", "_GMM"))
        for k in [
            "TP",
            "FP",
            "FN",
            "TN",
            "precision",
            "recall",
            "F1",
            "AP",
            "ROC_AUC",
            "positive_fraction_absolute_error",
        ]:
            m["delta_" + k] = m[k + "_Nimbus"] - m[k + "_GMM"]
        m.insert(0, "model", model)
        diffs.append(m)
    pd.concat(diffs, ignore_index=True).to_csv(
        R / "results/paired_vs_GMM.csv", index=False
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=[
            "Nimbus_outerP1P99_pretrained",
            "Nimbus_outerP1P99_finetuned",
            "GMM",
            "all",
        ],
        default="all",
    )
    a = ap.parse_args()
    for model in (
        ["GMM", "Nimbus_outerP1P99_pretrained", "Nimbus_outerP1P99_finetuned"]
        if a.model == "all"
        else [a.model]
    ):
        evaluate_cached(model)
    combine()
