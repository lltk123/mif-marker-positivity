"""Research analyze cd8 for the frozen PhenoBIC CD8 experiment."""

import os

os.environ.update(
    OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    NUMEXPR_NUM_THREADS="2",
)
from pathlib import Path
import json, csv, sys, time, hashlib, platform, traceback, importlib
from collections import defaultdict
import numpy as np, pandas as pd, anndata as ad
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
)
import scimap_rescale_upstream as sm

P = Path("/mnt/disk3/mIHC/mif_marker_positivity_20261022")
A = P / "analyses/cd8_scimap_gmm_20261023"
CONFIG = dict(
    marker="CD8",
    feature="Arithmetic mean of original channel pixels within the original nonzero cell mask; float64 accumulation",
    gmm_components=3,
    random_state=0,
    log=True,
    clipping_percentiles=[0.01, 99.99],
    positive_rule="upstream rescale score > 0.5; exactly 0.5 assigned negative",
    fit_population="All segmented cells with finite nonnegative mean, irrespective of annotation",
    unit="PhenoBIC: each original WSI (internal tiles reconstructed without duplicated overlap pixels) or each TMA core; Pan-M: each FOV",
    evaluation="Original splits preserved; no label tuning. Per-unit unsupervised calibration on target-image intensities is transductive, not frozen-threshold inference.",
    ranking_metrics="ROC-AUC and average precision use continuous SCIMAP rescaled scores; scores are not calibrated probabilities.",
    pan_ambiguous="Original 2/3 retained in prediction tables and counted separately; undefined as binary ground truth, so do not enter 0/1 metrics.",
    split_policy="PhenoBIC development benchmark, including original train/internal/external partitions; Pan-M separate external validation; no random resplit",
    statistics="Cell-weighted pooled confusion statistics and ROC/AP; equal-weight unit macro mean/std; descriptive only, no cell-independence p-values or CIs",
    source_commit="e8f1609fbcf838e265541983af7e63490e5c7a13",
)


class AuditGMM(GaussianMixture):
    last = None

    def fit(self, *args, **kwargs):
        out = super().fit(*args, **kwargs)
        AuditGMM.last = self
        return out


sm.GaussianMixture = AuditGMM


def metrics(df):
    d = df[df.ground_truth_binary.isin([0, 1]) & df.rescale_score.notna()]
    y = d.ground_truth_binary.to_numpy(dtype=int)
    pred = d.prediction_binary.to_numpy(dtype=int)
    s = d.rescale_score.to_numpy()
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    n = len(y)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else np.nan
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    ap = average_precision_score(y, s) if y.sum() > 0 else np.nan
    return dict(
        n_evaluated=n,
        n_positive=int(y.sum()),
        n_negative=int(n - y.sum()),
        TP=tp,
        FP=fp,
        FN=fn,
        TN=tn,
        ROC_AUC=auc,
        AP=ap,
        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        precision=precision,
        recall=recall,
        specificity=specificity,
        accuracy=(tp + tn) / n if n else np.nan,
        balanced_accuracy=(recall + specificity) / 2,
        MCC=matthews_corrcoef(y, pred) if n else np.nan,
        true_prevalence=float(y.mean()) if n else np.nan,
        predicted_prevalence=float(pred.mean()) if n else np.nan,
        prevalence_error=float(pred.mean() - y.mean()) if n else np.nan,
    )


def unit_features(entries):
    frames = []
    observations = []
    for e in entries:
        z = np.load(A / "features" / (e["key"] + ".npz"))
        df = pd.DataFrame(
            {
                "cell_id": z["cell_id"],
                "pixel_count": z["owned_pixel_count"],
                "intensity_sum": z["owned_intensity_sum"],
                "x_sum": z["owned_x_sum"],
                "y_sum": z["owned_y_sum"],
                "nonfinite_pixels": z["nonfinite_pixels"],
            }
        )
        frames.append(df)
        if len(entries) > 1 and len(df):
            observations.append(
                pd.DataFrame(
                    {
                        "cell_id": z["cell_id"],
                        "cx": z["x_sum"] / z["pixel_count"],
                        "cy": z["y_sum"] / z["pixel_count"],
                        "patch": e["patch"],
                    }
                )
            )
    full = pd.concat(frames, ignore_index=True)
    spans = {}
    duplicates = 0
    if observations:
        obs = pd.concat(observations, ignore_index=True)
        g = obs.groupby("cell_id")
        sizes = g.size()
        duplicates = int((sizes > 1).sum())
        span = g[["cx", "cy"]].max() - g[["cx", "cy"]].min()
        spans = dict(
            duplicated_cell_ids=duplicates,
            max_centroid_span_x=float(span.cx.max()),
            max_centroid_span_y=float(span.cy.max()),
            ids_with_centroid_span_over_500=int((span.max(axis=1) > 500).sum()),
        )
        if spans["ids_with_centroid_span_over_500"]:
            raise RuntimeError("Potential local/global ID collision " + str(spans))
    df = full.groupby("cell_id", sort=True).sum().reset_index()
    obs_counts = full.groupby("cell_id").size()
    df["n_tile_observations"] = df.cell_id.map(obs_counts).astype(int)
    df = df[df.pixel_count > 0].copy()
    df["mean_CD8"] = df.intensity_sum / df.pixel_count
    df["centroid_x"] = df.x_sum / df.pixel_count
    df["centroid_y"] = df.y_sum / df.pixel_count
    df["feature_valid"] = (
        (df.nonfinite_pixels == 0) & np.isfinite(df.mean_CD8) & (df.mean_CD8 >= 0)
    )
    return df, spans


def labels_for(entries):
    e = entries[0]
    root = P / "datasets" / e["dataset"]
    r = e["source"]
    gt = pd.read_csv(root / r["labels"])
    if e["dataset"] == "PhenoBIC-Data":
        if r["label_patch_filter"]:
            gt = gt[gt.Patch.astype(str) == r["label_patch_filter"]].copy()
        gt = gt.rename(columns={"label": "cell_id", "CD8": "ground_truth_original"})
        gt["ground_truth_binary"] = gt.ground_truth_original.map({"neg": 0, "pos": 1})
    else:
        gt = gt[
            (gt.dataset == e["split"]) & (gt.fov == e["patch"]) & (gt.channel == "CD8")
        ].copy()
        gt = gt.rename(columns={"activity": "ground_truth_original"})
        gt["ground_truth_binary"] = gt.ground_truth_original.where(
            gt.ground_truth_original.isin([0, 1])
        )
    gt = gt[["cell_id", "ground_truth_original", "ground_truth_binary"]]
    if gt.cell_id.duplicated().any():
        raise RuntimeError("Duplicate ground truth keys")
    return gt


def run(dataset, audit):
    prefix = "phenobic" if dataset == "PhenoBIC-Data" else "pan_gold"
    out = A / "results" / prefix
    out.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    for e in audit:
        unit = (
            e["context"]
            if e["split"] in ("training_WSIs", "internal_val_WSIs")
            else e["patch"]
        )
        grouped[(e["split"], e["context"], unit)].append(e)
    unitmetrics = []
    gates = []
    coverage = []
    all_eval = []
    intensity = []
    unitfiles = []
    for (split, context, unit), entries in grouped.items():
        uid = dataset + "/" + split + "/" + context + "/" + unit
        key = hashlib.sha256(uid.encode()).hexdigest()[:20]
        print("UNIT", uid, flush=True)
        df, spans = unit_features(entries)
        df["unit_id"] = uid
        df["split"] = split
        df["context"] = context
        df["unit"] = unit
        valid = df.feature_valid
        x = df.loc[valid, "mean_CD8"].to_numpy(dtype=np.float64)
        if len(x) < 3 or len(np.unique(x)) < 3:
            raise RuntimeError(
                "Insufficient distinct intensities for 3-component GMM " + uid
            )
        ann = ad.AnnData(
            x[:, None],
            obs=pd.DataFrame(
                {"imageid": unit},
                index=["cell_" + str(i) for i in df.loc[valid, "cell_id"]],
            ),
            var=pd.DataFrame(index=["CD8"]),
            dtype=np.float64,
        )
        # The unchanged upstream function is called once per unit to avoid its multi-image gate-mapping defect.
        ann = sm.rescale(
            ann,
            gate=None,
            log=True,
            imageid="imageid",
            failed_markers=None,
            method="all",
            verbose=False,
            random_state=0,
            gmm_components=3,
        )
        scores = np.asarray(ann.X).ravel()
        gate = float(ann.uns["gates"].iloc[0, 0])
        gm = AuditGMM.last
        assert (
            np.isfinite(scores).all()
            and scores.min() >= 0
            and scores.max() <= 1.0000000001
        )
        assert np.isclose(gate, np.sort(gm.means_.ravel())[-2:].mean())
        df["rescale_score"] = np.nan
        df.loc[valid, "rescale_score"] = scores
        df["prediction_binary"] = np.nan
        df.loc[valid, "prediction_binary"] = (scores > 0.5).astype(int)
        df["prediction_CD8"] = df.prediction_binary.map({0: "neg", 1: "pos"})
        gateinfo = dict(
            unit_id=uid,
            split=split,
            context=context,
            unit=unit,
            n_fit=len(x),
            gate_log1p=gate,
            gate_raw=float(np.expm1(gate)),
            nearest_observed_gate_raw=float(x[np.abs(np.log1p(x) - gate).argmin()]),
            component_means_log1p=gm.means_.ravel().tolist(),
            component_weights=gm.weights_.tolist(),
            component_variances=gm.covariances_.ravel().tolist(),
            converged=bool(gm.converged_),
            iterations=int(gm.n_iter_),
            score_equal_0_5=int((scores == 0.5).sum()),
            direct_threshold_vs_score_disagreements=int(
                ((np.log1p(x) > gate) != (scores > 0.5)).sum()
            ),
            **spans,
        )
        gates.append(gateinfo)
        if not gm.converged_:
            raise RuntimeError("GMM did not converge " + uid)
        # Human labels are accessed only after unsupervised fitting and scores are final.
        gt = labels_for(entries)
        joined = df.merge(
            gt, on="cell_id", how="outer", validate="one_to_one", indicator=True
        )
        joined["record_status"] = (
            joined["_merge"]
            .map(
                {
                    "both": "matched",
                    "left_only": "segmented_unannotated",
                    "right_only": "annotation_without_mask",
                }
            )
            .astype(str)
        )
        joined = joined.drop(columns="_merge")
        for col, val in [
            ("unit_id", uid),
            ("split", split),
            ("context", context),
            ("unit", unit),
        ]:
            joined[col] = val
        joined["eligible_binary_evaluation"] = (
            joined.ground_truth_binary.isin([0, 1]) & joined.rescale_score.notna()
        )
        filename = key + "_cells.csv.gz"
        joined.to_csv(out / filename, index=False, compression="gzip")
        unitfiles.append(
            dict(
                unit_id=uid,
                file="results/" + prefix + "/" + filename,
                n_records=len(joined),
            )
        )
        c = dict(
            unit_id=uid,
            split=split,
            context=context,
            unit=unit,
            n_segmented=len(df),
            n_fit=len(x),
            n_annotations=len(gt),
            matched_annotations=int((joined.record_status == "matched").sum()),
            annotations_without_mask=int(
                (joined.record_status == "annotation_without_mask").sum()
            ),
            segmented_unannotated=int(
                (joined.record_status == "segmented_unannotated").sum()
            ),
            ambiguous_or_other_annotations=int((gt.ground_truth_binary.isna()).sum()),
            invalid_features=int((~valid).sum()),
            n_binary_evaluated=int(joined.eligible_binary_evaluation.sum()),
            duplicated_tile_cell_ids=spans.get("duplicated_cell_ids", 0),
        )
        coverage.append(c)
        m = metrics(joined)
        m.update(unit_id=uid, split=split, context=context, unit=unit)
        unitmetrics.append(m)
        ev = joined[joined.eligible_binary_evaluation][
            [
                "unit_id",
                "split",
                "context",
                "cell_id",
                "mean_CD8",
                "ground_truth_binary",
                "rescale_score",
                "prediction_binary",
            ]
        ].copy()
        all_eval.append(ev)
        for label in [0, 1]:
            vals = ev.loc[ev.ground_truth_binary == label, "mean_CD8"]
            intensity.append(
                dict(
                    unit_id=uid,
                    split=split,
                    context=context,
                    ground_truth=label,
                    n=len(vals),
                    mean=vals.mean(),
                    std=vals.std(),
                    q25=vals.quantile(0.25),
                    median=vals.median(),
                    q75=vals.quantile(0.75),
                )
            )
        print(
            "METRICS",
            json.dumps(
                {
                    k: m[k]
                    for k in [
                        "n_evaluated",
                        "ROC_AUC",
                        "AP",
                        "F1",
                        "precision",
                        "recall",
                    ]
                }
            ),
            flush=True,
        )
    evaldata = pd.concat(all_eval, ignore_index=True)
    um = pd.DataFrame(unitmetrics)
    um.to_csv(out / "metrics_per_unit.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "annotation_coverage.csv", index=False)
    pd.DataFrame(intensity).to_csv(out / "intensity_by_ground_truth.csv", index=False)
    pd.DataFrame(unitfiles).to_csv(out / "cell_table_index.csv", index=False)
    (out / "gmm_gates.json").write_text(json.dumps(gates, indent=2))
    pd.DataFrame(
        [{k: v for k, v in x.items() if not isinstance(v, list)} for x in gates]
    ).to_csv(out / "gmm_gates.csv", index=False)
    groups = [("all", prefix, evaldata)]
    groups.extend(("split", s, g) for s, g in evaldata.groupby("split", sort=False))
    groups.extend(
        ("context", s + "/" + c, g)
        for (s, c), g in evaldata.groupby(["split", "context"], sort=False)
    )
    summary = []
    macro = []
    curve_index = []
    metriccols = [
        "ROC_AUC",
        "AP",
        "F1",
        "precision",
        "recall",
        "specificity",
        "accuracy",
        "balanced_accuracy",
        "MCC",
    ]
    for level, name, d in groups:
        m = metrics(d)
        m.update(level=level, group=name, n_units=d.unit_id.nunique())
        summary.append(m)
        per = um[um.unit_id.isin(d.unit_id.unique())]
        for col in metriccols:
            v = per[col]
            macro.append(
                dict(
                    level=level,
                    group=name,
                    metric=col,
                    n_units=len(v),
                    mean=v.mean(),
                    std=v.std(),
                    median=v.median(),
                    min=v.min(),
                    max=v.max(),
                )
            )
        if level in ("all", "split"):
            y = d.ground_truth_binary.to_numpy(dtype=int)
            s = d.rescale_score.to_numpy()
            fpr, tpr, rt = roc_curve(y, s)
            prec, rec, pt = precision_recall_curve(y, s)
            key = hashlib.sha256((level + "/" + name).encode()).hexdigest()[:12]
            np.savez_compressed(
                out / (key + "_curves.npz"),
                fpr=fpr,
                tpr=tpr,
                roc_threshold=rt,
                precision=prec,
                recall=rec,
                pr_threshold=pt,
            )
            curve_index.append(
                dict(
                    level=level,
                    group=name,
                    file=key + "_curves.npz",
                    prevalence=m["true_prevalence"],
                    n=m["n_evaluated"],
                )
            )
    pd.DataFrame(summary).to_csv(out / "metrics_summary.csv", index=False)
    pd.DataFrame(macro).to_csv(out / "metrics_macro.csv", index=False)
    (out / "curve_index.json").write_text(json.dumps(curve_index, indent=2))
    status = dict(
        dataset=dataset,
        status="complete",
        n_units=len(unitmetrics),
        n_evaluated=len(evaldata),
        n_segmented=sum(c["n_segmented"] for c in coverage),
        missing_annotations=sum(c["annotations_without_mask"] for c in coverage),
        ambiguous_or_other_annotations=sum(
            c["ambiguous_or_other_annotations"] for c in coverage
        ),
        total_annotations=sum(c["n_annotations"] for c in coverage),
        all_gmms_converged=all(g["converged"] for g in gates),
        config=CONFIG,
    )
    (out / "STATUS.json").write_text(json.dumps(status, indent=2))
    print("ANALYSIS COMPLETE", prefix, json.dumps(status), flush=True)
    return status


if __name__ == "__main__":
    (A / "provenance/analysis_plan.json").write_text(json.dumps(CONFIG, indent=2))
    versions = {"python": sys.version, "platform": platform.platform()}
    for name in ["numpy", "pandas", "scipy", "sklearn", "anndata", "tifffile", "h5py"]:
        versions[name] = importlib.import_module(name).__version__
    (A / "provenance/runtime_versions.json").write_text(json.dumps(versions, indent=2))
    dataset = sys.argv[1] if len(sys.argv) > 1 else "PhenoBIC-Data"
    prefix = "phenobic" if dataset == "PhenoBIC-Data" else "pan_gold"
    marker = (
        A
        / "features"
        / (
            "PHENOBIC_EXTRACTION_COMPLETE.json"
            if prefix == "phenobic"
            else "PAN_EXTRACTION_COMPLETE.json"
        )
    )
    for _ in range(120):
        if marker.exists():
            break
        log = (
            A
            / "logs"
            / ("extraction.log" if prefix == "phenobic" else "pan_extraction.log")
        ).read_text(errors="replace")
        if "Traceback (most recent call last)" in log:
            raise RuntimeError("Extraction failed; see extraction log")
        time.sleep(10)
    else:
        raise RuntimeError("Extraction completion marker timeout")
    audit = json.loads(
        (A / "features" / (prefix + "_extraction_audit.json")).read_text()
    )
    run(dataset, audit)
