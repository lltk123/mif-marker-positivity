"""Historical baseline training entry point; the v2 script is the final reported run."""

import os

os.environ.update(
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    NUMEXPR_NUM_THREADS="2",
)
from pathlib import Path
import json, csv, time, sys, hashlib, traceback
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
)
from mif_features import FEATURES, WHOLE, COLUMNS, META_STR, META_FLOAT, COUNTS, QC

A = Path(__file__).resolve().parents[1]
G = A.parent / "cd8_scimap_gmm_20261023"
SCHEMA = pa.schema(
    [
        (
            c,
            pa.string()
            if c in META_STR
            else pa.int64()
            if c in COUNTS + ["cell_id"]
            else pa.bool_()
            if c in QC
            else pa.float64(),
        )
        for c in COLUMNS
    ]
)


def metric(y, score, pred):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pred = np.asarray(pred, int)
    n = len(y)
    assert np.isfinite(score).all()
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return dict(
        n=n,
        n_positive=int(y.sum()),
        ROC_AUC=roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        AP=average_precision_score(y, score) if y.sum() else np.nan,
        F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        precision=precision,
        recall=recall,
        specificity=specificity,
        accuracy=(tp + tn) / n if n else np.nan,
        balanced_accuracy=(recall + specificity) / 2,
        MCC=matthews_corrcoef(y, pred) if n else np.nan,
        TP=tp,
        FP=fp,
        FN=fn,
        TN=tn,
        true_prevalence=float(y.mean()),
        predicted_prevalence=float(pred.mean()),
    )


def main():
    marker = A / "features/FEATURES_COMPLETE.json"
    for _ in range(360):
        if marker.exists():
            break
        log = (A / "logs/full_features.log").read_text(errors="replace")
        if "Traceback (most recent call last)" in log:
            raise RuntimeError("Extraction failed; see full_features.log")
        time.sleep(10)
    else:
        raise RuntimeError("Features not complete after one hour; no model trained")
    protocol = json.loads((A / "TRAINING_PROTOCOL.json").read_text())
    assert not protocol["hyperparameter_search"] and not protocol["threshold_tuning"]
    files = sorted((A / "features").glob("*.parquet"))
    dataset = ds.dataset([str(p) for p in files], format="parquet", schema=SCHEMA)
    consolidated = A / "PhenoBIC_CD8_features.parquet"
    if not consolidated.exists():
        count = 0
        with pq.ParquetWriter(consolidated, SCHEMA, compression="zstd") as writer:
            for batch in dataset.scanner(batch_size=65536).to_batches():
                writer.write_batch(batch)
                count += batch.num_rows
        assert count == 1817783
    assert pq.read_metadata(consolidated).num_rows == 1817783
    columns = [
        "dataset_id",
        "image_id",
        "cell_id",
        "source_image_id",
        "split",
        "context",
        "observed_pixel_count",
        "observed_mean_CD8",
        "qc_truncated",
        "qc_disconnected",
    ] + FEATURES
    data = pd.read_parquet(consolidated, columns=columns)
    assert not data[["dataset_id", "image_id", "cell_id"]].duplicated().any()
    assert set(data.dataset_id) == {"PhenoBIC-Data"}
    # Labels and the pre-existing GMM predictions are read only after feature generation is complete.
    annotations = []
    for r in csv.DictReader((G / "results/phenobic/cell_table_index.csv").open()):
        dataset_id, split, context, unit = r["unit_id"].split("/")
        image_id = (
            split
            + "/"
            + context
            + ("/" + unit if split == "external_val_TMA_cores" else "")
        )
        ann = pd.read_csv(
            G / r["file"],
            usecols=[
                "cell_id",
                "pixel_count",
                "mean_CD8",
                "ground_truth_original",
                "ground_truth_binary",
                "rescale_score",
                "prediction_binary",
            ],
        )
        ann["dataset_id"] = dataset_id
        ann["image_id"] = image_id
        ann = ann.rename(
            columns={
                "pixel_count": "gmm_pixel_count",
                "mean_CD8": "gmm_mean_CD8",
                "rescale_score": "GMM_score",
                "prediction_binary": "GMM_prediction",
            }
        )
        annotations.append(ann)
    annotation = pd.concat(annotations, ignore_index=True)
    assert not annotation[["dataset_id", "image_id", "cell_id"]].duplicated().any()
    data = data.merge(
        annotation,
        on=["dataset_id", "image_id", "cell_id"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    assert len(data) == 1817783 and (data["_merge"] == "both").all()
    data = data.drop(columns="_merge")
    assert (data.observed_pixel_count == data.gmm_pixel_count).all()
    assert np.allclose(
        data.observed_mean_CD8, data.gmm_mean_CD8, rtol=1e-10, atol=1e-10
    )
    labeled = data.ground_truth_binary.isin([0, 1])
    train = labeled & (data.split == "training_WSIs")
    assert int(train.sum()) == 101057 and int(labeled.sum()) == 1077669
    y = data.loc[train, "ground_truth_binary"].to_numpy(int)
    groups = data.loc[train, "image_id"].to_numpy()
    assert len(set(groups)) == 3
    params = protocol["parameters"]
    sets = {"intensity_8": WHOLE, "morphospatial_90": FEATURES}
    assert len(FEATURES) == 90 and not (
        set(FEATURES)
        & set(
            [
                "cell_id",
                "image_id",
                "patient_id",
                "batch_id",
                "context",
                "split",
                "GMM_score",
                "GMM_prediction",
                "ground_truth_binary",
            ]
        )
    )
    train_indices = np.flatnonzero(train.to_numpy())
    foldresults = []
    oofrows = []
    for name, columns in sets.items():
        print(
            "FIT MODEL",
            name,
            "features",
            len(columns),
            "training cells",
            len(y),
            flush=True,
        )
        X = data.loc[train, columns].to_numpy(float)
        oof = np.full(len(y), np.nan)
        for heldout in sorted(set(groups)):
            fit = groups != heldout
            val = ~fit
            model = HistGradientBoostingClassifier(**params)
            start = time.time()
            model.fit(X[fit], y[fit])
            oof[val] = model.predict_proba(X[val])[:, 1]
            score = metric(y[val], oof[val], oof[val] > 0.5)
            score.update(
                model=name,
                heldout_image_id=heldout,
                training_cells=int(fit.sum()),
                seconds=time.time() - start,
            )
            foldresults.append(score)
            print(
                "OOF FOLD",
                name,
                heldout,
                json.dumps(
                    {
                        k: score[k]
                        for k in [
                            "n",
                            "ROC_AUC",
                            "AP",
                            "F1",
                            "precision",
                            "recall",
                            "seconds",
                        ]
                    }
                ),
                flush=True,
            )
        assert np.isfinite(oof).all()
        m = metric(y, oof, oof > 0.5)
        m.update(model=name, split="training_WSI_out_of_fold")
        oofrows.append(m)
        model = HistGradientBoostingClassifier(**params)
        model.fit(X, y)
        joblib.dump(
            dict(
                estimator=model,
                features=columns,
                positive_threshold=0.5,
                positive_rule="probability > 0.5",
                training_image_ids=sorted(set(groups)),
                protocol=protocol,
            ),
            A / "models" / (name + ".joblib"),
            compress=3,
        )
        pred = np.empty(len(data), float)
        for start in range(0, len(data), 100000):
            stop = min(start + 100000, len(data))
            pred[start:stop] = model.predict_proba(
                data.iloc[start:stop][columns].to_numpy(float)
            )[:, 1]
        data[name + "_score"] = pred
        data[name + "_prediction"] = (pred > 0.5).astype(np.int8)
        oofdf = data.iloc[train_indices][
            ["dataset_id", "image_id", "cell_id", "ground_truth_binary"]
        ].copy()
        oofdf["score"] = oof
        oofdf["prediction"] = (oof > 0.5).astype(np.int8)
        oofdf.to_parquet(A / "results" / (name + "_training_oof.parquet"), index=False)
    pd.DataFrame(foldresults).to_csv(
        A / "results/training_oof_per_WSI.csv", index=False
    )
    pd.DataFrame(oofrows).to_csv(A / "results/training_oof_summary.csv", index=False)
    models = {
        "GMM": ("GMM_score", "GMM_prediction"),
        "intensity_8": ("intensity_8_score", "intensity_8_prediction"),
        "morphospatial_90": ("morphospatial_90_score", "morphospatial_90_prediction"),
    }
    summary = []
    perunit = []
    contexts = []
    curves = []
    qcmetrics = []
    evaldata = data[labeled]
    for model, (sc, pr) in models.items():
        for split, d in evaldata.groupby("split", sort=False):
            y = d.ground_truth_binary.to_numpy(int)
            s = d[sc].to_numpy(float)
            pred = d[pr].to_numpy(int)
            m = metric(y, s, pred)
            m.update(
                model=model,
                split=split,
                n_units=d.image_id.nunique(),
                training_resubstitution=(split == "training_WSIs" and model != "GMM"),
            )
            summary.append(m)
            fpr, tpr, rt = roc_curve(y, s)
            precision, recall, pt = precision_recall_curve(y, s)
            file = model + "_" + split + "_curves.npz"
            np.savez_compressed(
                A / "results" / file,
                fpr=fpr,
                tpr=tpr,
                precision=precision,
                recall=recall,
            )
            curves.append(dict(model=model, split=split, file=file, n=len(d)))
            print("EVALUATION", json.dumps(m), flush=True)
        for image_id, d in evaldata.groupby("image_id", sort=False):
            m = metric(d.ground_truth_binary, d[sc], d[pr])
            m.update(
                model=model,
                image_id=image_id,
                split=d.split.iloc[0],
                context=d.context.iloc[0],
            )
            perunit.append(m)
        for (split, context), d in evaldata.groupby(["split", "context"], sort=False):
            m = metric(d.ground_truth_binary, d[sc], d[pr])
            m.update(model=model, split=split, context=context)
            contexts.append(m)
        for (split, flag), d in evaldata.groupby(["split", "qc_truncated"], sort=False):
            if len(d):
                m = metric(d.ground_truth_binary, d[sc], d[pr])
                m.update(model=model, split=split, qc_truncated=bool(flag))
                qcmetrics.append(m)
    summarydf = pd.DataFrame(summary)
    summarydf.to_csv(A / "results/model_comparison.csv", index=False)
    unitdf = pd.DataFrame(perunit)
    unitdf.to_csv(A / "results/metrics_per_unit.csv", index=False)
    pd.DataFrame(contexts).to_csv(A / "results/metrics_per_context.csv", index=False)
    pd.DataFrame(qcmetrics).to_csv(
        A / "results/metrics_by_truncation_QC.csv", index=False
    )
    (A / "results/curve_index.json").write_text(json.dumps(curves, indent=2))
    macro = unitdf.groupby(["model", "split"])[
        ["ROC_AUC", "AP", "F1", "precision", "recall"]
    ].agg(["mean", "std", "min", "max"])
    macro.to_csv(A / "results/metrics_macro.csv")
    keep = [
        "dataset_id",
        "image_id",
        "cell_id",
        "source_image_id",
        "split",
        "context",
        "ground_truth_original",
        "ground_truth_binary",
        "qc_truncated",
        "qc_disconnected",
    ] + [c for pair in models.values() for c in pair]
    data[keep].to_parquet(
        A / "PhenoBIC_CD8_model_predictions.parquet", index=False, compression="zstd"
    )
    # Audit the baseline against the preceding completed run; no re-fitting or changed threshold.
    previous = pd.read_csv(G / "results/phenobic/metrics_summary.csv")
    previous = previous[previous.level == "split"]
    for _, r in summarydf[summarydf.model == "GMM"].iterrows():
        old = previous[previous.group == r["split"]].iloc[0]
        for col in [
            "ROC_AUC",
            "AP",
            "F1",
            "precision",
            "recall",
            "TP",
            "FP",
            "FN",
            "TN",
        ]:
            assert np.isclose(r[col], old[col], rtol=1e-12, atol=1e-12), (
                col,
                r[col],
                old[col],
            )
    audit = dict(
        feature_rows=len(data),
        unique_keys=True,
        all_GMM_pixel_counts_match=True,
        all_GMM_raw_means_match=True,
        training_cells=int(train.sum()),
        evaluation_labelled_cells=int(labeled.sum()),
        GMM_baseline_reproduced=True,
        metadata_and_outcomes_excluded_from_features=True,
        models=list(sets),
        supervised_positive_threshold=0.5,
        model_parameters=params,
        pan_gold_accessed=False,
    )
    (A / "results/MODEL_COMPLETE.json").write_text(json.dumps(audit, indent=2))
    print("MODEL COMPLETE", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
