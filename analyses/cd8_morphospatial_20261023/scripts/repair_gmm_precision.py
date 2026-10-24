"""Historical correction for round-trip CSV parsing; preserves the original GMM."""

from train_compare_v2 import *
import shutil


def main():
    protocol = json.loads((A / "TRAINING_PROTOCOL.json").read_text())
    params = protocol["parameters"]
    sets = {"intensity_8": WHOLE, "morphospatial_90": FEATURES}
    prediction = A / "PhenoBIC_CD8_model_predictions.parquet"
    data = pd.read_parquet(prediction)
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
                "rescale_score",
                "prediction_binary",
            ],
            float_precision="round_trip",
        )
        ann["dataset_id"] = dataset_id
        ann["image_id"] = image_id
        ann = ann.rename(
            columns={
                "rescale_score": "roundtrip_score",
                "prediction_binary": "roundtrip_prediction",
            }
        )
        annotations.append(ann)
    keys = ["dataset_id", "image_id", "cell_id"]
    data = data.merge(
        pd.concat(annotations, ignore_index=True),
        on=keys,
        validate="one_to_one",
        how="left",
    )
    assert (
        len(data) == 1817783
        and not data[keys].duplicated().any()
        and data.roundtrip_score.notna().all()
    )
    assert np.array_equal(data.GMM_prediction, data.roundtrip_prediction)
    error = np.abs(data.GMM_score - data.roundtrip_score)
    diff = dict(
        changed_scores=int((error > 0).sum()),
        max_absolute_difference=float(error.max()),
        positive_predictions_changed=0,
        reason="pandas default CSV float parsing changes low bits and tied ranking; round_trip reproduces original metrics exactly",
        models_refitted=False,
    )
    feature = pd.read_parquet(
        A / "PhenoBIC_CD8_features.parquet",
        columns=keys + ["observed_pixel_count", "observed_mean_CD8"],
    )
    verify = feature.merge(
        data[keys + ["pixel_count", "mean_CD8"]],
        on=keys,
        validate="one_to_one",
        how="inner",
    )
    assert len(verify) == 1817783 and np.array_equal(
        verify.observed_pixel_count, verify.pixel_count
    )
    assert np.allclose(
        verify.observed_mean_CD8, verify.mean_CD8, rtol=1e-10, atol=1e-10
    )
    data["GMM_score"] = data.roundtrip_score
    data = data.drop(
        columns=["roundtrip_score", "roundtrip_prediction", "pixel_count", "mean_CD8"]
    )
    labeled = data.ground_truth_binary.isin([0, 1])
    train = labeled & (data.split == "training_WSIs")
    assert train.sum() == 101057 and labeled.sum() == 1077669
    for name, cols in sets.items():
        bundle = joblib.load(A / "models" / (name + ".joblib"))
        assert bundle["features"] == cols and bundle["protocol"] == protocol
        assert len(bundle["training_image_ids"]) == 3 and all(
            x.startswith("training_WSIs/") for x in bundle["training_image_ids"]
        )
    archive = A / "results/csv_precision_initial"
    archive.mkdir(exist_ok=False)
    for f in (
        list((A / "results").glob("*.csv"))
        + list((A / "results").glob("*_curves.npz"))
        + list((A / "results").glob("curve_index.json"))
    ):
        if f.name.startswith(("feature_", "qc_", "training_")):
            continue
        shutil.copy2(f, archive / f.name)
    backup = A / "PhenoBIC_CD8_model_predictions_initial_csv_precision.parquet"
    assert not backup.exists()
    prediction.rename(backup)
    (A / "results/CSV_PRECISION_AUDIT.json").write_text(json.dumps(diff, indent=2))
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
    previous = pd.read_csv(
        G / "results/phenobic/metrics_summary.csv", float_precision="round_trip"
    )
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
