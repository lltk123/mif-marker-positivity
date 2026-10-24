"""Final cross-artifact consistency audit; reuses saved scores, never refits a model."""
from common import *


def main():
    assert (R / "provenance/prediction_integrity_passed.json").exists()
    cells = load_cells()
    expected = key_hash(cells)
    rows = []
    for model in ["GMM", "Nimbus_pretrained", "Nimbus_finetuned"]:
        d = pd.read_parquet(R / "predictions" / model / "all_cells.parquet")
        exact_keys(d)
        assert len(d) == 1817783 and key_hash(d) == expected
        assert np.isfinite(d.score).all()
        spec = json.loads((R / "models" / (model + "_thresholds.json")).read_text())
        for mode, threshold in [
            ("official_default", 0.5),
            ("validation_selected", spec["validation_selected"]),
        ]:
            assert np.array_equal(
                d["prediction_" + mode].to_numpy(int),
                (d.score.to_numpy() > threshold).astype(int),
            )
        if model != "GMM":
            meta = json.loads((R / "models" / (model + "_manifest.json")).read_text())
            assert d.model_version.eq(model + "_" + meta["weight_sha256"][:12]).all()
            assert sha(meta["weight_path"]) == meta["weight_sha256"]
        else:
            old = pd.read_parquet(
                E / "audit/cells.parquet",
                columns=KEYS[:3] + ["GMM_score", "GMM_prediction"],
            )
            old["marker"] = "CD8"
            paired = d[KEYS + ["score", "prediction_official_default"]].merge(
                old, on=KEYS, validate="one_to_one"
            )
            mismatch = int(
                (
                    paired.prediction_official_default.to_numpy()
                    != paired.GMM_prediction.to_numpy()
                ).sum()
            )
            delta = float(np.max(np.abs(paired.score - paired.GMM_score)))
            pd.DataFrame(
                [
                    dict(
                        n=len(paired),
                        max_score_difference=delta,
                        default_prediction_mismatches=mismatch,
                        source=str(E / "audit/cells.parquet"),
                        source_sha256=sha(E / "audit/cells.parquet"),
                        refitted=False,
                    )
                ]
            ).to_csv(R / "results/GMM_reuse_audit.csv", index=False)
            assert delta == 0 and mismatch == 0
        gt = d[KEYS + ["ground_truth_binary"]].merge(
            cells[KEYS + ["ground_truth_binary"]],
            on=KEYS,
            validate="one_to_one",
            suffixes=("_result", "_source"),
        )
        a = gt.ground_truth_binary_result
        b = gt.ground_truth_binary_source
        assert ((a == b) | (a.isna() & b.isna())).all()
        for split in ["training_WSIs", "internal_val_WSIs", "external_val_TMA_cores"]:
            z = np.load(R / "results/curves" / (model + "_" + split + ".npz"))
            assert len(z["roc_thresholds"]) == len(z["roc_fpr"]) == len(z["roc_tpr"])
            assert (
                len(z["pr_thresholds"]) + 1
                == len(z["pr_precision"])
                == len(z["pr_recall"])
            )
        rows.append(
            dict(
                model=model,
                n_cells=len(d),
                n_labelled=int(d.ground_truth_binary.isin([0, 1]).sum()),
                key_sha256=expected,
                all_finite=True,
                labels_unchanged=True,
                threshold_predictions_verified=True,
                complete_curve_arrays_verified=True,
            )
        )
    pd.DataFrame(rows).to_csv(R / "results/final_delivery_checks.csv", index=False)
    metrics = pd.read_csv(R / "results/all_metrics.csv")
    assert metrics.coverage.eq(1).all()
    assert (metrics[["TP", "FP", "FN", "TN"]].sum(axis=1) == metrics.n).all()
    hashes = pd.read_csv(R / "provenance/adapter_code_hashes.csv")
    for row in hashes.itertuples():
        assert sha(R / row.path) == row.sha256, (
            "code changed after snapshot: " + row.path
        )
    dump(
        R / "provenance/final_numeric_checks_passed.json",
        dict(
            all_checks_passed=True,
            adapter_commit=json.loads(
                (R / "provenance/adapter_code_commit.json").read_text()
            )["commit"],
            models=3,
            cells_per_model=1817783,
        ),
    )
    print("FINAL_NUMERIC_DELIVERY_CHECKS_PASSED")


if __name__ == "__main__":
    main()
