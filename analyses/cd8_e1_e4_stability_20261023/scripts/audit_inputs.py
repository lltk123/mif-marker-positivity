"""Freeze input identity, baseline provenance, calibration units and split membership."""

import os

os.environ.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
import csv, json, sys, joblib, platform, importlib, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from common import *
from io_adapter import AUDIT, boxes, image_id
from gmm_interface import mapping_from_gate, score_fixed, fit_score


def main():
    assert not (A / "audit/AUDIT_COMPLETE.json").exists()
    print("READ BASELINES", flush=True)
    pred = (
        pd.read_parquet(OLD / "PhenoBIC_CD8_model_predictions.parquet")
        .sort_values(KEYS)
        .reset_index(drop=True)
    )
    assert len(pred) == 1817783 and not pred[KEYS].duplicated().any()
    ann = []
    for r in csv.DictReader((G / "results/phenobic/cell_table_index.csv").open()):
        dataset, split, context, unit = r["unit_id"].split("/")
        assert dataset == "PhenoBIC-Data"
        d = pd.read_csv(
            G / r["file"],
            usecols=[
                "cell_id",
                "pixel_count",
                "mean_CD8",
                "ground_truth_binary",
                "rescale_score",
                "prediction_binary",
            ],
            float_precision="round_trip",
        )
        d["dataset_id"] = dataset
        d["image_id"] = (
            split
            + "/"
            + context
            + ("/" + unit if split == "external_val_TMA_cores" else "")
        )
        d["gmm_unit_id"] = r["unit_id"]
        ann.append(d)
    ann = pd.concat(ann, ignore_index=True)
    pred = pred.merge(
        ann,
        on=KEYS,
        validate="one_to_one",
        how="outer",
        suffixes=("", "_check"),
        indicator=True,
    )
    assert len(pred) == 1817783 and (pred["_merge"] == "both").all()
    pred = pred.drop(columns="_merge")
    assert np.allclose(
        pred.GMM_score, pred.rescale_score, rtol=0, atol=0
    ) and np.array_equal(pred.GMM_prediction, pred.prediction_binary)
    assert np.allclose(
        pred.ground_truth_binary, pred.ground_truth_binary_check, equal_nan=True
    )
    pred = pred.drop(
        columns=["ground_truth_binary_check", "rescale_score", "prediction_binary"]
    )
    expected = {
        "training_WSIs": 101057,
        "internal_val_WSIs": 539132,
        "external_val_TMA_cores": 437480,
    }
    keymeta = []
    for split, g in pred.groupby("split", sort=True):
        lab = g[g.ground_truth_binary.isin([0, 1])]
        assert len(lab) == expected[split]
        for population, d in [("all", g), ("labelled", lab)]:
            path = A / "keys" / (split + "_" + population + ".parquet")
            d[KEYS].sort_values(KEYS).to_parquet(path, index=False)
            keymeta.append(
                dict(
                    split=split,
                    population=population,
                    rows=len(d),
                    canonical_keys_sha256=key_hash(d),
                    file=str(path.relative_to(A)),
                )
            )
    pred[KEYS].sort_values(KEYS).to_parquet(A / "keys/all_cells.parquet", index=False)
    pd.DataFrame(keymeta).to_csv(A / "keys/key_manifest.csv", index=False)
    rng = np.random.RandomState(0)
    samples = []
    for uid, d in pred[pred.ground_truth_binary.isin([0, 1])].groupby(
        "image_id", sort=True
    ):
        d = d.sort_values("cell_id")
        pick = np.sort(rng.choice(len(d), min(200, len(d)), replace=False))
        samples.append(
            d.iloc[pick][
                KEYS + ["split", "context", "source_image_id", "ground_truth_binary"]
            ]
        )
    sample = pd.concat(samples, ignore_index=True).sort_values(KEYS)
    sample.to_parquet(A / "keys/stability_sample.parquet", index=False)
    sample.to_csv(A / "keys/stability_sample.csv", index=False)
    write_json(
        A / "keys/sample_manifest.json",
        dict(
            seed=0,
            algorithm="Single RandomState(0), units lexicographic; uniformly choose without replacement from cell-ID-sorted labelled cells; maximum 200/unit",
            rows=len(sample),
            units=sample.image_id.nunique(),
            key_sha256=key_hash(sample),
        ),
    )
    gates = {
        g["unit_id"]: g
        for g in json.loads((G / "results/phenobic/gmm_gates.json").read_text())
    }
    states = {}
    mappingaudit = []
    for uid, d in pred.groupby("image_id", sort=True):
        d = d.sort_values("cell_id")
        gate = gates[d.gmm_unit_id.iloc[0]]
        assert np.isclose(
            gate["gate_log1p"], np.sort(gate["component_means_log1p"])[-2:].mean()
        )
        state = mapping_from_gate(
            d.mean_CD8.to_numpy(), gate["gate_log1p"], gmm_info=gate
        )
        scores = score_fixed(d.mean_CD8, state)
        error = float(np.max(abs(scores - d.GMM_score.to_numpy())))
        assert error < 1e-12, (uid, error)
        assert np.array_equal(scores > 0.5, d.GMM_prediction.to_numpy().astype(bool))
        states[uid] = state
        mappingaudit.append(
            dict(
                image_id=uid,
                n_full_unit=len(d),
                max_score_absolute_difference=error,
                binary_disagreements=0,
            )
        )
    write_json(A / "calibration/original_gmm_states.json", states)
    pd.DataFrame(mappingaudit).to_csv(A / "audit/GMM_mapping_recovery.csv", index=False)
    print("GMM MAPPINGS VERIFIED", len(states), flush=True)
    ref = pd.read_csv(
        OLD / "results/model_comparison.csv", float_precision="round_trip"
    )
    rows = []
    for model in ["GMM", "intensity_8", "morphospatial_90"]:
        for split, d in pred[pred.ground_truth_binary.isin([0, 1])].groupby("split"):
            r = metrics(d.ground_truth_binary, d[model + "_score"])
            old = ref[(ref.model == model) & (ref.split == split)].iloc[0]
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
                assert np.isclose(r[col], old[col], atol=1e-12, rtol=1e-12), (
                    model,
                    split,
                    col,
                )
            assert np.array_equal(
                (d[model + "_score"] > 0.5).to_numpy(),
                d[model + "_prediction"].to_numpy().astype(bool),
            )
            r.update(model=model, split=split, baseline_match=True)
            rows.append(r)
    pd.DataFrame(rows).to_csv(A / "audit/old_baseline_reproduction.csv", index=False)
    pred.to_parquet(A / "audit/cells.parquet", index=False, compression="zstd")
    # Re-score both saved supervised models on the complete frozen original feature table.
    for model in ["intensity_8", "morphospatial_90"]:
        bundle = joblib.load(OLD / "models" / (model + ".joblib"))
        cols = bundle["features"]
        f = pd.read_parquet(OLD / "PhenoBIC_CD8_features.parquet", columns=KEYS + cols)
        sc = np.empty(len(f))
        for start in range(0, len(f), 100000):
            sc[start : start + 100000] = bundle["estimator"].predict_proba(
                f.iloc[start : start + 100000][cols].to_numpy(float)
            )[:, 1]
        f = f[KEYS]
        f["recomputed"] = sc
        v = pred[KEYS + [model + "_score"]].merge(f, on=KEYS, validate="one_to_one")
        assert len(v) == 1817783 and np.allclose(
            v.recomputed, v[model + "_score"], rtol=0, atol=1e-12
        )
        print(
            "OLD MODEL VERIFIED",
            model,
            float(np.max(abs(v.recomputed - v[model + "_score"]))),
            flush=True,
        )
    selected = []
    for split, context in [
        ("training_WSIs", None),
        ("internal_val_WSIs", None),
        ("external_val_TMA_cores", "TMA1_CD8"),
        ("external_val_TMA_cores", "TMA2_CD8"),
        ("external_val_TMA_cores", "TMA3_CD8"),
    ]:
        d = pred[pred.split == split]
        if context:
            d = d[d.context == context]
        selected.append(d.groupby("image_id").size().idxmin())
    checks = []
    for uid in selected:
        d = pred[pred.image_id == uid].sort_values("cell_id")
        scores, state = fit_score(d.mean_CD8.to_numpy())
        err = float(np.max(abs(scores - d.GMM_score.to_numpy())))
        assert err < 1e-12, (uid, err)
        checks.append(
            dict(
                image_id=uid,
                n_full_unit=len(d),
                max_score_difference=err,
                gate_difference=state["gate_log1p"] - states[uid]["gate_log1p"],
            )
        )
    pd.DataFrame(checks).to_csv(A / "audit/GMM_refit_reproduction.csv", index=False)
    write_json(A / "audit/representative_units.json", selected)
    print("HASH INPUT FILES", flush=True)
    paths = set()
    for e in AUDIT:
        for col in ["image", "mask"]:
            paths.add(DATA / e["source"][col])
        paths.add(OLD / "bboxes" / (e["key"] + ".npz"))
    paths.update(
        [
            OLD / "PhenoBIC_CD8_features.parquet",
            OLD / "PhenoBIC_CD8_model_predictions.parquet",
            OLD / "models/intensity_8.joblib",
            OLD / "models/morphospatial_90.joblib",
        ]
    )

    def fingerprint(p):
        return dict(
            path=str(p),
            bytes=p.stat().st_size,
            mtime_ns=p.stat().st_mtime_ns,
            sha256=sha(p),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        manifest = list(pool.map(fingerprint, sorted(paths)))
    pd.DataFrame(manifest).to_csv(A / "audit/input_manifest.csv", index=False)
    protocol = dict(
        dataset="PhenoBIC CD8 only",
        dataset_commit="aeffd288d922ee5df4cbb414f3c1f77507ba0001",
        pan_gold="sealed; no access",
        key_columns=KEYS,
        expected_all=1817783,
        expected_labelled=expected,
        feature_sets=SETS,
        parameters=PARAMS,
        new_model_fits=16,
        calibration=dict(
            unit="original WSI / pooled internal WSI with unique cells and nonduplicated pixels / each external core",
            eligible="original instance with at least 5 finite pixels; independent of labels and geometry QC",
            L="minimum across eligible per-cell raw p10",
            U="maximum across eligible per-cell raw p90",
            pixels="float64 clip((I-L)/(U-L),0,1); invalid support stays invalid",
            invalid_range="U-L <= 32*float64_eps*max(abs(L),abs(U),float64_tiny)",
            boundary_statistics_population="finite pixels inside eligible original instances; each unique instance/global pixel counted once; strict clipping <L/>U and boundary <=L/>=U both reported",
            bounds_reuse="Existing exact per-cell p10/p90 reused when finite; missing geometry-QC rows re-read from raw pixels independently; normalized 8 features always re-extracted from clipped pixels",
        ),
        geometry=dict(
            mask_type="expanded_nucleus",
            edge="max(1,0.2*max EDT)",
            reference_inner_px=2,
            reference_outer_px=5,
            neighbor_distance_px=2,
            minimum_region_pixels=5,
            minimum_reference_pixels=20,
            epsilon=1e-8,
            high_signal_k=3,
            tissue_mask=False,
            pixel_spacing=None,
            halo=8,
        ),
        thresholds=dict(
            primary="strict score > 0.5",
            supplement="E1-E4 only: select grid .01:.01:.99 maximizing mean of 3 WSI F1 from training OOF; ties closest .5 then higher; frozen before new validation evaluation",
            old_baselines="Keep original >.5 in both comparison panels; never label baseline threshold as OOF-tuned",
            oof_reporting="Selected-threshold OOF metric is threshold development fitting value, not unbiased performance",
        ),
        stability=dict(
            seed=0,
            max_labelled_cells_per_unit=200,
            perturbations=PERTURB,
            affine="a*I+b applied to entire calibration unit including local contexts; float64 no bit-depth clipping; full-unit L/U and GMM re-estimated; affine sufficient statistics permitted only after full-pixel representative-unit verification",
            rotation="np.rot90 image, all instance IDs, valid and optional tissue together; preserve IDs; GMM score reusable only after raw mean invariance check",
            mask="1px Euclidean erosion or dilation into ORIGINAL background only; equidistant competitors retain background; all neighbor IDs perturbed; original L/U and original GMM mapping frozen, scores recomputed from changed-mask raw mean",
            input_valid="finite required observed intensity with >=5 finite pixels; mask present; required image calibration valid; S-only NaN allowed; paired_valid also requires finite outputs before and after",
            both_threshold_modes=True,
        ),
        evaluation="all fixed labelled keys; no NaN/QC filtering; failures retained and full-set metrics marked incomplete rather than subset masquerading as full",
        statistics="paired descriptive per image; no cell-level p-values/bootstrap; cores not assumed independent patients",
        data_manifest_sha256=sha(A / "audit/input_manifest.csv"),
        key_manifest_sha256=sha(A / "keys/key_manifest.csv"),
        sample_sha256=key_hash(sample),
    )
    write_json(A / "E1_E4_PROTOCOL.json", protocol)
    write_json(
        A / "audit/AUDIT_COMPLETE.json",
        dict(
            status="complete",
            all_cells=len(pred),
            sample_cells=len(sample),
            units=len(states),
            baseline_metrics_reproduced=True,
            baseline_supervised_predictions_reproduced=True,
            GMM_score_reconstructed=True,
            source_data_files_hashed=len(manifest),
            data_fingerprint=protocol["data_manifest_sha256"],
            pan_gold_accessed=False,
        ),
    )
    print("AUDIT COMPLETE", len(pred), len(sample), flush=True)


if __name__ == "__main__":
    main()
