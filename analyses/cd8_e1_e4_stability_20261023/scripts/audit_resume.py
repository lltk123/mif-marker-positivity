"""Research audit resume for the frozen PhenoBIC CD8 experiment."""

from common import *
from io_adapter import AUDIT
from concurrent.futures import ThreadPoolExecutor


def main():
    pred = pd.read_parquet(A / "audit/cells.parquet")
    sample = pd.read_parquet(A / "keys/stability_sample.parquet")
    states = json.loads((A / "calibration/original_gmm_states.json").read_text())
    expected = {
        "training_WSIs": 101057,
        "internal_val_WSIs": 539132,
        "external_val_TMA_cores": 437480,
    }
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
    protocol[
        "scope_update"
    ] = "User requests no rerun of old experiments. Reuse saved old models/predictions; only new E1-E4 fits and required perturbation GMM estimation. Initial read-only baseline checks had already completed before this update."
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
