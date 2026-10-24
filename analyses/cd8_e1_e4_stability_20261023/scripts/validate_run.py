"""Audit full experiment outputs, model coverage and stability results."""

import json, time, joblib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from common import *


def main():
    checks = []

    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), detail=str(detail)))
        assert passed, (name, detail)

    manifest = pd.read_csv(A / "audit/input_manifest.csv")
    changed = []

    def filecheck(r):
        p = Path(r.path)
        return (
            str(p),
            p.stat().st_size == r.bytes and p.stat().st_mtime_ns == r.mtime_ns,
            sha(p) == r.sha256,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(filecheck, manifest.itertuples()))
    check("original_inputs_sha256", all(r[2] for r in results), len(results))
    check("original_inputs_size_mtime", all(r[1] for r in results), len(results))
    fm = json.loads((A / "features/FEATURES_COMPLETE.json").read_text())
    check(
        "full_feature_sha256",
        sha(A / "PhenoBIC_E1_E4_features.parquet") == fm["parquet_sha256"],
        fm["rows"],
    )
    for name, n in [("E1", 8), ("E2", 17), ("E3", 9), ("E4", 18)]:
        b = joblib.load(A / "models" / (name + ".joblib"))
        check(
            name + "_features",
            b["features"] == SETS[name] and len(b["features"]) == n,
            len(b["features"]),
        )
        check(
            name + "_params",
            all(b["estimator"].get_params()[k] == v for k, v in PARAMS.items()),
            PARAMS,
        )
    fit = pd.read_csv(A / "results/new_model_fit_log.csv")
    check(
        "new_model_fit_count",
        len(fit) == 16 and (fit.fit_kind == "OOF").sum() == 12,
        len(fit),
    )
    for mode in ["fixed_0.5", "OOF_threshold"]:
        d = pd.read_parquet(
            A / "predictions" / ("final_all_cells_" + mode + ".parquet")
        )
        check(
            "all_prediction_keys_" + mode,
            len(d) == 1817783 and not d[KEYS].duplicated().any(),
            len(d),
        )
        check(
            "all_prediction_key_hash_" + mode,
            key_hash(d) == fm["unique_key_hash"],
            key_hash(d),
        )
        check(
            "all_scores_finite_" + mode,
            np.isfinite(
                d[
                    [
                        n + "_score"
                        for n in ["GMM", "intensity_8", "morphospatial_90"] + list(SETS)
                    ]
                ].to_numpy()
            ).all(),
            len(d),
        )
        for n in ["GMM", "intensity_8", "morphospatial_90"] + list(SETS):
            check(
                "strict_threshold_" + n + "_" + mode,
                np.array_equal(
                    (d[n + "_score"] > d[n + "_threshold"]).astype("int8"),
                    d[n + "_prediction"],
                ),
                len(d),
            )
        o = pd.read_parquet(A / "predictions" / ("training_OOF_" + mode + ".parquet"))
        check(
            "OOF_keys_" + mode,
            len(o) == 101057 and not o[KEYS].duplicated().any(),
            len(o),
        )
    coverage = []
    for key, g in d.groupby(["split", "context", "image_id"]):
        for n in ["GMM", "intensity_8", "morphospatial_90"] + list(SETS):
            coverage.append(
                dict(
                    split=key[0],
                    context=key[1],
                    image_id=key[2],
                    model=n,
                    n_all=len(g),
                    n_labelled=int(g.ground_truth_binary.isin([0, 1]).sum()),
                    n_finite_score=int(np.isfinite(g[n + "_score"]).sum()),
                    n_calibration_invalid=int(g.qc_calibration_invalid.sum()),
                    n_intensity_input_invalid=int((~g.intensity_input_valid).sum())
                    if n != "GMM"
                    else 0,
                    n_any_spatial_QC=int(
                        g[
                            [
                                c
                                for c in g
                                if c.startswith("qc_")
                                and c
                                not in [
                                    "qc_reference_tissue_unknown",
                                    "qc_truncated_old",
                                    "qc_disconnected_old",
                                ]
                            ]
                        ]
                        .any(axis=1)
                        .sum()
                    ),
                )
            )
    pd.DataFrame(coverage).to_csv(
        A / "results/prediction_input_coverage.csv", index=False
    )
    s = pd.read_parquet(A / "stability/paired_cell_results.parquet")
    check(
        "stability_rows",
        len(s) == 604800 and not s[KEYS + ["model", "perturbation"]].duplicated().any(),
        len(s),
    )
    check(
        "paired_valid_definition",
        np.array_equal(
            s.paired_valid,
            s.input_valid_before
            & s.input_valid
            & s.score_finite_before
            & s.score_finite,
        ),
        len(s),
    )
    st = pd.read_csv(A / "stability/stability_unit.csv")
    check("unit_stability_rows", len(st) == 6048, len(st))
    check(
        "total_pairs_per_unit", (st.total_pairs == 200).all(), st.total_pairs.unique()
    )
    inv = pd.read_csv(A / "stability/feature_invariance.csv")
    rot = inv[inv.perturbation.str.startswith("rotate")]
    check(
        "rotation_R1_R2_offset",
        rot[
            rot.feature.isin(
                ["angular_R1", "angular_R2", "signal_centroid_offset_normalized"]
            )
        ].absolute_change_max.max()
        < 1e-10,
        rot[
            rot.feature.isin(
                ["angular_R1", "angular_R2", "signal_centroid_offset_normalized"]
            )
        ].absolute_change_max.max(),
    )
    aff = pd.read_csv(A / "tests/full_unit_affine_equivalence.csv")
    check(
        "full_unit_affine_verification",
        len(aff) == 20 and aff.passed.all(),
        aff.max_normalized_pixel_difference.max(),
    )
    rec = pd.read_csv(A / "stability/original_score_reproduction.csv")
    check(
        "sample_original_score_reproduction",
        len(rec) == 48 * 7 and rec.max_original_score_reproduction_error.max() < 1e-10,
        rec.max_original_score_reproduction_error.max(),
    )
    for p in (A / "curves").glob("*.npz"):
        z = np.load(p)
        check(
            "curve_arrays_" + p.stem,
            len(z["ROC_fpr"]) == len(z["ROC_threshold"])
            and len(z["PR_precision"]) == len(z["PR_threshold"]) + 1,
            len(z["ROC_threshold"]),
        )
    pd.DataFrame(checks).to_csv(A / "results/FINAL_VALIDATION_CHECKS.csv", index=False)
    write_json(
        A / "results/FINAL_VALIDATION.json",
        dict(
            status="passed",
            checks=len(checks),
            original_files_rehashed=len(results),
            pan_gold_accessed=False,
            script_sha256=sha(__file__),
        ),
    )
    print("VALIDATION PASSED", len(checks), flush=True)
    print(
        st[
            (st.model.isin(["E1", "E2", "E3", "E4"]))
            & (st.threshold_mode == "fixed_0.5")
        ]
        .groupby(["model", "perturbation"])[["flip_count", "unscoreable"]]
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    main()
