"""Apply the prespecified affine, rotation and mask perturbations to frozen models."""

import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, time, joblib, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from common import *
from io_adapter import BYKEY, boxes, roi
from compact_features import extract_compact, perturb_mask
from legacy_features import extract_roi, FeatureConfig, FEATURES, WHOLE
from gmm_interface import fit_score, score_fixed

MODELS = ["GMM", "intensity_8", "morphospatial_90"] + list(SETS)
CFG = FeatureConfig(mask_type="expanded_nucleus")


def extract_pair(im, lab, cid, L, U, valid, trunc):
    c = extract_compact(im, lab, cid, L, U, valid, force_truncated=trunc)
    if (lab == cid).any():
        raw = extract_roi(
            im, lab, cid, config=CFG, valid_mask=valid, force_truncated=trunc
        )
    else:
        raw = {k: np.nan for k in FEATURES}
        raw.update(whole_valid_pixel_count=0, observed_mean_CD8=np.nan)
    return c, raw


def score_frames(c, raw, gs):
    out = {"GMM": gs}
    valid = {
        "GMM": (c.finite_pixel_count.to_numpy() >= 5)
        & np.isfinite(c.observed_raw_mean)
        & (c.observed_raw_mean >= 0)
    }
    rawvalid = np.isfinite(raw[WHOLE].to_numpy(float)).all(axis=1) & (
        raw.whole_valid_pixel_count.to_numpy() >= 5
    )
    for name in ["intensity_8", "morphospatial_90"]:
        b = BUNDLES[name]
        out[name] = b["estimator"].predict_proba(raw[b["features"]].to_numpy(float))[
            :, 1
        ]
        valid[name] = rawvalid
    f = c.copy()
    f["GMM_score"] = gs
    for name in SETS:
        b = BUNDLES[name]
        out[name] = b["estimator"].predict_proba(f[b["features"]].to_numpy(float))[:, 1]
        valid[name] = c.intensity_input_valid.to_numpy(bool) & (
            np.isfinite(gs) if name in ["E3", "E4"] else True
        )
    return out, valid


def unit_job(uid):
    code = digest(uid)[:16]
    path = A / "stability/shards" / (code + ".parquet")
    meta = path.with_suffix(".json")
    if path.exists():
        m = json.loads(meta.read_text())
        assert m["signature"] == SIGNATURE and sha(path) == m["sha256"]
        return uid, len(pd.read_parquet(path))
    samp = SAMPLE[SAMPLE.image_id == uid].sort_values("cell_id").reset_index(drop=True)
    allunit = CELLS[CELLS.image_id == uid].sort_values("cell_id")
    bounds = BOUNDS[uid]
    L = float(bounds["L"])
    U = float(bounds["U"])
    oldstate = STATES[uid]
    assert len(allunit) == oldstate["n_fit"]
    fullmeans = allunit.mean_CD8.to_numpy(float)
    states = {}
    cals = []
    for perturb, a, b in [
        ("gain_0.5", 0.5, 0),
        ("gain_2", 2, 0),
        ("bias_0.05", 1, 0.05 * (U - L)),
        ("bias_0.10", 1, 0.10 * (U - L)),
    ]:
        _, st = fit_score(a * fullmeans + b)
        states[perturb] = st
        cals.append(
            dict(
                image_id=uid,
                perturbation=perturb,
                L=a * L + b,
                U=a * U + b,
                n_full_GMM=len(fullmeans),
                n_calibration_eligible=bounds["n_eligible"],
                gate_log1p=st["gate_log1p"],
                L_U_reestimated=True,
                GMM_action="full_unit_refit",
                method="positive affine transform of full-unit sufficient statistics, verified by 20 full-pixel representative-unit checks",
            )
        )
    write_json(
        A / "stability/gmm_states" / (code + ".json"),
        dict(image_id=uid, signature=SIGNATURE, states=states),
    )
    contexts = []
    idx = {}
    basec = []
    baser = []
    normal_checks = []
    for row in samp.itertuples():
        e = BYKEY[row.source_key]
        if row.source_key not in idx:
            ids, bb = boxes(e)
            idx[row.source_key] = {int(cid): box for cid, box in zip(ids, bb)}
        im, lab, v, origin, trunc = roi(
            e, idx[row.source_key][int(row.cell_id)], halo=8
        )
        c, r = extract_pair(im, lab, int(row.cell_id), L, U, v, trunc)
        basec.append(c)
        baser.append(r)
        contexts.append((im, lab, v, trunc, int(row.cell_id)))
    basec = pd.DataFrame(basec)
    baser = pd.DataFrame(baser)
    g0 = score_fixed(basec.observed_raw_mean.to_numpy(), oldstate)
    base_scores, basevalid = score_frames(basec, baser, g0)
    originals = allunit.set_index("cell_id").loc[samp.cell_id]
    for name in MODELS:
        expected = originals[name + "_score"].to_numpy(float)
        diff = np.abs(base_scores[name] - expected)
        err = float(np.nanmax(diff))
        assert err < 1e-10, (uid, name, err)
        normal_checks.append(
            dict(
                image_id=uid,
                model=name,
                n=len(samp),
                max_original_score_reproduction_error=err,
            )
        )
        # Stored original score is the exact baseline, after verifying local reconstruction.
        base_scores[name] = expected
    rows = []
    invariant = []
    featureqc = []
    for perturb in PERTURB:
        cs = []
        rs = []
        for im, lab, v, trunc, cid in contexts:
            if perturb.startswith("gain"):
                a = float(perturb.split("_")[1])
                pi = im.astype(np.float64) * a
                pm = lab
                pv = v
                pL = a * L
                pU = a * U
            elif perturb.startswith("bias"):
                b = float(perturb.split("_")[1]) * (U - L)
                pi = im.astype(np.float64) + b
                pm = lab
                pv = v
                pL = L + b
                pU = U + b
            elif perturb.startswith("rotate"):
                k = int(perturb.split("_")[1]) // 90
                pi = np.rot90(im, k)
                pm = np.rot90(lab, k)
                pv = np.rot90(v, k)
                pL = L
                pU = U
            else:
                pi = im
                pm = perturb_mask(lab, perturb, v)
                pv = v
                pL = L
                pU = U
            c, r = extract_pair(pi, pm, cid, pL, pU, pv, trunc)
            cs.append(c)
            rs.append(r)
        c = pd.DataFrame(cs)
        r = pd.DataFrame(rs)
        if perturb.startswith("rotate"):
            md = float(np.max(abs(c.observed_raw_mean - basec.observed_raw_mean)))
            assert np.allclose(
                c.observed_raw_mean,
                basec.observed_raw_mean,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
            gs = g0.copy()
            action = "reuse_verified_permutation"
            lure = False
        elif perturb in states:
            gs = score_fixed(c.observed_raw_mean.to_numpy(), states[perturb])
            action = "full_unit_refit"
            lure = True
            md = np.nan
        else:
            gs = score_fixed(c.observed_raw_mean.to_numpy(), oldstate)
            action = "fixed_original_mapping_new_intensity"
            lure = False
            md = np.nan
        scores, valids = score_frames(c, r, gs)
        for feat in I8 + S9:
            x = basec[feat].to_numpy(float)
            z = c[feat].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(z)
            diff = abs(x[ok] - z[ok])
            invariant.append(
                dict(
                    image_id=uid,
                    perturbation=perturb,
                    feature=feat,
                    n=len(samp),
                    n_paired_finite=int(ok.sum()),
                    NaN_pattern_changes=int((np.isfinite(x) != np.isfinite(z)).sum()),
                    absolute_change_median=float(np.median(diff))
                    if len(diff)
                    else np.nan,
                    absolute_change_p95=float(np.percentile(diff, 95))
                    if len(diff)
                    else np.nan,
                    absolute_change_max=float(diff.max()) if len(diff) else np.nan,
                    raw_mean_rotation_error=md,
                )
            )
        for j, row in samp.iterrows():
            q = dict(
                dataset_id=row.dataset_id,
                image_id=uid,
                cell_id=int(row.cell_id),
                split=row.split,
                context=row.context,
                perturbation=perturb,
                ground_truth_binary=row.ground_truth_binary,
                observed_pixels=int(c.iloc[j].observed_pixel_count),
                finite_pixels=int(c.iloc[j].finite_pixel_count),
                mask_missing=bool(c.iloc[j].qc_mask_missing),
                intensity_input_valid=bool(c.iloc[j].intensity_input_valid),
                qc_reasons=";".join(
                    k for k in c if k.startswith("qc_") and bool(c.iloc[j][k])
                ),
            )
            featureqc.append(q)
        for name in MODELS:
            before = base_scores[name]
            after = scores[name]
            vb = np.asarray(basevalid[name], bool)
            va = np.asarray(valids[name], bool)
            fb = np.isfinite(before)
            fa = np.isfinite(after)
            paired = vb & va & fb & fa
            for j, row in samp.iterrows():
                rr = dict(
                    dataset_id=row.dataset_id,
                    image_id=uid,
                    cell_id=int(row.cell_id),
                    split=row.split,
                    context=row.context,
                    ground_truth_binary=int(row.ground_truth_binary),
                    model=name,
                    perturbation=perturb,
                    score_before=float(before[j]),
                    score_after=float(after[j]),
                    input_valid_before=bool(vb[j]),
                    input_valid=bool(va[j]),
                    score_finite_before=bool(fb[j]),
                    score_finite=bool(fa[j]),
                    paired_valid=bool(paired[j]),
                    L_U_reestimated=lure,
                    GMM_action=action if name in ["GMM", "E3", "E4"] else "not_used",
                    experiment_family="affine_end_to_end"
                    if perturb in states
                    else "rotation"
                    if perturb.startswith("rotate")
                    else "mask_fixed_calibration",
                )
                for mode, t in [
                    ("fixed_0.5", 0.5),
                    ("OOF_threshold", THRESH.get(name, 0.5)),
                ]:
                    rr["threshold_" + mode] = t
                    rr["prediction_before_" + mode] = (
                        int(before[j] > t) if vb[j] and fb[j] else -1
                    )
                    rr["prediction_after_" + mode] = (
                        int(after[j] > t) if va[j] and fa[j] else -1
                    )
                rows.append(rr)
        print("UNIT", uid, perturb, flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(path, index=False, compression="zstd")
    pd.DataFrame(invariant).to_csv(
        A / "stability/checks" / (code + "_invariance.csv"), index=False
    )
    pd.DataFrame(normal_checks).to_csv(
        A / "stability/checks" / (code + "_baseline.csv"), index=False
    )
    pd.DataFrame(cals).to_csv(
        A / "stability/checks" / (code + "_calibration.csv"), index=False
    )
    pd.DataFrame(featureqc).to_parquet(
        A / "stability/checks" / (code + "_feature_QC.parquet"),
        index=False,
        compression="zstd",
    )
    write_json(
        meta, dict(signature=SIGNATURE, sha256=sha(path), rows=len(out), image_id=uid)
    )
    return uid, len(out)


def summarize(d):
    rows = []
    for level, grouping in [
        ("unit", ["model", "perturbation", "split", "context", "image_id"]),
        ("split", ["model", "perturbation", "split"]),
        ("context", ["model", "perturbation", "split", "context"]),
    ]:
        for key, g in d.groupby(grouping):
            base = dict(zip(grouping, key))
            paired = g[g.paired_valid]
            n = len(g)
            npair = len(paired)
            delta = abs(paired.score_after - paired.score_before)
            for mode in ["fixed_0.5", "OOF_threshold"]:
                r = dict(
                    base,
                    level=level,
                    threshold_mode=mode,
                    threshold=float(g["threshold_" + mode].iloc[0]),
                    threshold_source=mode
                    if base["model"] in SETS
                    else "original_baseline",
                    total_pairs=n,
                    valid_pairs=npair,
                    unscoreable=n - npair,
                    coverage=npair / n,
                    score_change_median=float(delta.median()),
                    score_change_p95=float(delta.quantile(0.95)),
                    L_U_reestimated=bool(g.L_U_reestimated.iloc[0]),
                    GMM_action=g.GMM_action.iloc[0],
                    experiment_family=g.experiment_family.iloc[0],
                )
                t = r["threshold"]
                pb = paired.score_before > t
                pa = paired.score_after > t
                pn = int((pb & ~pa).sum())
                np_ = int((~pb & pa).sum())
                flips = pn + np_
                r.update(
                    positive_to_negative=pn,
                    negative_to_positive=np_,
                    positive_to_negative_rate=pn / npair if npair else np.nan,
                    negative_to_positive_rate=np_ / npair if npair else np.nan,
                    flip_count=flips,
                    flip_rate=flips / npair if npair else np.nan,
                    flip_or_unscoreable_rate=(flips + n - npair) / n,
                    positive_fraction_before=float(pb.mean()),
                    positive_fraction_after=float(pa.mean()),
                    positive_fraction_change=float(pa.mean() - pb.mean()),
                    input_invalid_before=int((~g.input_valid_before).sum()),
                    input_invalid_after=int((~g.input_valid).sum()),
                    nonfinite_score_after=int((~g.score_finite).sum()),
                )
                for which in ["before", "after"]:
                    m = metrics(paired.ground_truth_binary, paired["score_" + which], t)
                    for c in ["precision", "recall", "F1", "TP", "FP", "FN", "TN"]:
                        r[c + "_" + which] = m[c]
                rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(A / "stability/stability_metrics.csv", index=False)
    for lev in ["unit", "split", "context"]:
        out[out.level == lev].to_csv(
            A / "stability" / ("stability_" + lev + ".csv"), index=False
        )


def main(workers):
    global CELLS, SAMPLE, BOUNDS, STATES, BUNDLES, THRESH, SIGNATURE
    for folder in [
        "stability",
        "stability/shards",
        "stability/checks",
        "stability/gmm_states",
    ]:
        (A / folder).mkdir(exist_ok=True)
    assert (A / "results/TRAINING_COMPLETE.json").exists()
    inputs = (
        [
            A / "E1_E4_PROTOCOL.json",
            A / "RUN_CONFIG.json",
            A / "THRESHOLDS_FROZEN.json",
            A / "keys/stability_sample.parquet",
            A / "calibration/original_gmm_states.json",
            A / "calibration/calibration_parameters.csv",
            A / "audit/input_manifest.csv",
            A / "predictions/final_all_cells_fixed_0.5.parquet",
        ]
        + [
            A / "scripts" / s
            for s in [
                "stability.py",
                "common.py",
                "compact_features.py",
                "legacy_features.py",
                "gmm_interface.py",
                "io_adapter.py",
            ]
        ]
        + [A / "models" / (n + ".joblib") for n in SETS]
        + [
            OLD / "models" / (n + ".joblib")
            for n in ["intensity_8", "morphospatial_90"]
        ]
    )
    signature_data = {str(p): sha(p) for p in inputs}
    SIGNATURE = digest(signature_data)
    write_json(
        A / "stability/STABILITY_RUN.json",
        dict(
            signature=SIGNATURE,
            inputs=signature_data,
            sample_size=9600,
            full_unit_GMM_affine_refits=192,
            perturbations=PERTURB,
            halo=8,
            models=MODELS,
        ),
    )
    CELLS = pd.read_parquet(
        A / "audit/cells.parquet",
        columns=KEYS
        + ["mean_CD8", "GMM_score", "intensity_8_score", "morphospatial_90_score"],
    )
    pred = pd.read_parquet(
        A / "predictions/final_all_cells_fixed_0.5.parquet",
        columns=KEYS + [n + "_score" for n in SETS],
    )
    CELLS = CELLS.merge(pred, on=KEYS, validate="one_to_one")
    del pred
    SAMPLE = pd.read_parquet(A / "keys/stability_sample.parquet")
    sources = pd.read_parquet(
        A / "PhenoBIC_E1_E4_features.parquet", columns=KEYS + ["source_key"]
    )
    SAMPLE = SAMPLE.merge(sources, on=KEYS, validate="one_to_one")
    del sources
    assert len(SAMPLE) == 9600 and SAMPLE.image_id.nunique() == 48
    BOUNDS = (
        pd.read_csv(
            A / "calibration/calibration_parameters.csv", float_precision="round_trip"
        )
        .set_index("image_id")
        .to_dict("index")
    )
    STATES = json.loads((A / "calibration/original_gmm_states.json").read_text())
    THRESH = json.loads((A / "THRESHOLDS_FROZEN.json").read_text())["thresholds"]
    BUNDLES = {
        n: joblib.load((A if n in SETS else OLD) / "models" / (n + ".joblib"))
        for n in MODELS
        if n != "GMM"
    }
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {
            pool.submit(unit_job, uid): uid for uid in sorted(SAMPLE.image_id.unique())
        }
        for i, future in enumerate(as_completed(jobs)):
            uid, n = future.result()
            print("COMPLETED UNIT", i + 1, 48, uid, n, flush=True)
    d = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((A / "stability/shards").glob("*.parquet"))
        ],
        ignore_index=True,
    )
    assert (
        len(d) == 9600 * 9 * 7
        and not d[KEYS + ["model", "perturbation"]].duplicated().any()
    )
    assert not np.isinf(d[["score_before", "score_after"]].to_numpy(float)).any()
    d.to_parquet(
        A / "stability/paired_cell_results.parquet", index=False, compression="zstd"
    )
    summarize(d)
    for suffix, name in [
        ("invariance", "feature_invariance"),
        ("baseline", "original_score_reproduction"),
        ("calibration", "perturbed_calibration"),
    ]:
        pd.concat(
            [
                pd.read_csv(p)
                for p in sorted((A / "stability/checks").glob("*_" + suffix + ".csv"))
            ],
            ignore_index=True,
        ).to_csv(A / "stability" / (name + ".csv"), index=False)
    qc = pd.concat(
        [
            pd.read_parquet(p)
            for p in sorted((A / "stability/checks").glob("*_feature_QC.parquet"))
        ],
        ignore_index=True,
    )
    qc.to_parquet(
        A / "stability/perturbed_feature_QC.parquet", index=False, compression="zstd"
    )
    write_json(
        A / "stability/STABILITY_COMPLETE.json",
        dict(
            status="complete",
            signature=SIGNATURE,
            paired_cell_model_rows=len(d),
            sample_cells=9600,
            units=48,
            perturbations=9,
            models=7,
            seconds=time.time() - start,
        ),
    )
    print("STABILITY COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    main(a.workers)
