"""Composition sensitivity: change estimator pools, never target or context pixels."""
import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json, hashlib, time, joblib
import numpy as np, pandas as pd
from experiment_common import *
from stage1_hgb import transform
import stage1_prepare as prep


def rank_rows(g, salt):
    keys = [
        int.from_bytes(
            hashlib.blake2b(
                (salt + "|" + str(z.image_id) + "|" + str(int(z.cell_id))).encode(),
                digest_size=8,
            ).digest(),
            "little",
        )
        for z in g.itertuples()
    ]
    return g.iloc[np.argsort(np.asarray(keys, np.uint64))]


def raw_target(row):
    e = prep.reader.BYKEY[row.source_key]
    if row.source_key not in prep.BBOX_CACHE:
        with np.load(prep.MORPH / "bboxes" / (row.source_key + ".npz")) as z:
            ids = z["cell_id"]
            boxes = z["bbox"]
        order = np.argsort(ids)
        prep.BBOX_CACHE[row.source_key] = (ids[order], boxes[order])
    ids, boxes = prep.BBOX_CACHE[row.source_key]
    ix = np.searchsorted(ids, int(row.cell_id))
    assert ids[ix] == int(row.cell_id)
    im, lab, valid, origin, trunc = prep.reader.virtual_roi(e, boxes[ix])
    pix = np.ascontiguousarray(im[(lab == int(row.cell_id)) & valid])
    return hashlib.sha256(pix.tobytes()).hexdigest()


def main():
    assert (RUN / "models/HGB_COMPLETE.json").exists()
    d, man = load()
    dev = d[
        (d.split != TEST) & d.h0_norm_whole_mean.notna() & d.h1_norm_whole_mean.notna()
    ]
    targets = []
    for uid, g in dev.groupby("image_id", sort=True):
        for label in [0, 1]:
            targets.append(
                rank_rows(g[g.ground_truth_binary == label], "composition_v1").head(25)
            )
    targets = pd.concat(targets).sort_values("row_id")
    assert len(targets) == 300
    c = pd.read_parquet(prep.E2 / "calibration/cell_percentiles.parquet")
    c = c[(c.finite_pixel_count >= 5) & c.raw_cell_p10.notna() & c.raw_cell_p90.notna()]
    assert not c.duplicated(["image_id", "cell_id"]).any()
    base = pd.read_csv(
        ROOT / "analyses/cd8_robust_calibration_20261030/calibration/unit_bounds.csv",
        float_precision="round_trip",
    ).set_index("image_id")
    bundles = {
        name: joblib.load(RUN / "models" / (name + ".joblib"))
        for name in ["H0", "H1", "H2"]
    }
    records = []
    started = time.time()
    for row in targets.itertuples(index=False):
        full = c[c.image_id == row.image_id]
        target = full[full.cell_id == int(row.cell_id)]
        assert len(target) == 1
        others = full[full.cell_id != int(row.cell_id)].sort_values(
            ["raw_cell_p90", "cell_id"]
        )
        n70 = int(np.ceil(0.7 * len(others)))
        q99 = others.raw_cell_p90.quantile(0.99)
        pools = {
            "all": others,
            "drop_top_1pct": others[others.raw_cell_p90 <= q99],
            "lowest_70pct": others.iloc[:n70],
            "highest_70pct": others.iloc[-n70:],
        }
        original_hash = raw_target(row)
        for pool_name, other in pools.items():
            cells = pd.concat([other, target], ignore_index=True)
            for method, names in [
                ("legacy_minmax", ["H0"]),
                ("outerP1P99", ["H1", "H2"]),
            ]:
                L = (
                    float(cells.raw_cell_p10.min())
                    if method == "legacy_minmax"
                    else float(np.percentile(cells.raw_cell_p10, 1))
                )
                U = (
                    float(cells.raw_cell_p90.max())
                    if method == "legacy_minmax"
                    else float(np.percentile(cells.raw_cell_p90, 99))
                )
                prep.BOUNDS = base.copy()
                prep.BOUNDS.loc[row.image_id, ["L", "U"]] = [L, U]
                features, network, ta, ot = prep.one_cell(row)
                assert raw_target(row) == original_hash
                for name in names:
                    b = bundles[name]
                    values = [
                        features[col.replace("h0_", "h1_", 1)]
                        if col.startswith("h0_")
                        else features[col]
                        for col in b["features"]
                    ]
                    score = float(
                        b["estimator"].predict_proba(
                            transform(
                                np.asarray(values, float)[None, :], b["preprocessing"]
                            )
                        )[0, 1]
                    )
                    records.append(
                        {
                            "row_id": int(row.row_id),
                            "image_id": row.image_id,
                            "split": row.split,
                            "cell_id": int(row.cell_id),
                            "label": int(row.ground_truth_binary),
                            "model": name,
                            "pool": pool_name,
                            "normalization": method,
                            "n_pool": len(cells),
                            "L": L,
                            "U": U,
                            "score": score,
                            "prediction_fixed_0_5": score > 0.5,
                            "target_raw_pixels_sha256": original_hash,
                        }
                    )
    result = pd.DataFrame(records)
    ref = result[result.pool == "all"][
        ["row_id", "model", "score", "prediction_fixed_0_5"]
    ].rename(
        columns={
            "score": "reference_score",
            "prediction_fixed_0_5": "reference_prediction",
        }
    )
    result = result.merge(ref, on=["row_id", "model"], validate="many_to_one")
    result["score_delta"] = result.score - result.reference_score
    result["prediction_changed"] = (
        result.prediction_fixed_0_5 != result.reference_prediction
    )
    result.to_csv(RUN / "results/COMPOSITION_DIAGNOSTIC.csv", index=False)
    checks = []
    for name in ["H0", "H1", "H2"]:
        prev = np.load(RUN / "predictions" / (name + "_development.npy"))
        g = result[(result.model == name) & (result.pool == "all")]
        err = float(np.max(abs(g.score.to_numpy() - prev[g.row_id.to_numpy()])))
        checks.append(
            {
                "model": name,
                "all_pool_score_max_abs_difference": err,
                "passed": err <= 1e-10,
            }
        )
    pd.DataFrame(checks).to_csv(
        RUN / "results/COMPOSITION_REFERENCE_CHECKS.csv", index=False
    )
    summary = (
        result.groupby(["model", "normalization", "pool"])
        .agg(
            n_targets=("row_id", "size"),
            mean_abs_score_delta=("score_delta", lambda x: float(np.mean(abs(x)))),
            max_abs_score_delta=("score_delta", lambda x: float(np.max(abs(x)))),
            prediction_change_fraction=("prediction_changed", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(RUN / "results/COMPOSITION_SUMMARY.csv", index=False)
    savej(
        RUN / "results/COMPOSITION_COMPLETE.json",
        {
            "targets": 300,
            "rows": len(result),
            "seconds": time.time() - started,
            "all_reference_checks_passed": all(x["passed"] for x in checks),
            "all_target_pixels_unchanged": True,
            "external_images_or_labels_used": False,
            "interpretation": "diagnostic only; no retraining; not independent generalization performance",
            "code_sha256": sha(__file__),
        },
    )
    if not all(x["passed"] for x in checks):
        raise ValueError("Composition baseline does not reproduce frozen model inputs")
    print(
        "COMPOSITION_COMPLETE",
        json.dumps(
            {"targets": 300, "rows": len(result), "seconds": time.time() - started}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
