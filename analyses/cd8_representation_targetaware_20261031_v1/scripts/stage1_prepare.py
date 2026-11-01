"""Prepare paired E2 representations and native-pixel target-aware crops.
Only the independent experiment directory is writable. Pan-M is never opened.
"""
import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import sys, json, time, hashlib, argparse, traceback
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor, as_completed
from runtime_paths import ROOT, RUN, SOURCE_ROOT

E2 = ROOT / "analyses/cd8_e1_e4_stability_20261023"
MORPH = ROOT / "analyses/cd8_morphospatial_20261023"
from legacy_features import I8, S9
import legacy_compact_features as compact

sys.path.insert(0, str(SOURCE_ROOT / "analyses/cd8_morphospatial_20261023/scripts"))
import legacy_batch_features as reader

sys.path.insert(0, str(RUN / "scripts"))
from repaired_features import describe_regions, NAMES

KEYS = ["dataset_id", "image_id", "cell_id"]
KEEP_S9 = [
    x for x in S9 if x not in ["high_signal_edge_coverage", "edge_signal_enrichment"]
]
D = None
BOUNDS = None
BBOX_CACHE = {}


def savej(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_data():
    old = pd.read_parquet(E2 / "PhenoBIC_E1_E4_features.parquet")
    ann = pd.read_parquet(
        E2 / "audit/cells.parquet",
        columns=KEYS
        + [
            "split",
            "context",
            "ground_truth_original",
            "ground_truth_binary",
            "GMM_score",
        ],
    )
    d = ann.merge(
        old[KEYS + ["source_key", "source_image_id"] + I8 + S9],
        on=KEYS,
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    assert len(d) == 1817783 and d._merge.eq("both").all()
    d = (
        d[d.ground_truth_binary.isin([0, 1])]
        .drop(columns="_merge")
        .reset_index(drop=True)
    )
    assert d.groupby("split").size().to_dict() == {
        "external_val_TMA_cores": 437480,
        "internal_val_WSIs": 539132,
        "training_WSIs": 101057,
    }
    d["row_id"] = np.arange(len(d))
    d["marker"] = "CD8"
    d = d.rename(columns={c: "h0_" + c for c in I8 + S9})
    bounds = pd.read_csv(
        ROOT / "analyses/cd8_robust_calibration_20261030/calibration/unit_bounds.csv",
        float_precision="round_trip",
    ).set_index("image_id")
    assert len(bounds) == 48 and not bounds.calibration_invalid.any()
    return d, bounds


def one_cell(row, old_reference=False):
    e = reader.BYKEY[row.source_key]
    bp = MORPH / "bboxes" / (row.source_key + ".npz")
    if row.source_key not in BBOX_CACHE:
        with np.load(bp) as z:
            ids = z["cell_id"]
            boxes = z["bbox"]
        order = np.argsort(ids)
        BBOX_CACHE[row.source_key] = (ids[order], boxes[order])
        if len(BBOX_CACHE) > 8:
            BBOX_CACHE.pop(next(k for k in BBOX_CACHE if k != row.source_key))
    ids, boxes = BBOX_CACHE[row.source_key]
    pos = np.searchsorted(ids, int(row.cell_id))
    if pos >= len(ids) or ids[pos] != int(row.cell_id):
        raise ValueError("Original bbox ID mismatch")
    bbox = boxes[pos]
    im, lab, valid, origin, trunc = reader.virtual_roi(e, bbox)
    b = BOUNDS.loc[row.image_id]
    L, U = float(b.L), float(b.U)
    result = compact.extract_compact(
        im, lab, int(row.cell_id), L, U, valid, force_truncated=trunc
    )
    if old_reference:
        op = OLD_BOUNDS.loc[row.image_id]
        prev = compact.extract_compact(
            im,
            lab,
            int(row.cell_id),
            float(op.L),
            float(op.U),
            valid,
            force_truncated=trunc,
        )
        for c in I8 + S9:
            a = float(prev[c])
            old = float(getattr(row, "h0_" + c))
            if not (
                (np.isnan(a) and np.isnan(old))
                or np.isclose(a, old, rtol=1e-10, atol=1e-12)
            ):
                raise ValueError(
                    "Legacy pixel reproduction mismatch "
                    + str((row.image_id, int(row.cell_id), c, a, old))
                )
    x = compact.normalize(im, L, U, valid)
    M = lab == int(row.cell_id)
    fin = valid & np.isfinite(x)
    distance = ndi.distance_transform_edt(np.pad(M, 1))[1:-1, 1:-1]
    edge = M & (distance <= max(1.0, 0.2 * distance.max()))
    inner = M & ~edge
    outside = ndi.distance_transform_edt(~M)
    ref = (outside > 2) & (outside <= 5) & (lab == 0) & fin
    rv, states = describe_regions(
        {
            "whole": x[M & fin],
            "edge": x[edge & fin],
            "inner": x[inner & fin],
            "reference": x[ref],
        },
        geometry_valid=bool(result["intensity_input_valid"]),
    )
    rec = {"row_id": int(row.row_id)}
    rec.update({"h1_" + c: result[c] for c in I8 + S9})
    rec.update(rv)
    rec.update({k: result[k] for k in result if k.startswith("qc_")})
    for region, state in states.items():
        if isinstance(state, dict):
            for k, v in state.items():
                rec["repair_" + region + "_" + k] = v
    rec["repair_geometry_valid"] = states["geometry_valid"]
    rec["L"] = L
    rec["U"] = U
    yy, xx = np.nonzero(M)
    cy = int(np.rint(yy.mean() + origin[0] - e["offset_y"]))
    cx = int(np.rint(xx.mean() + origin[1] - e["offset_x"]))
    # virtual_roi adds the legacy halo of 7 px; this pseudo-box gives exactly 96px.
    cropbox = (cy - 41, cy + 41, cx - 41, cx + 41)
    cim, clab, cv, co, ct = reader.virtual_roi(e, cropbox)
    assert cim.shape == (96, 96)
    target = (clab == int(row.cell_id)) & cv
    other = (clab > 0) & ~target & cv
    cn = compact.normalize(cim, L, U, cv)
    network = np.where(np.isfinite(cn), cn, 0).astype(np.float32)
    if not np.isfinite(network).all():
        raise ValueError("Nonfinite crop")
    rec["crop_target_fraction"] = float(target.sum() / M.sum())
    rec["crop_context_valid_fraction"] = float(cv.mean())
    rec["crop_all_zero"] = bool(not np.any(network > 0))
    rec["crop_target_pixels"] = int(target.sum())
    return rec, network, np.packbits(target.reshape(-1)), np.packbits(other.reshape(-1))


def job(args):
    chunk_id, indices = args
    stem = "chunk_%05d" % chunk_id
    out = RUN / "data/chunks" / (stem + ".parquet")
    done = RUN / "data/chunks" / (stem + ".json")
    if done.exists():
        meta = json.loads(done.read_text())
        assert (
            meta["n"] == len(indices)
            and meta["row_ids_sha256"]
            == hashlib.sha256(np.asarray(indices, np.int64).tobytes()).hexdigest()
        )
        return meta
    selected = D.iloc[indices]
    n = len(selected)
    start = time.time()
    records = []
    image = np.lib.format.open_memmap(
        RUN / "data/crops" / (stem + "_image.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(n, 96, 96),
    )
    target = np.lib.format.open_memmap(
        RUN / "data/crops" / (stem + "_target.npy"),
        mode="w+",
        dtype=np.uint8,
        shape=(n, 1152),
    )
    other = np.lib.format.open_memmap(
        RUN / "data/crops" / (stem + "_other.npy"),
        mode="w+",
        dtype=np.uint8,
        shape=(n, 1152),
    )
    for k, row in enumerate(selected.itertuples(index=False)):
        rec, a, b, c = one_cell(row)
        records.append(rec)
        image[k] = a
        target[k] = b
        other[k] = c
    image.flush()
    target.flush()
    other.flush()
    del image, target, other
    pd.DataFrame(records).to_parquet(out, index=False)
    meta = {
        "chunk_id": chunk_id,
        "source_key": str(selected.source_key.iloc[0]),
        "n": n,
        "seconds": time.time() - start,
        "row_ids_sha256": hashlib.sha256(
            np.asarray(indices, np.int64).tobytes()
        ).hexdigest(),
        "feature_sha256": sha(out),
    }
    savej(done, meta)
    print("CHUNK", json.dumps(meta), flush=True)
    return meta


def main():
    global D, BOUNDS, OLD_BOUNDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pilot", "full"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    D, BOUNDS = load_data()
    ob = pd.read_csv(E2 / "calibration/unit_bounds.csv", float_precision="round_trip")
    OLD_BOUNDS = ob.set_index("image_id")
    for d in ["data/chunks", "data/crops", "provenance", "logs"]:
        (RUN / d).mkdir(parents=True, exist_ok=True)
    if args.stage == "pilot":
        records = []
        start = time.time()
        dev = D[D.split != "external_val_TMA_cores"]
        for uid, g in dev.groupby("image_id", sort=True):
            sample = g.sort_values("cell_id").head(20)
            for row in sample.itertuples(index=False):
                rec, im, ta, ot = one_cell(row, old_reference=True)
                records.append(
                    {
                        "image_id": uid,
                        "cell_id": int(row.cell_id),
                        "legacy_all17_equal": True,
                        "crop96": list(im.shape),
                        "target_present": int(np.unpackbits(ta).sum()) > 0,
                        "repaired_features_finite": int(
                            sum(np.isfinite(rec[c]) for c in NAMES)
                        ),
                    }
                )
        pd.DataFrame(records).to_csv(RUN / "results/real_pilot_checks.csv", index=False)
        savej(
            RUN / "results/PILOT_COMPLETE.json",
            {
                "n": len(records),
                "seconds": time.time() - start,
                "all_legacy_equal": True,
                "all_targets_present": all(x["target_present"] for x in records),
                "no_external_images_used": True,
            },
        )
        print(
            json.dumps(
                {
                    "pilot_n": len(records),
                    "seconds": time.time() - start,
                    "all_checks_passed": True,
                }
            ),
            flush=True,
        )
        return
    assert (RUN / "results/PILOT_COMPLETE.json").exists()
    metadata_file = RUN / "data/cells_base.parquet"
    if not metadata_file.exists():
        D.to_parquet(metadata_file, index=False)
    tasks = []
    chunks = np.empty(len(D), np.int32)
    offset = np.empty(len(D), np.int32)
    cid = 0
    for key, g in D.groupby("source_key", sort=True):
        ix = g.index.to_numpy()
        for st in range(0, len(ix), 2500):
            v = ix[st : st + 2500]
            tasks.append((cid, v))
            chunks[v] = cid
            offset[v] = np.arange(len(v))
            cid += 1
    np.save(RUN / "data/cache_chunk.npy", chunks)
    np.save(RUN / "data/cache_offset.npy", offset)
    start = time.time()
    summ = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, t) for t in tasks]
        for fut in as_completed(futures):
            summ.append(fut.result())
    features = pd.concat(
        [
            pd.read_parquet(RUN / "data/chunks" / ("chunk_%05d.parquet" % c))
            for c in range(cid)
        ],
        ignore_index=True,
    ).sort_values("row_id")
    assert (
        np.array_equal(features.row_id.to_numpy(), np.arange(len(D)))
        and not features.row_id.duplicated().any()
    )
    full = D.merge(
        features, on="row_id", validate="one_to_one", how="left"
    ).sort_values("row_id")
    # Reuse the already frozen raw-image risk definitions and audit measurements.
    cols = KEYS + [
        "audit_raw_whole_mean",
        "audit_raw_neighbor_edge_area_fraction",
        "audit_raw_neighbor_edge_mean_diff",
        "audit_raw_local_reference_median",
        "audit_raw_sector_signal_cv",
        "audit_raw_edge_inner_mean_diff",
    ]
    raw = pd.read_parquet(
        ROOT / "analyses/cd8_tabm_robust_20261030/data/cells.parquet", columns=cols
    )
    full = full.merge(raw, on=KEYS, validate="one_to_one", how="left").sort_values(
        "row_id"
    )
    full.to_parquet(RUN / "data/features.parquet", index=False)
    pd.DataFrame(summ).to_csv(RUN / "results/extraction_resources.csv", index=False)
    savej(
        RUN / "data/FEATURES_COMPLETE.json",
        {
            "rows": len(full),
            "n_chunks": cid,
            "seconds": time.time() - start,
            "feature_sha256": sha(RUN / "data/features.parquet"),
            "protocol_sha256": sha(RUN / "PROTOCOL.json"),
            "input_old_features_sha256": sha(E2 / "PhenoBIC_E1_E4_features.parquet"),
            "input_old_cells_sha256": sha(E2 / "audit/cells.parquet"),
            "input_new_bounds_sha256": sha(
                ROOT
                / "analyses/cd8_robust_calibration_20261030/calibration/unit_bounds.csv"
            ),
            "feature_sets": {
                "H0": ["h0_" + c for c in I8 + S9],
                "H1": ["h1_" + c for c in I8 + S9],
                "H2": ["h1_" + c for c in I8 + KEEP_S9] + list(NAMES),
            },
            "mask": "expanded_nucleus",
            "crop_dtype": "float32; packed binary masks; no intensity quantization",
            "PanM_accessed": False,
        },
    )
    print("PREPARATION_COMPLETE", len(full), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        savej(
            RUN / "logs/PREPARATION_FAILED.json",
            {"traceback": traceback.format_exc(), "time": time.time()},
        )
        raise
