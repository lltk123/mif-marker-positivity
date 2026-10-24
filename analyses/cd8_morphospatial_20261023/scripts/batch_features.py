"""PhenoBIC batch extraction with original instance IDs and reconstructed FOV context."""

import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
)
from pathlib import Path
import sys, json, csv, hashlib, time, argparse, importlib, traceback
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import tifffile
from mif_features import *

P = Path("/mnt/disk3/mIHC/mif_marker_positivity_20261022")
G = P / "analyses/cd8_scimap_gmm_20261023"
A = P / "analyses/cd8_morphospatial_20261023"
ROOT = P / "datasets/PhenoBIC-Data"
CFG = FeatureConfig(mask_type="expanded_nucleus")
AUDIT = json.loads((G / "features/phenobic_extraction_audit.json").read_text())
BYKEY = {e["key"]: e for e in AUDIT}
GROUPS = defaultdict(list)
for e in AUDIT:
    GROUPS[(e["split"], e["context"])].append(e)
CACHE = OrderedDict()


def read_image(path):
    path = str(path)
    if path in CACHE:
        CACHE.move_to_end(path)
        return CACHE[path]
    try:
        a = tifffile.memmap(path, mode="r").squeeze()
    except ValueError:
        a = tifffile.imread(path).squeeze()
    if a.ndim != 2:
        raise ValueError("Expected two dimensions")
    CACHE[path] = a
    while len(CACHE) > 6:
        CACHE.popitem(last=False)
    return a


def bbox_index(e):
    dest = A / "bboxes" / (e["key"] + ".npz")
    ids = np.load(G / "features" / (e["key"] + ".npz"))["cell_id"]
    if dest.exists():
        z = np.load(dest)
        assert np.array_equal(ids, z["cell_id"])
        return z["cell_id"], z["bbox"]
    mask = read_image(ROOT / e["source"]["mask"])
    im = read_image(ROOT / e["source"]["image"])
    assert mask.shape == im.shape == tuple(e["shape"]) and np.issubdtype(
        mask.dtype, np.integer
    )
    assert mask.min() >= 0
    # ID lookup uses the documented ndimage label-index mapping, with explicit original IDs from the established index.
    boxes = ndi.find_objects(mask) if len(ids) else []
    bounds = []
    for cid in ids:
        sl = boxes[int(cid) - 1]
        assert sl is not None
        bounds.append([sl[0].start, sl[0].stop, sl[1].start, sl[1].stop])
    b = np.asarray(bounds, dtype=np.int32).reshape(-1, 4)
    np.savez_compressed(dest, cell_id=ids, bbox=b)
    print("BBOX", e["split"], e["context"], e["patch"], len(ids), flush=True)
    return ids, b


def virtual_roi(e, bbox, config=CFG):
    h, w = e["shape"]
    y1, y2, x1, x2 = map(int, bbox)
    halo = int(np.ceil(max(config.r_outer, config.neighbor_distance))) + 2
    gy1 = y1 + e["offset_y"] - halo
    gy2 = y2 + e["offset_y"] + halo
    gx1 = x1 + e["offset_x"] - halo
    gx2 = x2 + e["offset_x"] + halo
    shape = (gy2 - gy1, gx2 - gx1)
    im = np.zeros(shape, dtype=e["image_dtype"])
    mask = np.zeros(shape, dtype=e["mask_dtype"])
    valid = np.zeros(shape, bool)
    candidates = (
        GROUPS[(e["split"], e["context"])] if e["split"] == "internal_val_WSIs" else [e]
    )
    for q in candidates:
        top = max(gy1, q["offset_y"])
        bottom = min(gy2, q["offset_y"] + q["shape"][0])
        left = max(gx1, q["offset_x"])
        right = min(gx2, q["offset_x"] + q["shape"][1])
        if top >= bottom or left >= right:
            continue
        dst = (slice(top - gy1, bottom - gy1), slice(left - gx1, right - gx1))
        src = (
            slice(top - q["offset_y"], bottom - q["offset_y"]),
            slice(left - q["offset_x"], right - q["offset_x"]),
        )
        qi = read_image(ROOT / q["source"]["image"])[src]
        qm = read_image(ROOT / q["source"]["mask"])[src]
        overlap = valid[dst]
        if overlap.any():
            if not np.allclose(
                im[dst][overlap], qi[overlap], rtol=0, atol=0, equal_nan=True
            ):
                raise RuntimeError("Overlapping images disagree")
            existing = mask[dst]
            conflict = (existing > 0) & (qm > 0) & (existing != qm)
            if conflict.any():
                raise RuntimeError("Overlapping instance IDs disagree")
        im[dst] = qi
        mask[dst] = np.where(qm > 0, qm, mask[dst])
        valid[dst] = True
    # A cell abutting the source image boundary is conservatively flagged even where adjacent FOV pixels exist.
    truncated = y1 == 0 or x1 == 0 or y2 == h or x2 == w
    return im, mask, valid, (gy1, gx1), truncated


def metadata(e):
    image_id = e["split"] + "/" + e["context"]
    if e["split"] == "external_val_TMA_cores":
        image_id += "/" + e["patch"]
    return dict(
        dataset_id="PhenoBIC-Data",
        image_id=image_id,
        source_image_id=e["patch"],
        split=e["split"],
        context=e["context"],
        patient_id="",
        batch_id="Opal_WSI_cohort"
        if e["split"] != "external_val_TMA_cores"
        else e["context"],
        cell_id_source="original TIFF integer instance ID",
    )


def extract_batch(job):
    key, start, stop, kind = job
    e = BYKEY[key]
    z = np.load(A / "bboxes" / (key + ".npz"))
    ids = z["cell_id"][start:stop]
    bboxes = z["bbox"][start:stop]
    name = key + f"_{start:07d}_{stop:07d}"
    folder = A / ("pilot" if kind == "pilot" else "features")
    dest = folder / (name + ".parquet")
    note = folder / (name + ".json")
    if dest.exists() and note.exists():
        return json.loads(note.read_text())
    t = time.time()
    rows = []
    for cid, bbox in zip(ids, bboxes):
        im, lab, valid, origin, truncated = virtual_roi(e, bbox)
        r = extract_roi(
            im,
            lab,
            int(cid),
            metadata(e),
            CFG,
            valid_mask=valid,
            origin=origin,
            force_truncated=truncated,
        )
        rows.append(r)
    df = fixed_frame(rows)
    if len(df) and df[["dataset_id", "image_id", "cell_id"]].duplicated().any():
        raise RuntimeError("Duplicate cell keys")
    if np.isinf(df[FEATURES].to_numpy()).any():
        raise RuntimeError("Infinite feature")
    df.to_parquet(dest, index=False, engine="pyarrow", compression="zstd")
    summary = dict(
        key=key,
        split=e["split"],
        context=e["context"],
        patch=e["patch"],
        start=start,
        stop=stop,
        rows=len(df),
        seconds=time.time() - t,
        output=str(dest.relative_to(A)),
        missing={c: int(df[c].isna().sum()) for c in FEATURES},
        qc={c: int(df[c].sum()) for c in QC},
    )
    note.write_text(json.dumps(summary, indent=2))
    return summary


def pilot():
    selected = [
        next(e for e in AUDIT if e["split"] == "training_WSIs"),
        next(
            e
            for e in AUDIT
            if e["split"] == "internal_val_WSIs" and e["n_cells"] > 5000
        ),
    ]
    selected.extend(
        next(e for e in AUDIT if e["context"] == c)
        for c in ["TMA1_CD8", "TMA2_CD8", "TMA3_CD8"]
    )
    results = []
    examples = []
    checks = []
    for e in selected:
        ids, b = bbox_index(e)
        start = max(0, len(ids) // 2 - 100)
        stop = min(len(ids), start + 200)
        result = extract_batch((e["key"], start, stop, "pilot"))
        results.append(result)
        frame = pd.read_parquet(A / result["output"])
        old = np.load(G / "features" / (e["key"] + ".npz"))
        lookup = {int(c): i for i, c in enumerate(old["cell_id"])}
        for row in frame.itertuples():
            i = lookup[int(row.cell_id)]
            assert row.observed_pixel_count == old["pixel_count"][i]
            oldmean = old["intensity_sum"][i] / old["pixel_count"][i]
            assert np.isclose(row.observed_mean_CD8, oldmean, rtol=1e-10, atol=1e-10)
        # Examples are selected by fixed cell ID positions, never by labels or classifier results.
        for pos in [start, start + 60, start + 120]:
            if pos >= stop:
                continue
            cid = int(ids[pos])
            im, lab, valid, origin, truncated = virtual_roi(e, b[pos])
            r, regions = extract_roi(
                im,
                lab,
                cid,
                metadata(e),
                CFG,
                valid_mask=valid,
                origin=origin,
                force_truncated=truncated,
                return_regions=True,
            )
            ex = e["key"] + "_" + str(cid)
            np.savez_compressed(
                A / "examples" / (ex + ".npz"), image=im, mask=lab, **regions
            )
            examples.append(
                dict(
                    example_id=ex,
                    cell_id=cid,
                    image_id=r["image_id"],
                    source_image=e["source"]["image"],
                    source_mask=e["source"]["mask"],
                    origin_y=origin[0],
                    origin_x=origin[1],
                    features=r,
                )
            )
        print(
            "PILOT",
            json.dumps({k: result[k] for k in ["split", "context", "rows", "seconds"]}),
            flush=True,
        )
    (A / "examples/example_index.json").write_text(json.dumps(examples, indent=2))
    status = dict(
        status="passed",
        images=len(results),
        cells=sum(r["rows"] for r in results),
        all_pixel_counts_equal_previous_GMM=True,
        all_observed_means_match_previous_GMM_rtol_atol_1e_10=True,
        results=results,
    )
    (A / "pilot/PILOT_PASSED.json").write_text(json.dumps(status, indent=2))
    print("PILOT PASSED", status["cells"], flush=True)


def full(workers):
    assert (
        (A / "tests/CONTROLLED_TESTS_PASSED.json").exists()
        and (A / "pilot/PILOT_PASSED.json").exists()
        and (A / "pilot/VISUAL_QA_APPROVED.json").exists()
    )
    manifest = dict(
        dataset_id="PhenoBIC-Data",
        dataset_commit="aeffd288d922ee5df4cbb414f3c1f77507ba0001",
        config=asdict(CFG),
        feature_columns=FEATURES,
        metadata_not_model_features=META_STR + COUNTS + META_FLOAT + QC,
        feature_module_sha256=hashlib.sha256(
            (A / "scripts/mif_features.py").read_bytes()
        ).hexdigest(),
        source_index=str(ROOT / "qa/image_mask_label_index.csv"),
        mask_type_evidence="https://pmc.ncbi.nlm.nih.gov/articles/PMC13308180/; Cell segmentation and Figure 2: buffered nuclear convex hull, hence expanded_nucleus for in-house WSI/TMA.",
        tissue_mask_provided=False,
        pixel_spacing_confirmed=False,
        workers=workers,
        pan_gold_policy="Sealed. No reads or use.",
        software={
            m: importlib.import_module(m).__version__
            for m in [
                "numpy",
                "pandas",
                "scipy",
                "skimage",
                "tifffile",
                "pyarrow",
                "sklearn",
            ]
        },
    )
    cfgfile = A / "RUN_CONFIG.json"
    if cfgfile.exists():
        assert json.loads(cfgfile.read_text()) == manifest
    else:
        cfgfile.write_text(json.dumps(manifest, indent=2))
    jobs = []
    for e in AUDIT:
        ids, boxes = bbox_index(e)
        if len(ids) == 0:
            jobs.append((e["key"], 0, 0, "full"))
        else:
            for start in range(0, len(ids), 2500):
                jobs.append((e["key"], start, min(start + 2500, len(ids)), "full"))
    results = []
    total = sum(e["n_cells"] for e in AUDIT)
    done = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract_batch, job) for job in jobs]
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            done += r["rows"]
            progress = dict(
                completed_shards=len(results),
                total_shards=len(jobs),
                cells_done=done,
                total_cells=total,
                elapsed_seconds=time.time() - started,
            )
            (A / "logs/progress.json").write_text(json.dumps(progress))
            print("PROGRESS", json.dumps(progress), flush=True)
    assert done == total == 1817783
    miss = {c: sum(r["missing"][c] for r in results) for c in FEATURES}
    qc = {c: sum(r["qc"][c] for r in results) for c in QC}
    (A / "results/feature_missingness.csv").write_text(
        "feature,missing,total,fraction\n"
        + "".join(f"{c},{n},{done},{n/done}\n" for c, n in miss.items())
    )
    (A / "results/qc_counts.csv").write_text(
        "flag,count,total,fraction\n"
        + "".join(f"{c},{n},{done},{n/done}\n" for c, n in qc.items())
    )
    (A / "features/FEATURES_COMPLETE.json").write_text(
        json.dumps(
            dict(
                status="complete",
                rows=done,
                shards=len(results),
                seconds=time.time() - started,
                features=len(FEATURES),
                morphology_features=len(MORPH),
                missing=miss,
                qc=qc,
            ),
            indent=2,
        )
    )
    print("FEATURES COMPLETE", done, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "full"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "pilot":
        pilot()
    else:
        full(args.workers)
