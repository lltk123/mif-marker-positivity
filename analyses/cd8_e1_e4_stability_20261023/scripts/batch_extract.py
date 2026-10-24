"""Calibrate full image units and extract the frozen E1-E4 feature matrix."""

import json, sys, time, hashlib, platform, importlib, argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from common import *
from io_adapter import AUDIT, BYKEY, boxes, roi, image_id
from compact_features import *

STRINGS = KEYS[:2] + ["source_image_id", "source_key", "split", "context", "mask_type"]
FIELDS = (
    STRINGS
    + ["cell_id"]
    + I8
    + S9
    + COUNTS
    + EXTRA
    + ["intensity_input_valid"]
    + ["qc_" + q for q in QC]
)
SCHEMA = pa.schema(
    [
        (
            c,
            pa.string()
            if c in STRINGS
            else pa.int64()
            if c in COUNTS + ["cell_id"]
            else pa.bool_()
            if c.startswith("qc_") or c == "intensity_input_valid"
            else pa.float64(),
        )
        for c in FIELDS
    ]
)
CALS = None
SIGNATURE = None
REPRESENTATIVES = None


def prepare():
    assert (A / "audit/AUDIT_COMPLETE.json").exists() and (
        A / "tests/PILOT_PASSED.json"
    ).exists()
    columns = KEYS + [
        "source_image_id",
        "split",
        "context",
        "whole_p10",
        "whole_p90",
        "whole_valid_pixel_count",
        "observed_pixel_count",
    ]
    f = pd.read_parquet(OLD / "PhenoBIC_CD8_features.parquet", columns=columns)
    f = f.rename(
        columns={
            "whole_p10": "raw_cell_p10",
            "whole_p90": "raw_cell_p90",
            "whole_valid_pixel_count": "finite_pixel_count",
        }
    )
    missing = f.raw_cell_p10.isna() | f.raw_cell_p90.isna()
    index = {(image_id(e), e["patch"]): e for e in AUDIT}
    repairs = []
    for ix, r in f[missing].iterrows():
        e = index[(r.image_id, r.source_image_id)]
        ids, bb = boxes(e)
        pos = np.flatnonzero(ids == int(r.cell_id))
        assert len(pos) == 1
        im, lab, valid, origin, tr = roi(e, bb[pos[0]])
        v = im[(lab == int(r.cell_id)) & valid & np.isfinite(im)].astype(float)
        assert len(v) == r.finite_pixel_count
        if len(v) >= 5:
            f.loc[ix, ["raw_cell_p10", "raw_cell_p90"]] = np.percentile(v, [10, 90])
        repairs.append(
            dict(
                image_id=r.image_id,
                cell_id=int(r.cell_id),
                finite_pixels=len(v),
                included=len(v) >= 5,
            )
        )
    pd.DataFrame(repairs).to_csv(
        A / "calibration/geometry_QC_bound_recovery.csv", index=False
    )
    f["eligible"] = f.finite_pixel_count >= 5
    params = []
    contributors = []
    top = []
    for uid, g in f.groupby("image_id", sort=True):
        e = g[g.eligible & g.raw_cell_p10.notna() & g.raw_cell_p90.notna()]
        L = float(e.raw_cell_p10.min()) if len(e) else np.nan
        U = float(e.raw_cell_p90.max()) if len(e) else np.nan
        low = e[e.raw_cell_p10 == L]
        high = e[e.raw_cell_p90 == U]
        for name, d in [("L", low), ("U", high)]:
            for cid in d.cell_id:
                contributors.append(dict(image_id=uid, bound=name, cell_id=int(cid)))
        ranked = e.sort_values(["raw_cell_p90", "cell_id"], ascending=[False, True])
        upper99 = float(e.raw_cell_p90.quantile(0.99)) if len(e) else np.nan
        params.append(
            dict(
                image_id=uid,
                split=g.split.iloc[0],
                context=g.context.iloc[0],
                L=L,
                U=U,
                range=U - L,
                calibration_invalid=not calibration_valid(L, U),
                n_instances=len(g),
                n_eligible=len(e),
                n_L_contributors=len(low),
                n_U_contributors=len(high),
                L_first_cell_id=int(low.cell_id.min()) if len(low) else -1,
                U_first_cell_id=int(high.cell_id.min()) if len(high) else -1,
                cell_p90_p99=upper99,
                U_to_cell_p90_p99=U / upper99 if upper99 > 0 else np.nan,
                second_largest_cell_p90=float(ranked.raw_cell_p90.iloc[1])
                if len(ranked) > 1
                else np.nan,
            )
        )
        for rank, (_, row) in enumerate(ranked.head(10).iterrows(), 1):
            top.append(
                dict(
                    image_id=uid,
                    rank=rank,
                    cell_id=int(row.cell_id),
                    raw_cell_p90=float(row.raw_cell_p90),
                )
            )
    f.to_parquet(
        A / "calibration/cell_percentiles.parquet", index=False, compression="zstd"
    )
    pd.DataFrame(contributors).to_parquet(
        A / "calibration/bound_contributors.parquet", index=False
    )
    pd.DataFrame(params).to_csv(A / "calibration/unit_bounds.csv", index=False)
    pd.DataFrame(top).to_csv(A / "calibration/upper_bound_diagnostics.csv", index=False)
    config = dict(
        protocol_sha256=sha(A / "E1_E4_PROTOCOL.json"),
        input_manifest_sha256=sha(A / "audit/input_manifest.csv"),
        bounds_sha256=sha(A / "calibration/unit_bounds.csv"),
        feature_sets=SETS,
        geometry=json.loads((A / "E1_E4_PROTOCOL.json").read_text())["geometry"],
        software={
            k: importlib.import_module(k).__version__
            for k in [
                "numpy",
                "pandas",
                "scipy",
                "skimage",
                "sklearn",
                "pyarrow",
                "tifffile",
            ]
        },
        python=platform.python_version(),
        extraction_code={
            n: sha(A / "scripts" / n)
            for n in [
                "common.py",
                "io_adapter.py",
                "legacy_features.py",
                "compact_features.py",
                "batch_extract.py",
            ]
        },
        raw_pixel_percentile_recoveries=len(repairs),
        all_normalized_statistics_recomputed_from_pixels=True,
    )
    write_json(A / "RUN_CONFIG.json", config)
    print(
        "CALIBRATION PREPARED",
        len(params),
        "missing raw percentile recoveries",
        len(repairs),
        flush=True,
    )


def init_worker():
    global CALS, SIGNATURE, REPRESENTATIVES
    f = pd.read_csv(A / "calibration/unit_bounds.csv", float_precision="round_trip")
    CALS = {r["image_id"]: r for r in f.to_dict("records")}
    SIGNATURE = sha(A / "RUN_CONFIG.json")
    REPRESENTATIVES = set(
        json.loads((A / "audit/representative_units.json").read_text())
    )


def batch(job):
    key, start, stop = job
    e = BYKEY[key]
    uid = image_id(e)
    cal = CALS[uid]
    L = cal["L"]
    U = cal["U"]
    ids, bb = boxes(e)
    ids = ids[start:stop]
    bb = bb[start:stop]
    name = key + f"_{start:07d}_{stop:07d}"
    dest = A / "features" / (name + ".parquet")
    note = A / "features" / (name + ".json")
    if note.exists():
        old = json.loads(note.read_text())
        assert old["signature"] == SIGNATURE and sha(dest) == old["parquet_sha256"]
        return old
    assert not dest.exists(), "Incomplete shard must be reviewed before resume"
    rows = []
    totals = dict(
        finite_pixels=0,
        below_L=0,
        above_U=0,
        at_or_below_L=0,
        at_or_above_U=0,
        raw_min=None,
        raw_max=None,
    )
    affine = {
        k: dict(
            L_min=None,
            U_max=None,
            max_normalized_pixel_difference=0.0,
            max_raw_mean_difference=0.0,
            cells=0,
        )
        for k in ["gain_0.5", "gain_2", "bias_0.05", "bias_0.10"]
    }
    t = time.time()
    for cid, bbox in zip(ids, bb):
        im, lab, valid, origin, tr = roi(e, bbox)
        r = extract_compact(im, lab, int(cid), L, U, valid, tr)
        r.update(
            dataset_id="PhenoBIC-Data",
            image_id=uid,
            cell_id=int(cid),
            source_image_id=e["patch"],
            source_key=key,
            split=e["split"],
            context=e["context"],
            mask_type="expanded_nucleus",
        )
        rows.append(r)
        raw = im[(lab == int(cid)) & valid & np.isfinite(im)].astype(np.float64)
        if len(raw) >= 5:
            totals["finite_pixels"] += len(raw)
            for key2, test in [
                ("below_L", raw < L),
                ("above_U", raw > U),
                ("at_or_below_L", raw <= L),
                ("at_or_above_U", raw >= U),
            ]:
                totals[key2] += int(test.sum())
            totals["raw_min"] = (
                min(totals["raw_min"], float(raw.min()))
                if totals["raw_min"] is not None
                else float(raw.min())
            )
            totals["raw_max"] = (
                max(totals["raw_max"], float(raw.max()))
                if totals["raw_max"] is not None
                else float(raw.max())
            )
            if uid in REPRESENTATIVES and calibration_valid(L, U):
                norm = normalize(im, L, U, valid)
                finite = valid & np.isfinite(im)
                base = im.astype(np.float64)
                for tag, a, b in [
                    ("gain_0.5", 0.5, 0),
                    ("gain_2", 2, 0),
                    ("bias_0.05", 1, 0.05 * (U - L)),
                    ("bias_0.10", 1, 0.10 * (U - L)),
                ]:
                    changed = a * base + b
                    rv = changed[(lab == int(cid)) & finite]
                    q = np.percentile(rv, [10, 90])
                    s = affine[tag]
                    s["L_min"] = (
                        min(s["L_min"], float(q[0]))
                        if s["L_min"] is not None
                        else float(q[0])
                    )
                    s["U_max"] = (
                        max(s["U_max"], float(q[1]))
                        if s["U_max"] is not None
                        else float(q[1])
                    )
                    s["cells"] += 1
                    s["max_normalized_pixel_difference"] = max(
                        s["max_normalized_pixel_difference"],
                        float(
                            np.max(
                                abs(
                                    normalize(changed, a * L + b, a * U + b, valid)[
                                        finite
                                    ]
                                    - norm[finite]
                                )
                            )
                        ),
                    )
                    s["max_raw_mean_difference"] = max(
                        s["max_raw_mean_difference"],
                        float(abs(rv.mean() - (a * raw.mean() + b))),
                    )
    df = pd.DataFrame(rows, columns=FIELDS)
    if len(df):
        assert not np.isinf(df[I8 + S9 + EXTRA].to_numpy(float)).any()
    pq.write_table(
        pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False),
        dest,
        compression="zstd",
    )
    summary = dict(
        source_key=key,
        image_id=uid,
        start=start,
        stop=stop,
        rows=len(df),
        seconds=time.time() - t,
        signature=SIGNATURE,
        parquet_sha256=sha(dest),
        pixel_statistics=totals,
        affine_verification=affine if uid in REPRESENTATIVES else {},
        missing={c: int(df[c].isna().sum()) for c in I8 + S9},
        qc={c: int(df[c].sum()) for c in df if c.startswith("qc_")},
    )
    write_json(note, summary)
    return summary


def full(workers=8):
    assert (A / "RUN_CONFIG.json").exists()
    config = json.loads((A / "RUN_CONFIG.json").read_text())
    for n, h in config["extraction_code"].items():
        assert sha(A / "scripts" / n) == h, "Code changed: " + n
    assert sha(A / "audit/input_manifest.csv") == config["input_manifest_sha256"]
    # Each input was fully hashed at initialization; reject any subsequent size/mtime change before cached reuse.
    for r in pd.read_csv(A / "audit/input_manifest.csv").to_dict("records"):
        p = __import__("pathlib").Path(r["path"])
        assert (
            p.stat().st_size == r["bytes"] and p.stat().st_mtime_ns == r["mtime_ns"]
        ), "Input changed: " + str(p)
    jobs = []
    for e in AUDIT:
        ids, bb = boxes(e)
        for start in range(0, len(ids), 2500):
            jobs.append((e["key"], start, min(start + 2500, len(ids))))
    done = []
    t = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        for future in as_completed([pool.submit(batch, j) for j in jobs]):
            r = future.result()
            done.append(r)
            progress = dict(
                shards=len(done),
                total_shards=len(jobs),
                cells=sum(x["rows"] for x in done),
                total=1817783,
                seconds=time.time() - t,
            )
            (A / "logs/feature_progress.json").write_text(json.dumps(progress))
            print("PROGRESS", json.dumps(progress), flush=True)
    assert sum(d["rows"] for d in done) == 1817783
    bounds = pd.read_csv(
        A / "calibration/unit_bounds.csv", float_precision="round_trip"
    )
    stats = []
    aff = []
    for uid, g in bounds.groupby("image_id", sort=True):
        row = g.iloc[0].to_dict()
        parts = [d for d in done if d["image_id"] == uid]
        ts = [d["pixel_statistics"] for d in parts]
        for key in [
            "finite_pixels",
            "below_L",
            "above_U",
            "at_or_below_L",
            "at_or_above_U",
        ]:
            row[key] = sum(s[key] for s in ts)
        row["raw_min"] = min(s["raw_min"] for s in ts if s["raw_min"] is not None)
        row["raw_max"] = max(s["raw_max"] for s in ts if s["raw_max"] is not None)
        for key in ["below_L", "above_U", "at_or_below_L", "at_or_above_U"]:
            row[key + "_fraction"] = (
                row[key] / row["finite_pixels"] if row["finite_pixels"] else np.nan
            )
        stats.append(row)
        for tag, a, b in [
            ("gain_0.5", 0.5, 0),
            ("gain_2", 2, 0),
            ("bias_0.05", 1, 0.05 * row["range"]),
            ("bias_0.10", 1, 0.10 * row["range"]),
        ]:
            samples = [
                d["affine_verification"][tag]
                for d in parts
                if d["affine_verification"]
                and d["affine_verification"][tag]["cells"] > 0
            ]
            if not samples:
                continue
            lp = min(s["L_min"] for s in samples if s["L_min"] is not None)
            up = max(s["U_max"] for s in samples if s["U_max"] is not None)
            r = dict(
                image_id=uid,
                perturbation=tag,
                cells=sum(s["cells"] for s in samples),
                L_pixel=lp,
                U_pixel=up,
                L_equivalent=a * row["L"] + b,
                U_equivalent=a * row["U"] + b,
                max_normalized_pixel_difference=max(
                    s["max_normalized_pixel_difference"] for s in samples
                ),
                max_raw_mean_difference=max(
                    s["max_raw_mean_difference"] for s in samples
                ),
            )
            assert (
                np.isclose(lp, r["L_equivalent"], rtol=1e-12, atol=1e-12)
                and np.isclose(up, r["U_equivalent"], rtol=1e-12, atol=1e-12)
                and r["max_normalized_pixel_difference"] < 1e-12
            )
            r["passed"] = True
            aff.append(r)
    pd.DataFrame(stats).to_csv(
        A / "calibration/calibration_parameters.csv", index=False
    )
    pd.DataFrame(aff).to_csv(A / "tests/full_unit_affine_equivalence.csv", index=False)
    with pq.ParquetWriter(
        A / "PhenoBIC_E1_E4_features.parquet", SCHEMA, compression="zstd"
    ) as writer:
        for p in sorted((A / "features").glob("*.parquet")):
            for b in pq.ParquetFile(p).iter_batches():
                writer.write_batch(b)
    f = pd.read_parquet(A / "PhenoBIC_E1_E4_features.parquet", columns=KEYS)
    assert len(f) == 1817783 and not f.duplicated().any()
    oldkeys = pd.read_parquet(A / "keys/all_cells.parquet")
    assert key_hash(f) == key_hash(oldkeys)
    write_json(
        A / "features/FEATURES_COMPLETE.json",
        dict(
            status="complete",
            rows=len(f),
            shards=len(done),
            seconds=time.time() - t,
            parquet_sha256=sha(A / "PhenoBIC_E1_E4_features.parquet"),
            unique_key_hash=key_hash(f),
            full_unit_affine_checks=len(aff),
            code_configuration_sha256=sha(A / "RUN_CONFIG.json"),
        ),
    )
    print("FEATURES COMPLETE", len(f), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "full"])
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.mode == "prepare":
        prepare()
    else:
        full(a.workers)
