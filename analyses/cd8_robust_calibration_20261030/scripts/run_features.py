"""Prepare full-unit bounds, pilot, extract and merge robust 90-feature data.



All caches bind raw inputs, old feature tables, code and configuration hashes.

Old files are read-only. Existing incompatible artifacts cause a hard failure.

"""

import os


os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
)

import argparse

from concurrent.futures import ProcessPoolExecutor, as_completed

from dataclasses import asdict

import importlib

import json

from pathlib import Path

import platform

import time

import numpy as np

import pandas as pd

import pyarrow as pa

import pyarrow.parquet as pq

from calibration_core import *

from feature_adapter import *


DEFAULT_PROJECT = Path("/mnt/disk3/mIHC/mif_marker_positivity_20261022")

STATE = {}


def csv_write(path, frame):
    payload = frame.to_csv(index=False).replace("\r\n", "\n")

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text() != payload:
            raise RuntimeError("Incompatible existing CSV: " + str(path))
    else:
        path.write_text(payload, encoding="utf-8")


def parquet_write(path, frame, metadata=None):
    path = Path(path)

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(frame, preserve_index=False)

    if metadata:
        combined = dict(table.schema.metadata or {})

        combined.update({k.encode(): str(v).encode() for k, v in metadata.items()})

        table = table.replace_schema_metadata(combined)
    temp = path.with_suffix(".parquet.tmp")

    pq.write_table(table, temp, compression="zstd")

    temp.replace(path)


def paths(project):
    return (
        project / "analyses/cd8_morphospatial_20261023",
        project / "analyses/cd8_e1_e4_stability_20261023",
        project / "analyses/cd8_scimap_gmm_20261023",
    )


def code_hashes(run):
    return {
        str(p.relative_to(run)): sha256(p)
        for p in sorted((run / "scripts").glob("*.py"))
    }


def prepare(args):
    run, project = args.run_dir, args.project_root
    completed = run / "calibration/calibration_manifest.json"
    if completed.exists():
        frozen = verify(run)
        previous = json.loads(completed.read_text())
        if (
            previous["protocol_sha256"] != sha256(run / "PROTOCOL.json")
            or previous["unit_bounds_sha256"]
            != sha256(run / "calibration/unit_bounds.csv")
            or previous["status"] != "complete"
            or frozen["morphology_cache"] != (not args.recompute_morphology)
            or frozen["chunk_size"] != args.chunk_size
        ):
            raise RuntimeError("Incompatible completed calibration")
        print("CALIBRATION_REUSED", previous["rows"], previous["n_units"], flush=True)
        return
    old, e1, gmm = paths(project)

    source = old / "PhenoBIC_CD8_features.parquet"

    percentiles = e1 / "calibration/cell_percentiles.parquet"

    input_manifest = e1 / "audit/input_manifest.csv"

    # The normalized output has no dependency on true labels or model predictions.

    legacy = load_legacy(old)

    config = json.loads((old / "RUN_CONFIG.json").read_text())["config"]

    assert config["epsilon"] == 1e-8 and config["mask_type"] == "expanded_nucleus"

    source_sha, pct_sha = sha256(source), sha256(percentiles)

    old_inputs = pd.read_csv(input_manifest)

    expected_source = old_inputs.loc[old_inputs.path == str(source), "sha256"]

    if len(expected_source) != 1 or expected_source.iloc[0] != source_sha:
        raise RuntimeError("Original feature file differs from audited E1-E4 source")
    original = pd.read_parquet(
        source,
        columns=KEYS3
        + [
            "source_image_id",
            "split",
            "context",
            "observed_pixel_count",
            "whole_valid_pixel_count",
        ],
    )

    raw = pd.read_parquet(percentiles)

    check = original.merge(
        raw,
        on=KEYS3,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_original", ""),
    )

    if not (check._merge == "both").all() or len(check) != EXPECTED_ROWS:
        raise RuntimeError("Original and percentile keys or counts differ")
    for col in ["source_image_id", "split", "context"]:
        if not check[col].equals(check[col + "_original"]):
            raise RuntimeError("Percentile metadata mismatch: " + col)
    if not np.array_equal(check.finite_pixel_count, check.whole_valid_pixel_count):
        raise RuntimeError("Raw finite-pixel support mismatch")
    if set(raw.dataset_id) != {"PhenoBIC-Data"}:
        raise ValueError("Unexpected dataset")
    raw["marker"] = "CD8"

    original["marker"] = "CD8"

    digest = key_hash(raw)

    if digest != key_hash(original):
        raise RuntimeError("Cell-key hash mismatch")
    bounds, witnesses = fit_unit_bounds(raw)

    if len(bounds) != EXPECTED_UNITS:
        raise RuntimeError("Expected 48 complete original calibration units")
    audit_path = gmm / "features/phenobic_extraction_audit.json"

    audit = json.loads(audit_path.read_text())

    raw_paths = sorted(
        {
            str(project / "datasets/PhenoBIC-Data" / e["source"][kind])
            for e in audit
            for kind in ["image", "mask"]
        }
    )

    inputs = []

    for filename in raw_paths:
        p = Path(filename)

        st = p.stat()

        previous = old_inputs.loc[old_inputs.path == filename]

        actual = sha256(p)

        if (
            len(previous) != 1
            or actual != previous.sha256.iloc[0]
            or st.st_size != int(previous.bytes.iloc[0])
        ):
            raise RuntimeError(
                "Raw image/mask differs from audited source: " + filename
            )
        inputs.append(
            dict(
                path=filename, bytes=st.st_size, mtime_ns=st.st_mtime_ns, sha256=actual
            )
        )
    for e in audit:
        bbox = old / "bboxes" / (e["key"] + ".npz")

        with np.load(bbox) as z:
            ids, bb = z["cell_id"], z["bbox"]

            if len(np.unique(ids)) != len(ids) or len(ids) != len(bb):
                raise RuntimeError("Invalid explicit bbox ID index: " + str(bbox))
        inputs.append(
            dict(
                path=str(bbox),
                bytes=bbox.stat().st_size,
                mtime_ns=bbox.stat().st_mtime_ns,
                sha256=sha256(bbox),
            )
        )
    for p in [
        source,
        percentiles,
        audit_path,
        old / "RUN_CONFIG.json",
        old / "scripts/mif_features.py",
        old / "scripts/batch_features.py",
        old / "FEATURE_DICTIONARY.csv",
    ]:
        inputs.append(
            dict(
                path=str(p),
                bytes=p.stat().st_size,
                mtime_ns=p.stat().st_mtime_ns,
                sha256=sha256(p),
            )
        )
    csv_write(run / "calibration/input_manifest.csv", pd.DataFrame(inputs))

    csv_write(run / "calibration/unit_bounds.csv", bounds)

    csv_write(run / "calibration/bound_interpolation_witnesses.csv", witnesses)

    keys_path = run / "calibration/all_cell_keys.parquet"

    if not keys_path.exists():
        parquet_write(
            keys_path,
            original[KEYS + ["source_image_id", "split", "context"]].sort_values(KEYS),
        )
    elif key_hash(pd.read_parquet(keys_path, columns=KEYS)) != digest:
        raise RuntimeError("Existing key manifest differs")
    for split, group in original.groupby("split"):
        p = run / "calibration" / ("keys_" + split + ".parquet")

        if not p.exists():
            parquet_write(p, group[KEYS].sort_values(KEYS))
    parent_protocol = (
        project / "analyses/cd8_robust_comparison_20261030/RETRAIN_PROTOCOL.json"
    )

    protocol = dict(
        method_id=METHOD_ID,
        dataset="PhenoBIC-Data CD8 only",
        rows=EXPECTED_ROWS,
        n_units=EXPECTED_UNITS,
        keys=KEYS,
        all_keys_sha256=digest,
        mask_type="expanded_nucleus",
        calibration=dict(
            inner_percentiles=[10, 90],
            outer_percentiles=[1, 99],
            min_finite_instance_pixels=5,
            method="linear",
            instance_weighting="equal",
            labels_used=False,
            tail_instances_removed=False,
            mapping="clip((I-L)/(U-L),0,1)",
            dtype="float64",
            invalid_support="NaN",
            units="original WSI; deduplicated internal WSI; single external core",
        ),
        geometry=config,
        feature_columns=legacy.FEATURES,
        chunk_size=args.chunk_size,
        morphology_cache=not args.recompute_morphology,
        morphology_source_sha256=source_sha,
        morphology_note="Explicit 3-key lookup; private function-globals substitution only; geometry QC recomputed; reference path --recompute-morphology",
        roi="Unmodified legacy virtual_roi, halo=7, exact overlaps, original owner-FOV force_truncated",
        clipping_denominator="Finite pixels of each unique original instance exactly once, including geometry-QC instances; excludes background and duplicated FOV views",
        source_feature_sha256=source_sha,
        cell_percentiles_source_sha256=pct_sha,
        source_input_manifest_sha256=sha256(input_manifest),
        input_manifest_sha256=sha256(run / "calibration/input_manifest.csv"),
        unit_bounds_sha256=sha256(run / "calibration/unit_bounds.csv"),
        code_sha256=code_hashes(run),
        parent_protocol_sha256=sha256(parent_protocol)
        if parent_protocol.exists()
        else None,
        software=dict(
            python=platform.python_version(),
            **{
                x: importlib.import_module(x).__version__
                for x in ["numpy", "pandas", "scipy", "skimage", "pyarrow", "tifffile"]
            },
        ),
    )

    json_write(run / "PROTOCOL.json", protocol)

    dictionary = pd.read_csv(old / "FEATURE_DICTIONARY.csv")

    dictionary["original_unit"] = dictionary["unit"]

    dictionary["unit"] = dictionary["unit"].replace(
        {
            "original intensity unit": "normalized intensity [0,1]",
            "intensity*px": "normalized intensity * px",
        }
    )

    dictionary["normalization_method"] = METHOD_ID

    dictionary["input_pixel_scale"] = "unit-common clipped float [0,1]"

    dictionary["definition_note"] = dictionary.name.map(
        lambda n: "Original geometry unchanged; pixel/radian units retained"
        if n in legacy.MORPH
        else "Legacy formula re-evaluated on normalized pixels; epsilon=1e-8; intensity-unit factors now normalized units"
    )

    csv_write(run / "FEATURE_DICTIONARY.csv", dictionary)

    json_write(
        run / "calibration/calibration_manifest.json",
        dict(
            status="complete",
            method_id=METHOD_ID,
            protocol_sha256=sha256(run / "PROTOCOL.json"),
            unit_bounds_file="calibration/unit_bounds.csv",
            unit_bounds_sha256=sha256(run / "calibration/unit_bounds.csv"),
            all_keys_sha256=digest,
            rows=EXPECTED_ROWS,
            n_units=EXPECTED_UNITS,
            cell_percentiles_source_sha256=pct_sha,
            source_feature_sha256=source_sha,
            source_input_manifest_sha256=sha256(input_manifest),
        ),
    )

    print("CALIBRATION_COMPLETE", len(raw), len(bounds), flush=True)


def verify(run):
    protocol = json.loads((run / "PROTOCOL.json").read_text())

    if code_hashes(run) != protocol["code_sha256"]:
        raise RuntimeError("Code changed after preparation; use a new run directory")
    if sha256(run / "calibration/unit_bounds.csv") != protocol["unit_bounds_sha256"]:
        raise RuntimeError("Calibration bounds changed")
    manifest = pd.read_csv(run / "calibration/input_manifest.csv")

    if (
        sha256(run / "calibration/input_manifest.csv")
        != protocol["input_manifest_sha256"]
    ):
        raise RuntimeError("Input manifest changed")
    for row in manifest.itertuples(index=False):
        p = Path(row.path)

        if (
            p.stat().st_size != row.bytes
            or p.stat().st_mtime_ns != row.mtime_ns
            or sha256(p) != row.sha256
        ):
            raise RuntimeError("Input content changed: " + str(p))
    return protocol


def initialize(run_string, project_string):
    run, project = Path(run_string), Path(project_string)

    old, e1, gmm = paths(project)

    legacy = load_legacy(old)

    reader = load_reader(old, legacy)

    protocol = json.loads((run / "PROTOCOL.json").read_text())

    columns = (
        KEYS3
        + legacy.MORPH
        + ["observed_mean_CD8", "observed_pixel_count", "whole_valid_pixel_count"]
    )

    source = pd.read_parquet(old / "PhenoBIC_CD8_features.parquet", columns=columns)

    STATE.update(
        run=run,
        project=project,
        old=old,
        legacy=legacy,
        reader=reader,
        protocol=protocol,
        config=legacy.FeatureConfig(**protocol["geometry"]),
        cache=MorphologyCache(source, legacy.MORPH, protocol["source_feature_sha256"]),
        original=source.set_index(KEYS3).sort_index(),
        bounds=pd.read_csv(
            run / "calibration/unit_bounds.csv", float_precision="round_trip"
        ).set_index("image_id"),
    )


def extract_one(e, cid, bbox, recompute=False):
    reader, legacy = STATE["reader"], STATE["legacy"]

    metadata = reader.metadata(e)

    b = STATE["bounds"].loc[metadata["image_id"]]

    im, lab, valid, origin, truncated = reader.virtual_roi(e, bbox, STATE["config"])

    result = extract_normalized_roi(
        legacy,
        im,
        lab,
        int(cid),
        metadata,
        STATE["config"],
        b.L,
        b.U,
        valid_mask=valid,
        origin=origin,
        force_truncated=truncated,
        morphology_cache=None
        if recompute or not STATE["protocol"]["morphology_cache"]
        else STATE["cache"],
    )

    original = STATE["original"].loc[
        (metadata["dataset_id"], metadata["image_id"], int(cid))
    ]

    if (
        result["observed_pixel_count"] != original.observed_pixel_count
        or result["raw_finite_pixel_count"] != original.whole_valid_pixel_count
    ):
        raise RuntimeError("Original finite pixel or mask support changed")
    if not np.isclose(
        result["raw_observed_mean_CD8"],
        original.observed_mean_CD8,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError("Original mean differs: ROI reconstruction changed")
    if recompute or not STATE["protocol"]["morphology_cache"]:
        actual = np.array([result[c] for c in legacy.MORPH], float)

        expected = original[legacy.MORPH].to_numpy(float)

        if not np.array_equal(actual, expected, equal_nan=True):
            raise RuntimeError("Recomputed morphology differs from frozen original")
    result["source_key"] = e["key"]

    return result


def pilot(args):
    verify(args.run_dir)

    initialize(str(args.run_dir), str(args.project_root))

    reader, legacy = STATE["reader"], STATE["legacy"]

    buckets = [("training_WSIs", None), ("internal_val_WSIs", None)] + [
        ("external_val_TMA_cores", "TMA%d_CD8" % k) for k in [1, 2, 3]
    ]

    rows, audits = [], []

    old, e1, _ = paths(args.project_root)

    raw = pd.read_parquet(e1 / "calibration/cell_percentiles.parquet").set_index(KEYS3)

    for split, context in buckets:
        entries = sorted(
            [
                e
                for e in reader.AUDIT
                if e["split"] == split and (context is None or e["context"] == context)
            ],
            key=lambda e: e["key"],
        )

        # Empty sources remain audited but cannot supply a cell pilot.
        e = None
        for candidate in entries:
            with np.load(old / "bboxes" / (candidate["key"] + ".npz")) as index:
                if len(index["cell_id"]):
                    e = candidate
                    break
        if e is None:
            raise RuntimeError(
                "No nonempty source for required pilot group: " + str((split, context))
            )
        z = np.load(old / "bboxes" / (e["key"] + ".npz"))

        positions = np.unique(np.linspace(0, len(z["cell_id"]) - 1, 20).astype(int))

        for ix in positions:
            cid, bbox = int(z["cell_id"][ix]), z["bbox"][ix]

            cached = extract_one(e, cid, bbox)

            reference = extract_one(e, cid, bbox, recompute=True)

            a = np.array([cached[c] for c in legacy.FEATURES], float)

            b = np.array([reference[c] for c in legacy.FEATURES], float)

            if not np.array_equal(a, b, equal_nan=True):
                raise RuntimeError(
                    "Morphology cache/reference full-feature disagreement"
                )
            im, lab, valid, origin, tr = reader.virtual_roi(e, bbox, STATE["config"])

            values = im[(lab == cid) & valid & np.isfinite(im)].astype(float)

            record = raw.loc[(cached["dataset_id"], cached["image_id"], cid)]

            q = (
                np.percentile(values, [10, 90])
                if len(values) >= 5
                else [np.nan, np.nan]
            )

            if len(values) >= 5 and not np.allclose(
                q, [record.raw_cell_p10, record.raw_cell_p90], rtol=0, atol=0
            ):
                raise RuntimeError(
                    "Recovered raw percentiles disagree with actual original pixels"
                )
            rows.append(cached)

            audits.append(
                dict(
                    image_id=cached["image_id"],
                    cell_id=cid,
                    source_key=e["key"],
                    all_90_cache_reference_equal=True,
                    original_morphology_exact=True,
                    raw_percentiles_exact=True,
                    raw_mean_exact_support=True,
                    normalization_scope="actual full-unit bounds",
                    finite_pixels=len(values),
                )
            )
    output = args.run_dir / "pilot/features.parquet"

    if not output.exists():
        parquet_write(output, rows_frame(legacy, rows))
    csv_write(args.run_dir / "pilot/validation.csv", pd.DataFrame(audits))

    json_write(
        args.run_dir / "pilot/PASSED.json",
        dict(
            status="passed",
            rows=len(rows),
            source_groups=5,
            protocol_sha256=sha256(args.run_dir / "PROTOCOL.json"),
            validation_sha256=sha256(args.run_dir / "pilot/validation.csv"),
        ),
    )

    print("PILOT_PASSED", len(rows), flush=True)


def shard(job):
    key, start, stop = job

    run, reader, legacy = STATE["run"], STATE["reader"], STATE["legacy"]

    name = key + "_%07d_%07d" % (start, stop)

    output = run / "shards" / (name + ".parquet")

    note = run / "shards" / (name + ".json")

    signature = sha256(run / "PROTOCOL.json")

    if output.exists() or note.exists():
        if not (output.exists() and note.exists()):
            raise RuntimeError(
                "Incomplete shard requires explicit investigation: " + name
            )
        info = json.loads(note.read_text())

        if info["protocol_sha256"] != signature or info["sha256"] != sha256(output):
            raise RuntimeError("Incompatible cached shard: " + name)
        return info
    e = reader.BYKEY[key]

    z = np.load(STATE["old"] / "bboxes" / (key + ".npz"))

    t = time.monotonic()

    rows = [
        extract_one(e, int(cid), bbox)
        for cid, bbox in zip(z["cell_id"][start:stop], z["bbox"][start:stop])
    ]

    frame = rows_frame(legacy, rows)

    if (
        frame.duplicated(KEYS).any()
        or np.isinf(frame[legacy.FEATURES].to_numpy()).any()
    ):
        raise RuntimeError("Invalid shard keys or infinite features")
    parquet_write(output, frame)

    info = dict(
        name=name,
        rows=len(rows),
        protocol_sha256=signature,
        sha256=sha256(output),
        seconds=time.monotonic() - t,
    )

    json_write(note, info)

    return info


def jobs_for(project, chunk_size):
    old, _, gmm = paths(project)

    audit = json.loads((gmm / "features/phenobic_extraction_audit.json").read_text())

    jobs = []

    for e in audit:
        with np.load(old / "bboxes" / (e["key"] + ".npz")) as z:
            n = len(z["cell_id"])
        jobs.extend(
            (e["key"], a, min(a + chunk_size, n)) for a in range(0, n, chunk_size)
        )
    return jobs


def extract(args):
    protocol = verify(args.run_dir)

    passed = json.loads((args.run_dir / "pilot/PASSED.json").read_text())

    if passed["protocol_sha256"] != sha256(args.run_dir / "PROTOCOL.json"):
        raise RuntimeError("Pilot signature mismatch")
    if args.chunk_size != protocol["chunk_size"]:
        raise RuntimeError("Chunk size differs from frozen protocol")
    jobs = jobs_for(args.project_root, args.chunk_size)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize,
        initargs=(str(args.run_dir), str(args.project_root)),
    ) as pool:
        futures = {pool.submit(shard, j): j for j in jobs}

        for i, future in enumerate(as_completed(futures), 1):
            info = future.result()

            print(
                "SHARD",
                i,
                len(jobs),
                info["name"],
                info["rows"],
                round(info["seconds"], 2),
                flush=True,
            )


def merge(args):
    protocol = verify(args.run_dir)

    old, _, _ = paths(args.project_root)

    legacy = load_legacy(old)

    output = args.run_dir / "PhenoBIC_CD8_features.parquet"

    manifest = args.run_dir / "extraction_manifest.json"

    if manifest.exists():
        saved = json.loads(manifest.read_text())

        if saved["feature_sha256"] != sha256(output) or saved[
            "protocol_sha256"
        ] != sha256(args.run_dir / "PROTOCOL.json"):
            raise RuntimeError("Completed output hash mismatch")
        return
    if output.exists():
        raise RuntimeError(
            "Output exists without complete manifest; inspect it before recovery"
        )
    frames = []

    signature = sha256(args.run_dir / "PROTOCOL.json")

    if args.chunk_size != protocol["chunk_size"]:
        raise RuntimeError("Chunk size differs from frozen protocol")
    for key, start, stop in jobs_for(args.project_root, args.chunk_size):
        name = key + "_%07d_%07d" % (start, stop)

        p = args.run_dir / "shards" / (name + ".parquet")

        info = json.loads(p.with_suffix(".json").read_text())

        if info["protocol_sha256"] != signature or info["sha256"] != sha256(p):
            raise RuntimeError("Shard signature/hash mismatch")
        frames.append(pd.read_parquet(p))
    frame = pd.concat(frames, ignore_index=True)

    if len(frame) != EXPECTED_ROWS or key_hash(frame) != protocol["all_keys_sha256"]:
        raise RuntimeError("Final rows or explicit cell-key hash differs")
    if np.isinf(frame[legacy.FEATURES].to_numpy()).any():
        raise RuntimeError("Infinite feature values")
    original = pd.read_parquet(
        old / "PhenoBIC_CD8_features.parquet", columns=KEYS3 + legacy.MORPH
    ).set_index(KEYS3)

    matched = original.reindex(pd.MultiIndex.from_frame(frame[KEYS3]))

    if not np.array_equal(
        frame[legacy.MORPH].to_numpy(), matched[legacy.MORPH].to_numpy(), equal_nan=True
    ):
        raise RuntimeError("Original morphology changed")
    audits = []

    for uid, group in frame.groupby("image_id", sort=True):
        n = int(group.raw_finite_pixel_count.sum())

        audits.append(
            dict(
                image_id=uid,
                split=group.split.iloc[0],
                context=group.context.iloc[0],
                rows=len(group),
                finite_instance_pixels=n,
                raw_min=group.raw_observed_min_CD8.min(),
                raw_max=group.raw_observed_max_CD8.max(),
                fraction_below_L=group.raw_below_L_count.sum() / n if n else np.nan,
                fraction_above_U=group.raw_above_U_count.sum() / n if n else np.nan,
                fraction_at_zero=group.raw_at_or_below_L_count.sum() / n
                if n
                else np.nan,
                fraction_at_one=group.raw_at_or_above_U_count.sum() / n
                if n
                else np.nan,
                calibration_valid=bool(group.calibration_valid.all()),
            )
        )
    csv_write(args.run_dir / "calibration/clipping_audit.csv", pd.DataFrame(audits))

    missing = []

    for (split, context), group in frame.groupby(["split", "context"]):
        for name in legacy.FEATURES:
            missing.append(
                dict(
                    split=split,
                    context=context,
                    feature=name,
                    rows=len(group),
                    missing=int(group[name].isna().sum()),
                    zero=int((group[name] == 0).sum()),
                    finite=int(np.isfinite(group[name]).sum()),
                )
            )
    csv_write(args.run_dir / "feature_missingness.csv", pd.DataFrame(missing))

    metadata = dict(
        normalization_method=METHOD_ID,
        input_scale="unit-common clipped float [0,1]",
        morphology="original explicit-key geometry unchanged",
        protocol_sha256=signature,
    )

    parquet_write(output, frame, metadata)

    json_write(
        manifest,
        dict(
            status="complete",
            method_id=METHOD_ID,
            rows=len(frame),
            n_units=frame.image_id.nunique(),
            feature_file=output.name,
            feature_sha256=sha256(output),
            protocol_sha256=signature,
            calibration_file="calibration/unit_bounds.csv",
            calibration_sha256=sha256(args.run_dir / "calibration/unit_bounds.csv"),
            all_keys_sha256=protocol["all_keys_sha256"],
            source_feature_sha256=protocol["source_feature_sha256"],
            feature_columns=legacy.FEATURES,
            schema_metadata=metadata,
            code_sha256=code_hashes(args.run_dir),
            dictionary_sha256=sha256(args.run_dir / "FEATURE_DICTIONARY.csv"),
        ),
    )

    print("EXTRACTION_COMPLETE", len(frame), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--run-dir", type=Path, required=True)

    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)

    parser.add_argument(
        "--stage",
        choices=["prepare", "pilot", "extract", "merge", "all"],
        default="all",
    )

    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--chunk-size", type=int, default=2000)

    parser.add_argument("--recompute-morphology", action="store_true")

    args = parser.parse_args()

    if args.workers < 1 or args.chunk_size < 1:
        parser.error("workers/chunk-size must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    for stage in (
        ["prepare", "pilot", "extract", "merge"]
        if args.stage == "all"
        else [args.stage]
    ):
        print("START", stage, time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)

        globals()[stage](args)


if __name__ == "__main__":
    main()
