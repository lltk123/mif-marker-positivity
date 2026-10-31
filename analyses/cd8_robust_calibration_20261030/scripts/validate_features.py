"""Controlled tests plus supplied real crops; never claims full-unit completion."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from calibration_core import *
from feature_adapter import *


def check(name, condition, records, detail=""):
    records.append(dict(test=name, passed=bool(condition), detail=detail))
    if not condition:
        raise AssertionError(name + ": " + detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--local-delivery", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path)
    args = parser.parse_args()
    old = args.legacy_dir or args.run_dir.parent / "cd8_morphospatial_20261023"
    legacy = load_legacy(old)
    config = legacy.FeatureConfig(
        **json.loads((old / "RUN_CONFIG.json").read_text())["config"]
    )
    records = []
    x = np.array([[0, 1, 4, 10, np.nan]], float)
    mapped = normalize_pixels(x, 1, 4, np.array([[1, 1, 1, 1, 0]], bool))
    check(
        "finite_support_and_clip",
        np.array_equal(mapped, np.array([[0, 0, 1, 1, np.nan]]), equal_nan=True),
        records,
    )
    check("invalid_bounds_NaN", np.isnan(normalize_pixels(x, 1, 1)).all(), records)
    check(
        "clipping_requires_pixel_reextraction",
        not np.isclose(
            np.mean(normalize_pixels(x[:, :4], 1, 4)),
            np.clip((np.mean(x[:, :4]) - 1) / 3, 0, 1),
        ),
        records,
    )
    # Unequal instance areas must not change the equally weighted outer quantiles.
    table = pd.DataFrame(
        dict(
            dataset_id=["D"] * 3,
            image_id=["I"] * 3,
            cell_id=[7, 31, 999],
            split=["training_WSIs"] * 3,
            context=["WSI"] * 3,
            finite_pixel_count=[5, 1000, 5],
            raw_cell_p10=[0.0, 10.0, 100.0],
            raw_cell_p90=[5.0, 20.0, 200.0],
        )
    )
    fitted, _ = fit_unit_bounds(table)
    check(
        "equal_instance_outer_quantiles",
        np.isclose(fitted.L.iloc[0], 0.2) and np.isclose(fitted.U.iloc[0], 196.4),
        records,
    )
    shuffled, _ = fit_unit_bounds(table.iloc[[2, 0, 1]])
    check("row_order_independence", fitted.equals(shuffled), records)
    failed = False
    try:
        fit_unit_bounds(pd.concat([table, table.iloc[[0]]]))
    except ValueError:
        failed = True
    check("duplicate_key_guard", failed, records)
    yy, xx = np.indices((65, 65))
    mask = np.zeros((65, 65), np.int32)
    mask[(yy - 32) ** 2 + (xx - 32) ** 2 <= 12**2] = 41
    d = ndi.distance_transform_edt(mask == 41)
    edge = (mask == 41) & (d <= max(1, 0.2 * d.max()))
    meta = dict(
        dataset_id="D",
        image_id="controlled",
        split="training_WSIs",
        context="controlled",
    )
    uniform = np.zeros(mask.shape, float)
    uniform[mask == 41] = 0.7
    ring = np.zeros(mask.shape, float)
    ring[edge] = 0.8
    bright = np.zeros(mask.shape, float)
    bright[edge & (xx > 36)] = 0.8
    call = lambda im, lab=mask: extract_normalized_roi(
        legacy, im, lab, 41, meta, config, 0, 1
    )
    u, r, b = call(uniform), call(ring), call(bright)
    check(
        "uniform_edge_inner_difference", abs(u["edge_inner_mean_diff"]) < 1e-12, records
    )
    check(
        "uniform_area_corrected_enrichment",
        abs(u["edge_signal_enrichment"] - 1) < 1e-12,
        records,
    )
    check(
        "ring_edge_enrichment",
        r["edge_signal_enrichment"] > u["edge_signal_enrichment"]
        and r["edge_inner_mean_diff"] > u["edge_inner_mean_diff"],
        records,
    )
    check(
        "one_sided_signal_offset",
        b["signal_centroid_offset_normalized"] > u["signal_centroid_offset_normalized"],
        records,
    )
    for name, im, lab in [
        ("dark", np.zeros(mask.shape), mask),
        ("tiny", np.zeros((9, 9)), np.pad(np.array([[41]], np.int32), 4)),
        ("truncated", np.ones((9, 9)), np.full((9, 9), 41, np.int32)),
    ]:
        row = call(im, lab)
        check(
            name + "_finite_or_nan",
            not np.isinf([row[k] for k in legacy.FEATURES]).any(),
            records,
        )
        check(name + "_has_QC", bool(row["qc_flags"]), records)
    absent_reference = np.full(mask.shape, 5, np.int32)
    absent_reference[mask == 41] = 41
    no_ref = call(uniform, absent_reference)
    check(
        "no_reference_is_nan",
        np.isnan(no_ref["local_reference_median"])
        and no_ref["qc_reference_insufficient"],
        records,
    )
    check(
        "empty_mask_fixed_schema",
        legacy.extract_image(
            np.zeros((8, 8)), np.zeros((8, 8), np.int32), "D", "empty"
        ).shape
        == (0, len(legacy.COLUMNS)),
        records,
    )
    original_morphometry = legacy.extract_roi.__globals__["morphometry"]
    cases = json.loads(
        (args.local_delivery / "examples/example_index.json").read_text(
            encoding="utf-8-sig"
        )
    )
    audits = []
    for case in cases:
        path = args.local_delivery / "examples" / (case["example_id"] + ".npz")
        z = np.load(path)
        cid = int(case["cell_id"])
        im = z["image"]
        lab = z["mask"]
        valid = z["valid"]
        metadata = {
            k: v
            for k, v in case["features"].items()
            if k in legacy.META_STR and k != "qc_flags"
        }
        metadata.update(dataset_id="PhenoBIC-Data", image_id=case["image_id"])
        origin = (case["origin_y"], case["origin_x"])
        force = bool(case["features"].get("qc_truncated", False))
        raw = legacy.extract_roi(
            im,
            lab,
            cid,
            metadata,
            config,
            valid_mask=valid,
            origin=origin,
            force_truncated=force,
        )
        finite = im[valid & np.isfinite(im)]
        # Deliberately diagnostic crop bounds, not estimates of full-unit L/U.
        L, U = np.percentile(finite, [5, 95]).astype(float)
        source = pd.DataFrame(
            [
                dict(
                    dataset_id="PhenoBIC-Data",
                    image_id=case["image_id"],
                    cell_id=cid,
                    **{k: raw[k] for k in legacy.MORPH},
                )
            ]
        )
        cache = MorphologyCache(source, legacy.MORPH, sha256(path))
        kwargs = dict(valid_mask=valid, origin=origin, force_truncated=force)
        reference = extract_normalized_roi(
            legacy, im, lab, cid, metadata, config, L, U, **kwargs
        )
        fast = extract_normalized_roi(
            legacy,
            im,
            lab,
            cid,
            metadata,
            config,
            L,
            U,
            morphology_cache=cache,
            **kwargs,
        )
        a = np.array([reference[k] for k in legacy.FEATURES])
        c = np.array([fast[k] for k in legacy.FEATURES])
        exact = np.array_equal(a, c, equal_nan=True)
        if not exact:
            print(
                [
                    (k, reference[k], fast[k], raw[k])
                    for k in legacy.FEATURES
                    if not np.array_equal([reference[k]], [fast[k]], equal_nan=True)
                ]
            )
        check("real_" + case["example_id"] + "_all90_cache_reference", exact, records)
        geometry_equal = np.array_equal(
            [reference[k] for k in legacy.MORPH],
            [raw[k] for k in legacy.MORPH],
            equal_nan=True,
        )
        check(
            "real_" + case["example_id"] + "_geometry_unchanged",
            geometry_equal,
            records,
        )
        normalized = normalize_pixels(im, L, U, valid)
        values = normalized[(lab == cid) & valid & np.isfinite(im)]
        manual = float(np.mean(values))
        check(
            "real_" + case["example_id"] + "_pixel_mean",
            np.isclose(
                manual,
                reference["observed_mean_CD8"],
                rtol=0,
                atol=1e-15,
                equal_nan=True,
            ),
            records,
        )
        old_morph = [case["features"][k] for k in legacy.MORPH]
        supplied_equal = np.allclose(
            old_morph,
            [raw[k] for k in legacy.MORPH],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        check(
            "real_" + case["example_id"] + "_supplied_geometry_agrees",
            supplied_equal,
            records,
        )
        gain = extract_normalized_roi(
            legacy,
            im.astype(float) * 2,
            lab,
            cid,
            metadata,
            config,
            L * 2,
            U * 2,
            **kwargs,
        )
        check(
            "real_" + case["example_id"] + "_gain_invariance",
            np.allclose(
                a,
                [gain[k] for k in legacy.FEATURES],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ),
            records,
        )
        audits.append(
            dict(
                example_id=case["example_id"],
                image_id=case["image_id"],
                cell_id=cid,
                L=L,
                U=U,
                bounds_scope="diagnostic crop P5/P95; NOT full-unit calibration",
                all90_cache_reference_exact=exact,
                geometry_unchanged=geometry_equal,
                supplied_geometry_agrees=supplied_equal,
                raw_mean=fast["raw_observed_mean_CD8"],
                normalized_mean=fast["observed_mean_CD8"],
                input_sha256=sha256(path),
            )
        )
    check(
        "original_module_globals_unchanged",
        legacy.extract_roi.__globals__["morphometry"] is original_morphometry,
        records,
    )
    check("five_real_sources", len(set(x["image_id"] for x in audits)) == 5, records)
    out = args.run_dir / "tests/local_validation"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "checks.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "real_crop_validation.csv", index=False)
    payload = dict(
        status="passed",
        scope="controlled + 15 supplied crops / 5 sources; no full-unit bounds or full extraction claimed",
        checks=len(records),
        cases=len(audits),
        code_sha256={
            p.name: sha256(p) for p in sorted((args.run_dir / "scripts").glob("*.py"))
        },
    )
    (out / "PASSED.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
