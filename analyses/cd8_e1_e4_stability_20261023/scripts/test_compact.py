"""Controlled checks for the frozen E1-E4 intensity and spatial feature definitions."""

import json, time
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from common import *
from compact_features import *
from io_adapter import AUDIT, boxes, roi, image_id
from legacy_features import extract_roi, FeatureConfig, WHOLE


def main():
    cfg = FeatureConfig(mask_type="expanded_nucleus")
    y, x = np.indices((90, 90))
    M = ((x - 45) / 13) ** 2 + ((y - 45) / 9) ** 2 <= 1
    lab = np.where(M, 37, 0).astype(np.int32)
    raw = 0.2 + 0.03 * np.sin(x / 4) + 0.02 * np.cos(y / 3)
    raw[M] += 1
    raw[M & (x > 51) & (y < 45)] += 3
    r = extract_compact(raw, lab, 37, 0, 4)
    norm = normalize(raw, 0, 4)
    old = extract_roi(norm, lab, 37, config=cfg)
    for a, b in zip(I8, WHOLE):
        assert np.isclose(r[a], old[b], equal_nan=True), a
    for c in [
        "edge_inner_normalized_diff",
        "edge_signal_enrichment",
        "high_signal_edge_coverage",
        "signal_centroid_offset_normalized",
        "neighbor_edge_area_fraction",
    ]:
        assert np.isclose(r[c], old[c], equal_nan=True), c
    rr = []
    for k in [1, 2, 3]:
        q = extract_compact(np.rot90(raw, k), np.rot90(lab, k), 37, 0, 4)
        delta = max(
            abs(q[c] - r[c])
            for c in ["angular_R1", "angular_R2", "signal_centroid_offset_normalized"]
        )
        assert delta < 1e-12
        rr.append(dict(rotation=k * 90, max_invariant_difference=delta))
    for a, b in [(0.5, 0), (2, 0), (1, 0.2), (1, 0.4)]:
        q = extract_compact(a * raw + b, lab, 37, b, 4 * a + b)
        for c in I8 + S9:
            assert np.isclose(q[c], r[c], rtol=1e-10, atol=1e-12, equal_nan=True), c
    # Clipping is applied on pixels before computing summaries.
    v = np.array([-10, 0, 1, 2, 3, 100], float)
    ll = np.zeros((20, 20), np.int32)
    ll[8:10, 8:11] = 37
    im = np.zeros(ll.shape)
    im[ll == 37] = v
    z = extract_compact(im, ll, 37, 0, 3)
    assert np.isclose(
        z["norm_whole_mean"], np.clip(v / 3, 0, 1).mean()
    ) and not np.isclose(z["norm_whole_mean"], v.mean() / 3)
    gap = np.zeros((12, 12), np.int32)
    gap[5, 4] = 3
    gap[5, 6] = 70
    d = perturb_mask(gap, "dilate_1")
    assert d[5, 5] == 0 and d[5, 4] == 3 and d[5, 6] == 70
    er = perturb_mask(gap, "erode_1")
    assert not er.any()
    missing = extract_compact(np.ones(gap.shape), er, 3, 0, 1)
    assert missing["qc_mask_missing"] and not missing["intensity_input_valid"]
    for cid in [37]:
        expected = ndi.distance_transform_edt(np.pad(lab == cid, 1))[1:-1, 1:-1] > 1
        assert np.array_equal(perturb_mask(lab, "erode_1") == cid, expected)
    invalid = extract_compact(raw, lab, 37, 1, 1)
    assert invalid["qc_calibration_invalid"] and all(
        np.isnan(invalid[c]) for c in I8 + S9
    )
    dark = extract_compact(np.zeros(lab.shape), lab, 37, 0, 1)
    assert (
        np.isnan(dark["angular_R1"])
        and np.isnan(dark["edge_signal_enrichment"])
        and dark["qc_reference_mad_zero"]
    )
    no = lab.copy()
    no[no == 0] = 73
    z = extract_compact(raw, no, 37, 0, 4)
    assert z["qc_reference_insufficient"] and np.isnan(z["edge_signal_enrichment"])
    bounded = lab.copy()
    valid = np.ones(lab.shape, bool)
    valid[:5] = False
    z = extract_compact(raw, bounded, 37, 0, 4, valid)
    assert z["qc_context_clipped"]
    tests = dict(
        status="passed",
        rotations=rr,
        checks=[
            "old8_match_on_normalized_pixels",
            "five_legacy_spatial_formulas_match",
            "rotation_invariance_R1_R2_offset",
            "gain_bias_pixel_invariance",
            "clip_before_summaries",
            "dilation_ties_background",
            "erosion_Euclidean_1px",
            "vanished_instance_preserved",
            "invalid_calibration_NaN",
            "dark_and_no_reference_QC",
            "invalid_support",
        ],
    )
    write_json(A / "tests/CONTROLLED_TESTS_PASSED.json", tests)
    selected = json.loads((A / "audit/representative_units.json").read_text())
    checks = []
    for uid in selected:
        entries = [e for e in AUDIT if image_id(e) == uid]
        count = 0
        for e in entries:
            ids, bb = boxes(e)
            if len(ids) == 0:
                continue
            inds = np.linspace(0, len(ids) - 1, min(20, len(ids))).astype(int)
            for ix in inds:
                im, mask, valid, origin, tr = roi(e, bb[ix])
                cid = int(ids[ix])
                v = im[(mask == cid) & valid & np.isfinite(im)].astype(float)
                L = float(np.percentile(v, 10))
                U = float(np.percentile(v, 90))
                if not calibration_valid(L, U):
                    continue
                new = extract_compact(im, mask, cid, L, U, valid, tr)
                old = extract_roi(
                    normalize(im, L, U, valid),
                    mask,
                    cid,
                    config=cfg,
                    valid_mask=valid,
                    force_truncated=tr,
                )
                for a, b in zip(I8, WHOLE):
                    assert np.isclose(
                        new[a], old[b], rtol=1e-10, atol=1e-12, equal_nan=True
                    ), (uid, cid, a)
                for c in [
                    "edge_inner_normalized_diff",
                    "edge_signal_enrichment",
                    "high_signal_edge_coverage",
                    "signal_centroid_offset_normalized",
                    "neighbor_edge_area_fraction",
                ]:
                    assert np.isclose(
                        new[c], old[c], rtol=1e-10, atol=1e-12, equal_nan=True
                    ), (uid, cid, c)
                for k in [1, 2, 3]:
                    rot = extract_compact(
                        np.rot90(im, k),
                        np.rot90(mask, k),
                        cid,
                        L,
                        U,
                        np.rot90(valid, k),
                        tr,
                    )
                    for c in I8 + S9:
                        assert np.isclose(
                            rot[c], new[c], rtol=1e-9, atol=1e-11, equal_nan=True
                        ), (uid, cid, c)
                count += 1
            if count >= 20:
                break
        checks.append(
            dict(
                image_id=uid,
                cells=count,
                legacy_formulas_match=True,
                rotation_invariance=True,
            )
        )
    pd.DataFrame(checks).to_csv(A / "tests/real_pilot_checks.csv", index=False)
    write_json(
        A / "tests/PILOT_PASSED.json",
        dict(
            status="passed",
            units=len(checks),
            cells=sum(d["cells"] for d in checks),
            tests=checks,
        ),
    )
    print("TESTS PASSED", tests, checks, flush=True)


if __name__ == "__main__":
    main()
