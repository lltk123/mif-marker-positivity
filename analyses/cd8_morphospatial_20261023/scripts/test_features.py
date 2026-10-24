"""Controlled-image checks for morphology and CD8 spatial features."""

from pathlib import Path
import sys, json, importlib.util, time
import numpy as np
from scipy import ndimage as ndi
from mif_features import *

A = Path(__file__).resolve().parents[1]
rows = []
cases = {}
cfg = FeatureConfig(mask_type="expanded_nucleus")
y, x = np.indices((80, 80))
M = (x - 40) ** 2 + (y - 40) ** 2 <= 12**2
lab = np.zeros(M.shape, np.int32)
lab[M] = 37
D = ndi.distance_transform_edt(np.pad(M, 1))[1:-1, 1:-1]
edge = M & (D <= max(1, 0.2 * D.max()))
cases["uniform"] = (np.where(M, 10.0, 0.0), lab)
cases["ring"] = (np.where(edge, 10.0, 0.0), lab)
spot = M & ((x - 50) ** 2 + (y - 40) ** 2 <= 3**2)
cases["one_sided"] = (np.where(spot, 30.0, 0.0), lab)
cases["dark"] = (np.zeros(M.shape), lab)
neighbor = (x - 65) ** 2 + (y - 40) ** 2 <= 11**2
nl = lab.copy()
nl[neighbor] = 900
ni = np.where(neighbor, 20.0, 0.0)
cases["bright_neighbor_without_spill"] = (ni, nl)
near = edge & (ndi.distance_transform_edt(~neighbor) <= cfg.neighbor_distance)
spill = ni.copy()
spill[near] = 12
cases["bright_neighbor_with_spill"] = (spill, nl)
tiny = np.zeros((10, 10), np.int32)
tiny[5, 5] = 37
cases["tiny"] = (np.ones((10, 10)), tiny)
no_ref = lab.copy()
no_ref[~M] = 900
cases["no_reference"] = (np.ones(M.shape) * 10, no_ref)
trunc = np.zeros((25, 25), np.int32)
trunc[:10, 4:18] = 37
cases["image_boundary"] = (np.ones((25, 25)), trunc)
res = {}
for name, (im, mask) in cases.items():
    r, regions = extract_roi(
        im,
        mask,
        37,
        dict(dataset_id="controlled", image_id=name),
        cfg,
        return_regions=True,
    )
    res[name] = r
    rows.append(r)
    assert not any(np.isinf(r[k]) for k in FEATURES + META_FLOAT if r[k] is not None)
    np.savez_compressed(A / "tests" / (name + ".npz"), image=im, mask=mask, **regions)
assert abs(res["uniform"]["edge_inner_mean_diff"]) < 1e-12
assert abs(res["uniform"]["edge_signal_enrichment"] - 1) < 1e-12
assert (
    res["ring"]["edge_inner_mean_diff"] > 9
    and res["ring"]["edge_signal_enrichment"] > 1.5
)
assert res["one_sided"]["sector_signal_cv"] > res["uniform"]["sector_signal_cv"] + 1
assert res["one_sided"]["signal_centroid_offset_normalized"] > 0.5
assert res["bright_neighbor_without_spill"]["whole_mean"] == 0
assert res["bright_neighbor_without_spill"]["neighbor_edge_area_fraction"] > 0
assert res["bright_neighbor_with_spill"]["neighbor_edge_mean_diff"] > 10
for key in [
    "edge_inner_normalized_diff",
    "edge_inner_log2_ratio",
    "edge_signal_enrichment",
    "signal_centroid_offset_normalized",
    "high_signal_edge_coverage",
]:
    assert np.isnan(res["dark"][key]), key
assert res["dark"]["qc_no_signal"] and res["dark"]["qc_reference_mad_zero"]
assert res["tiny"]["qc_small_whole"] and np.isnan(res["tiny"]["whole_mean"])
assert res["no_reference"]["qc_reference_insufficient"] and np.isnan(
    res["no_reference"]["local_reference_median"]
)
assert res["image_boundary"]["qc_truncated"] and all(
    np.isnan(res["image_boundary"][k]) for k in MORPH + WHOLE + REGIONAL
)
empty = extract_image(
    np.zeros((10, 10)), np.zeros((10, 10), np.int32), "test", "empty", cfg
)
assert len(empty) == 0 and list(empty.columns) == COLUMNS
multi = np.zeros((80, 80), np.int32)
multi[10:20, 10:20] = 7
multi[40:55, 40:55] = 105
f = extract_image(np.ones((80, 80)), multi, "test", "ids", cfg)
assert f.cell_id.tolist() == [7, 105]
# Compare the explicit 19-column implementation against the actual released HTK function.
spec = importlib.util.spec_from_file_location(
    "htk_reference", A / "provenance/histomicstk_1_4_0_reference.py"
)
htk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(htk)
comparison = []
for name, mask in [
    ("disk", M),
    ("ellipse", ((x - 40) / 8) ** 2 + ((y - 40) / 16) ** 2 <= 1),
    ("rectangle", (abs(x - 40) < 8) & (abs(y - 40) < 16)),
]:
    actual = htk.compute_morphometry_features(mask.astype(np.uint8))
    assert all(c in actual.columns for c in HTK_COLUMNS)
    expected = actual[HTK_COLUMNS].iloc[0].to_numpy(float)
    ours = np.array([morphometry(mask)[k] for k in MORPH])
    assert np.allclose(expected, ours, rtol=1e-12, atol=1e-12, equal_nan=True), (
        name,
        expected,
        ours,
    )
    comparison.append(
        dict(case=name, max_absolute_difference=float(np.max(abs(expected - ours))))
    )
fixed_frame(rows).to_csv(A / "tests/controlled_features.csv", index=False)
report = dict(
    status="passed",
    controlled_cases=list(cases),
    fixed_empty_schema_columns=len(COLUMNS),
    morphology_count=len(MORPH),
    numeric_model_candidates=len(FEATURES),
    actual_histomicstk_columns=list(actual.columns),
    selected_histomicstk_columns=HTK_COLUMNS,
    histomicstk_version="1.4.0 source loaded independently, package not installed",
    comparisons=comparison,
    definitions_differ="Degenerate shapes and truncated/small regions produce NaN rather than HTK finite fallback denominators. Orientation follows actual skimage row-axis radians, not the horizontal-axis wording in HTK docstring.",
)
(A / "tests/CONTROLLED_TESTS_PASSED.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
