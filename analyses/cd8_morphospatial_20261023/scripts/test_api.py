"""Check the reusable feature interface and fixed output schema."""

from pathlib import Path
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mif_features import *

A = Path(__file__).resolve().parents[1]
cfg = FeatureConfig(mask_type="expanded_nucleus")
y, x = np.indices((70, 70))
lab = np.zeros((70, 70), np.int32)
lab[(x - 35) ** 2 + (y - 35) ** 2 < 100] = 17
im = np.where(lab == 17, 10.0, 2.0)
r = extract_roi(im, lab, 17, config=cfg)
# Sufficient ROI halo must reproduce whole-image results, including distance edge.
crop = extract_roi(im[17:54, 17:54], lab[17:54, 17:54], 17, config=cfg, origin=(17, 17))
for k in FEATURES + ["centroid_x_px", "centroid_y_px", "edge_width_px"]:
    assert np.isclose(r[k], crop[k], equal_nan=True), k
assert r["edge_signal_enrichment"] == 1
bad = im.copy()
bad[35, 35] = np.nan
b = extract_roi(bad, lab, 17, config=cfg)
assert b["qc_invalid_pixels"] and np.isnan(b["whole_mean"])
t = extract_roi(im, lab, 17, config=cfg, tissue_mask=lab > 0)
assert t["qc_reference_insufficient"] and not t["qc_reference_tissue_unknown"]
scaled = extract_roi(
    im, lab, 17, config=FeatureConfig(mask_type="expanded_nucleus", pixel_size_um=0.5)
)
assert (
    scaled["Area_um2"] == scaled["Area"] * 0.25
    and scaled["Perimeter_um"] == scaled["Perimeter"] * 0.5
)
broken = lab.copy()
broken[5, 5] = 17
b = extract_roi(im, broken, 17, config=cfg)
assert b["qc_disconnected"] and np.isnan(b["Area"])
for kwargs in [
    dict(labels=lab.astype(float)),
    dict(labels=lab[:-1]),
    dict(metadata={"ground_truth_binary": 1}),
    dict(tissue_mask=np.ones((2, 2))),
]:
    args = dict(image=im, labels=lab, cell_id=17, config=cfg)
    args.update(kwargs)
    try:
        extract_roi(**args)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("Invalid input accepted")
empty = extract_image(
    np.zeros((4, 4)), np.zeros((4, 4), np.int32), "synthetic", "empty"
)
schema = pa.schema(
    [
        (
            c,
            pa.string()
            if c in META_STR
            else pa.int64()
            if c in COUNTS + ["cell_id"]
            else pa.bool_()
            if c in QC
            else pa.float64(),
        )
        for c in COLUMNS
    ]
)
p = A / "tests/empty_fixed_schema.parquet"
pq.write_table(pa.Table.from_pandas(empty, schema=schema, preserve_index=False), p)
assert pq.read_schema(p) == schema and pq.read_metadata(p).num_rows == 0
report = dict(
    status="passed",
    checks=[
        "crop_and_padding_invariance",
        "nonfinite_target_QC",
        "tissue_exclusion",
        "optional_physical_units",
        "disconnected_QC",
        "invalid_dtype_and_shape_rejection",
        "outcome_metadata_rejection",
        "empty_Parquet_schema",
    ],
)
(A / "tests/API_TESTS_PASSED.json").write_text(json.dumps(report, indent=2))
print(report)
