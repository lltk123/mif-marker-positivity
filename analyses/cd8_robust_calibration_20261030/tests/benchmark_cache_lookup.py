from pathlib import Path
import sys, time, json
import pandas as pd
import numpy as np

r = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030"
)
sys.path.insert(0, str(r / "scripts"))
from feature_adapter import load_legacy, MorphologyCache

old = r.parent / "cd8_morphospatial_20261023"
legacy = load_legacy(old)
f = pd.read_parquet(
    old / "PhenoBIC_CD8_features.parquet",
    columns=["dataset_id", "image_id", "cell_id"]
    + legacy.MORPH
    + ["observed_mean_CD8", "observed_pixel_count", "whole_valid_pixel_count"],
)
f = f.set_index(["dataset_id", "image_id", "cell_id"]).sort_index()
keys = list(f.index[:: max(1, len(f) // 100)])[:100]
columns = legacy.MORPH
records = []
for mode in ["combined_loc", "row_then_columns"]:
    t = time.monotonic()
    values = []
    for key in keys:
        row = f.loc[key, columns] if mode == "combined_loc" else f.loc[key][columns]
        values.append(row.to_numpy(float))
    records.append(dict(mode=mode, queries=len(keys), seconds=time.monotonic() - t))
    if mode == "combined_loc":
        reference = values
    else:
        assert np.array_equal(reference, values, equal_nan=True)
    print(records[-1], flush=True)
p = r / "tests/cache_lookup_benchmark.csv"
assert not p.exists()
pd.DataFrame(records).to_csv(p, index=False)
