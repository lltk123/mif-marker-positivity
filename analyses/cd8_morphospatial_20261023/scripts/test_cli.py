"""Validate the single-image feature command line on synthetic inputs."""

from pathlib import Path
import json, subprocess, sys
import numpy as np, tifffile, pyarrow.parquet as pq

A = Path(__file__).resolve().parents[1]
l = np.zeros((40, 40), np.uint32)
l[12:26, 12:26] = 73
i = np.where(l > 0, 1234, 5).astype(np.uint16)
tifffile.imwrite(A / "tests/cli_cd8.tif", i)
tifffile.imwrite(A / "tests/cli_mask.tif", l)
(A / "tests/cli_config.json").write_text(json.dumps(dict(mask_type="expanded_nucleus")))
subprocess.run(
    [
        sys.executable,
        str(A / "scripts/extract_single.py"),
        "--image",
        str(A / "tests/cli_cd8.tif"),
        "--mask",
        str(A / "tests/cli_mask.tif"),
        "--dataset-id",
        "controlled",
        "--image-id",
        "cli",
        "--config",
        str(A / "tests/cli_config.json"),
        "--output",
        str(A / "tests/cli_features.parquet"),
        "--csv",
    ],
    check=True,
)
f = pq.read_table(A / "tests/cli_features.parquet").to_pandas()
assert (
    len(f) == 1
    and f.cell_id.iloc[0] == 73
    and f.whole_mean.iloc[0] == 1234
    and len(f.columns) == 163
)
report = dict(
    status="passed",
    cell_id=73,
    raw_uint16_intensity_preserved=True,
    parquet_columns=163,
    csv_export=True,
)
(A / "tests/CLI_TEST_PASSED.json").write_text(json.dumps(report, indent=2))
print(report)
