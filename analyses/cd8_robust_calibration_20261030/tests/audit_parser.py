from pathlib import Path
import pandas as pd
import numpy as np
import json

r = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030"
)
a = pd.read_csv(r / "calibration/unit_bounds.csv")
b = pd.read_csv(r / "calibration/unit_bounds.csv", float_precision="round_trip")
rows = []
for i in range(len(a)):
    for c in ["L", "U"]:
        if a.loc[i, c] != b.loc[i, c]:
            rows.append(
                dict(
                    image_id=a.loc[i, "image_id"],
                    bound=c,
                    default_value=float(a.loc[i, c]),
                    round_trip_value=float(b.loc[i, c]),
                    difference=float(a.loc[i, c] - b.loc[i, c]),
                )
            )
p = r / "tests/implementation_fix_001/parser_differences.csv"
assert not p.exists()
pd.DataFrame(
    rows,
    columns=["image_id", "bound", "default_value", "round_trip_value", "difference"],
).to_csv(p, index=False, float_format="%.17g")
print(
    json.dumps(
        dict(
            different_bound_values=len(rows),
            max_abs_difference=max([abs(x["difference"]) for x in rows] or [0]),
            pandas=pd.__version__,
            numpy=np.__version__,
        )
    )
)
