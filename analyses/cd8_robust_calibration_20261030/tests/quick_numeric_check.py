from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd

r = Path(
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030"
)
sys.path.insert(0, str(r / "scripts"))
import run_features as f

f.initialize(str(r), str(r.parent.parent))
old = f.STATE["old"]
reader = f.STATE["reader"]
legacy = f.STATE["legacy"]
rows = []
for split, context in [("training_WSIs", None), ("internal_val_WSIs", None)] + [
    ("external_val_TMA_cores", "TMA%d_CD8" % i) for i in [1, 2, 3]
]:
    for e in sorted(reader.AUDIT, key=lambda x: x["key"]):
        if e["split"] != split or (context is not None and e["context"] != context):
            continue
        with np.load(old / "bboxes" / (e["key"] + ".npz")) as z:
            ids = z["cell_id"].copy()
            boxes = z["bbox"].copy()
        if len(ids):
            break
    for i in sorted(set([0, len(ids) // 2, len(ids) - 1])):
        t = time.monotonic()
        a = f.extract_one(e, int(ids[i]), boxes[i])
        b = f.extract_one(e, int(ids[i]), boxes[i], recompute=True)
        assert np.array_equal(
            [a[k] for k in legacy.FEATURES],
            [b[k] for k in legacy.FEATURES],
            equal_nan=True,
        )
        rows.append(
            dict(
                image_id=a["image_id"],
                cell_id=a["cell_id"],
                all90_exact=True,
                seconds=time.monotonic() - t,
            )
        )
        print(rows[-1], flush=True)
out = r / "tests/quick_numeric_check.csv"
assert not out.exists()
pd.DataFrame(rows).to_csv(out, index=False)
print("QUICK_NUMERIC_PASSED", len(rows), flush=True)
