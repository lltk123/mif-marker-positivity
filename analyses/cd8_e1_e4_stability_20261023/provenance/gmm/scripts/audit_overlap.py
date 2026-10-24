"""Audit shared FOV coordinates and repeated instance observations."""

from pathlib import Path
import csv, json, sys, re
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_cd8 import read, P, A

root = P / "datasets/PhenoBIC-Data"
rows = [
    r
    for r in csv.DictReader((root / "qa/image_mask_label_index.csv").open())
    if r["split"] == "internal_val_WSIs"
]
prior = {}
issues = []
for r in rows:
    y, x = map(int, re.search(r"_y_(\d+)_x_(\d+)$", r["patch"]).groups())
    mask = read(root / r["mask"])
    h, w = mask.shape
    for q in prior.get(r["context"], []):
        top = max(y, q["y"])
        left = max(x, q["x"])
        bottom = min(y + h, q["y"] + q["h"])
        right = min(x + w, q["x"] + q["w"])
        if top >= bottom or left >= right:
            continue
        pm = read(root / q["r"]["mask"])
        a = mask[top - y : bottom - y, left - x : right - x]
        b = pm[top - q["y"] : bottom - q["y"], left - q["x"] : right - q["x"]]
        bad = a != b
        if bad.any():
            pairs, counts = np.unique(
                np.stack([a[bad], b[bad]], axis=1), axis=0, return_counts=True
            )
            nz = (a > 0) & (b > 0) & bad
            item = dict(
                context=r["context"],
                patch=r["patch"],
                other=q["r"]["patch"],
                mismatches=int(bad.sum()),
                both_nonzero_mismatches=int(nz.sum()),
                pairs=[
                    {"ids": p.tolist(), "pixels": int(c)}
                    for p, c in zip(pairs[:20], counts[:20])
                ],
            )
            issues.append(item)
    prior.setdefault(r["context"], []).append(dict(y=y, x=x, h=h, w=w, r=r))
(A / "provenance/internal_overlap_mask_audit.json").write_text(
    json.dumps(issues, indent=2)
)
print(
    "OVERLAPS WITH MISMATCHES",
    len(issues),
    "TOTAL PIXELS",
    sum(x["mismatches"] for x in issues),
    "BOTH NONZERO",
    sum(x["both_nonzero_mismatches"] for x in issues),
)
print(json.dumps(issues[:4], indent=2))
