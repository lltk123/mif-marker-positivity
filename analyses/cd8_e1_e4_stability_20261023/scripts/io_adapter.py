"""Read-only reuse of original instance index and overlapping-FOV reconstruction."""
from collections import OrderedDict, defaultdict
import json
import numpy as np
import tifffile
from common import A, OLD, G, DATA

AUDIT = json.loads((G / "features/phenobic_extraction_audit.json").read_text())
assert all(e["dataset"] == "PhenoBIC-Data" for e in AUDIT)
BYKEY = {e["key"]: e for e in AUDIT}
GROUPS = defaultdict(list)
CACHE = OrderedDict()
for e in AUDIT:
    GROUPS[(e["split"], e["context"])].append(e)


def image_id(e):
    return (
        e["split"]
        + "/"
        + e["context"]
        + ("/" + e["patch"] if e["split"] == "external_val_TMA_cores" else "")
    )


def read_image(path):
    path = str(path)
    if path in CACHE:
        CACHE.move_to_end(path)
        return CACHE[path]
    try:
        a = tifffile.memmap(path, mode="r").squeeze()
    except ValueError:
        a = tifffile.imread(path).squeeze()
    assert a.ndim == 2
    CACHE[path] = a
    while len(CACHE) > 6:
        CACHE.popitem(last=False)
    return a


def boxes(e):
    p = OLD / "bboxes" / (e["key"] + ".npz")
    z = np.load(p)
    old = np.load(G / "features" / (e["key"] + ".npz"))["cell_id"]
    assert np.array_equal(old, z["cell_id"])
    return z["cell_id"], z["bbox"]


def roi(e, bbox, halo=8):
    assert halo >= 8
    h, w = e["shape"]
    y1, y2, x1, x2 = map(int, bbox)
    gy1 = y1 + e["offset_y"] - halo
    gy2 = y2 + e["offset_y"] + halo
    gx1 = x1 + e["offset_x"] - halo
    gx2 = x2 + e["offset_x"] + halo
    shape = (gy2 - gy1, gx2 - gx1)
    im = np.zeros(shape, dtype=e["image_dtype"])
    lab = np.zeros(shape, dtype=e["mask_dtype"])
    valid = np.zeros(shape, bool)
    candidates = (
        GROUPS[(e["split"], e["context"])] if e["split"] == "internal_val_WSIs" else [e]
    )
    for q in candidates:
        top = max(gy1, q["offset_y"])
        bottom = min(gy2, q["offset_y"] + q["shape"][0])
        left = max(gx1, q["offset_x"])
        right = min(gx2, q["offset_x"] + q["shape"][1])
        if top >= bottom or left >= right:
            continue
        dst = (slice(top - gy1, bottom - gy1), slice(left - gx1, right - gx1))
        src = (
            slice(top - q["offset_y"], bottom - q["offset_y"]),
            slice(left - q["offset_x"], right - q["offset_x"]),
        )
        qi = read_image(DATA / q["source"]["image"])[src]
        qm = read_image(DATA / q["source"]["mask"])[src]
        overlap = valid[dst]
        if overlap.any():
            assert np.allclose(
                im[dst][overlap], qi[overlap], rtol=0, atol=0, equal_nan=True
            )
            assert not ((lab[dst] > 0) & (qm > 0) & (lab[dst] != qm)).any()
        im[dst] = qi
        lab[dst] = np.where(qm > 0, qm, lab[dst])
        valid[dst] = True
    return im, lab, valid, (gy1, gx1), bool(y1 == 0 or x1 == 0 or y2 == h or x2 == w)
