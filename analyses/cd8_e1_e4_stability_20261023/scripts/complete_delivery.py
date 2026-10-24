"""Research complete delivery for the frozen PhenoBIC CD8 experiment."""

import json
import numpy as np, pandas as pd
from common import *
from io_adapter import BYKEY, GROUPS, boxes, roi, read_image


def main():
    f = pd.read_parquet(
        A / "PhenoBIC_E1_E4_features.parquet",
        columns=KEYS
        + ["source_key", "intensity_input_valid"]
        + [
            c
            for c in __import__("pyarrow.parquet", fromlist=["read_schema"])
            .read_schema(A / "PhenoBIC_E1_E4_features.parquet")
            .names
            if c.startswith("qc_")
        ],
    )
    q = f[
        KEYS + ["intensity_input_valid"] + [c for c in f if c.startswith("qc_")]
    ].copy()
    q["calibration_valid"] = ~q.qc_calibration_invalid
    for mode in ["fixed_0.5", "OOF_threshold"]:
        p = A / "predictions" / ("training_OOF_" + mode + ".parquet")
        d = pd.read_parquet(p)
        merged = d.merge(q, on=KEYS, validate="one_to_one")
        assert len(merged) == 101057
        dest = p.with_name(p.stem + "_with_QC.parquet")
        assert not dest.exists()
        merged.to_parquet(dest, index=False, compression="zstd")
    sample = pd.read_parquet(A / "keys/stability_sample.parquet").merge(
        f, on=KEYS, validate="one_to_one"
    )
    checks = []
    for uid in json.loads((A / "audit/representative_units.json").read_text()):
        g = sample[(sample.image_id == uid) & sample.intensity_input_valid].sort_values(
            "cell_id"
        )
        chosen = None
        for row in g.itertuples():
            e = BYKEY[row.source_key]
            ids, bb = boxes(e)
            box = bb[np.flatnonzero(ids == row.cell_id)[0]]
            im, lab, v, origin, trunc = roi(e, box, 8)
            if v.all():
                chosen = (row, e, im, lab, v, origin)
                break
        assert chosen is not None
        row, e, im, lab, v, origin = chosen
        if e["split"] == "internal_val_WSIs":
            entries = GROUPS[(e["split"], e["context"])]
            oy = min(x["offset_y"] for x in entries)
            ox = min(x["offset_x"] for x in entries)
            H = max(x["offset_y"] + x["shape"][0] for x in entries) - oy
            W = max(x["offset_x"] + x["shape"][1] for x in entries) - ox
            fullim = np.zeros((H, W), dtype=e["image_dtype"])
            fullmask = np.zeros((H, W), dtype=e["mask_dtype"])
            fullvalid = np.zeros((H, W), bool)
            for x in entries:
                sl = (
                    slice(x["offset_y"] - oy, x["offset_y"] - oy + x["shape"][0]),
                    slice(x["offset_x"] - ox, x["offset_x"] - ox + x["shape"][1]),
                )
                fi = read_image(DATA / x["source"]["image"])
                fm = read_image(DATA / x["source"]["mask"])
                fullim[sl] = fi
                fullmask[sl] = np.where(fm > 0, fm, fullmask[sl])
                fullvalid[sl] = True
        else:
            fullim = read_image(DATA / e["source"]["image"])
            fullmask = read_image(DATA / e["source"]["mask"])
            fullvalid = np.ones(fullim.shape, bool)
            oy = e["offset_y"]
            ox = e["offset_x"]
            H, W = fullim.shape
        y1 = origin[0] - oy
        x1 = origin[1] - ox
        y2 = y1 + im.shape[0]
        x2 = x1 + im.shape[1]
        assert np.array_equal(fullim[y1:y2, x1:x2], im) and np.array_equal(
            fullmask[y1:y2, x1:x2], lab
        )
        for k in [1, 2, 3]:
            if k == 1:
                b = (W - x2, W - x1, y1, y2)
            elif k == 2:
                b = (H - y2, H - y1, W - x2, W - x1)
            else:
                b = (x1, x2, H - y2, H - y1)
            sl = (slice(b[0], b[1]), slice(b[2], b[3]))
            a = np.rot90(fullim, k)[sl]
            bmask = np.rot90(fullmask, k)[sl]
            bvalid = np.rot90(fullvalid, k)[sl]
            passed = (
                np.array_equal(a, np.rot90(im, k))
                and np.array_equal(bmask, np.rot90(lab, k))
                and np.array_equal(bvalid, np.rot90(v, k))
            )
            assert passed
            checks.append(
                dict(
                    image_id=uid,
                    cell_id=int(row.cell_id),
                    rotation_degrees=k * 90,
                    full_canvas_height=H,
                    full_canvas_width=W,
                    full_image_and_all_instance_IDs_rotated=True,
                    all_context_pixels_identical=passed,
                    method="Rotate full original field/complete internal FOV union, then crop transformed global coordinates; compare against joint ROI rotation",
                )
            )
        del fullim, fullmask, fullvalid
    pd.DataFrame(checks).to_csv(A / "tests/full_field_rotation_checks.csv", index=False)
    write_json(
        A / "results/DELIVERY_AUGMENTATION.json",
        dict(
            OOF_with_QC=True,
            OOF_score_changes=False,
            full_field_rotation_checks=len(checks),
            passed=True,
        ),
    )
    print("DELIVERY AUGMENTATION COMPLETE", len(checks))


if __name__ == "__main__":
    main()
