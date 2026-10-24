"""Extract raw CD8 instance statistics without double-counting overlapping FOVs."""

import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
from pathlib import Path
import csv, json, time, re, hashlib, traceback
from collections import defaultdict
import numpy as np
import tifffile

P = Path("/mnt/disk3/mIHC/mif_marker_positivity_20261022")
A = P / "analyses/cd8_scimap_gmm_20261023"


def read(p):
    try:
        return tifffile.memmap(p, mode="r").squeeze()
    except ValueError:
        return tifffile.imread(p).squeeze()


def extract(dataset, rows):
    root = P / "datasets" / dataset
    prior = defaultdict(list)
    audit = []
    for k, r in enumerate(rows):
        split = r.get("split", r.get("subset"))
        ctx = r.get("context", r.get("fov"))
        patch = r.get("patch", r.get("fov"))
        key = hashlib.sha256(
            (dataset + "/" + split + "/" + ctx + "/" + patch).encode()
        ).hexdigest()[:20]
        dest = A / "features" / (key + ".npz")
        meta = A / "features" / (key + ".json")
        im = read(root / r["image"])
        mask = read(root / r["mask"])
        assert im.ndim == mask.ndim == 2 and im.shape == mask.shape
        h, w = mask.shape
        coord = (
            re.search(r"_y_(\d+)_x_(\d+)$", patch)
            if split == "internal_val_WSIs"
            else None
        )
        y0, x0 = map(int, coord.groups()) if coord else (0, 0)
        previous = prior[(dataset, split, ctx)] if coord else []
        if dest.exists() and meta.exists():
            info = json.loads(meta.read_text())
            audit.append(info)
            previous.append(dict(y=y0, x=x0, h=h, w=w, r=r))
            print("REUSE", k + 1, split, ctx, patch, flush=True)
            continue
        start = time.time()
        maxid = int(mask.max())
        assert mask.min() >= 0
        if maxid > 20000000:
            raise RuntimeError("Unexpected sparse huge mask IDs")
        n = maxid + 1
        count = np.zeros(n, np.int64)
        sums = np.zeros(n)
        xs = np.zeros(n)
        ys = np.zeros(n)
        bad = np.zeros(n, np.int64)
        overlap = np.zeros((h, w), bool) if previous else None
        overlapchecks = []
        for q in previous:
            top = max(y0, q["y"])
            left = max(x0, q["x"])
            bottom = min(y0 + h, q["y"] + q["h"])
            right = min(x0 + w, q["x"] + q["w"])
            if top >= bottom or left >= right:
                continue
            sl = (slice(top - y0, bottom - y0), slice(left - x0, right - x0))
            sq = (
                slice(top - q["y"], bottom - q["y"]),
                slice(left - q["x"], right - q["x"]),
            )
            pm = read(root / q["r"]["mask"])
            pi = read(root / q["r"]["image"])
            mismask = int(np.count_nonzero(mask[sl] != pm[sq]))
            misimage = int(
                np.count_nonzero(
                    ~np.isclose(im[sl], pi[sq], rtol=0, atol=0, equal_nan=True)
                )
            )
            overlapchecks.append(
                dict(
                    other=q["r"]["patch"],
                    pixels=(bottom - top) * (right - left),
                    mask_mismatches=mismask,
                    image_mismatches=misimage,
                )
            )
            both_nonzero = int(
                np.count_nonzero((mask[sl] != pm[sq]) & (mask[sl] > 0) & (pm[sq] > 0))
            )
            overlapchecks[-1]["both_nonzero_id_conflicts"] = both_nonzero
            if both_nonzero or misimage:
                raise RuntimeError(
                    "Overlap ID/image conflict " + str(overlapchecks[-1])
                )
            # Preserve the union of original nonzero masks. Count each global pixel once.
            overlap[sl] |= pm[sq] > 0
            del pm, pi
        owned_count = np.zeros(n, np.int64)
        owned_sum = np.zeros(n)
        owned_x = np.zeros(n)
        owned_y = np.zeros(n)
        negative_pixels = 0
        nonfinite_pixels = 0
        vmin = float("inf")
        vmax = float("-inf")
        for y in range(0, h, 128):
            b = mask[y : y + 128]
            v = im[y : y + 128]
            labels = b.ravel().astype(np.int64)
            good = np.isfinite(v)
            seg = b > 0
            nonfinite_pixels += int(np.count_nonzero(seg & ~good))
            negative_pixels += int(np.count_nonzero(seg & (v < 0)))
            if np.any(seg & good):
                sv = v[seg & good]
                vmin = min(vmin, float(sv.min()))
                vmax = max(vmax, float(sv.max()))
            values = np.where(good, v, 0).ravel().astype(np.float64)
            bc = np.bincount(labels, minlength=n)
            bs = np.bincount(labels, weights=values, minlength=n)
            xx = np.broadcast_to(
                np.arange(x0, x0 + w, dtype=float), (b.shape[0], w)
            ).ravel()
            yy = np.repeat(np.arange(y0 + y, y0 + y + b.shape[0], dtype=float), w)
            bx = np.bincount(labels, weights=xx, minlength=n)
            by = np.bincount(labels, weights=yy, minlength=n)
            count += bc
            sums += bs
            xs += bx
            ys += by
            if not good.all():
                bad += np.bincount(labels[~good.ravel()], minlength=n)
            if overlap is not None and overlap[y : y + 128].any():
                ol = overlap[y : y + 128].ravel()
                ll = labels[ol]
                bc = bc - np.bincount(ll, minlength=n)
                bs = bs - np.bincount(ll, weights=values[ol], minlength=n)
                bx = bx - np.bincount(ll, weights=xx[ol], minlength=n)
                by = by - np.bincount(ll, weights=yy[ol], minlength=n)
            owned_count += bc
            owned_sum += bs
            owned_x += bx
            owned_y += by
            if h > 10000 and y % 4096 == 0:
                print("BLOCK", ctx, y, "/", h, flush=True)
        ids = np.flatnonzero(count)
        ids = ids[ids != 0]
        np.savez_compressed(
            dest,
            cell_id=ids,
            pixel_count=count[ids],
            intensity_sum=sums[ids],
            x_sum=xs[ids],
            y_sum=ys[ids],
            nonfinite_pixels=bad[ids],
            owned_pixel_count=owned_count[ids],
            owned_intensity_sum=owned_sum[ids],
            owned_x_sum=owned_x[ids],
            owned_y_sum=owned_y[ids],
        )
        info = dict(
            dataset=dataset,
            split=split,
            context=ctx,
            patch=patch,
            key=key,
            source=r,
            shape=[h, w],
            image_dtype=str(im.dtype),
            mask_dtype=str(mask.dtype),
            offset_y=y0,
            offset_x=x0,
            n_cells=len(ids),
            segmented_pixels=int(count[ids].sum()),
            owned_segmented_pixels=int(owned_count[ids].sum()),
            negative_pixels=negative_pixels,
            nonfinite_pixels=nonfinite_pixels,
            segmented_pixel_min=vmin,
            segmented_pixel_max=vmax,
            overlap_checks=overlapchecks,
            seconds=time.time() - start,
        )
        meta.write_text(json.dumps(info, indent=2))
        audit.append(info)
        previous.append(dict(y=y0, x=x0, h=h, w=w, r=r))
        print(
            "DONE",
            k + 1,
            "/",
            len(rows),
            dataset,
            split,
            ctx,
            patch,
            "cells",
            len(ids),
            "seconds",
            round(time.time() - start, 1),
            flush=True,
        )
        del im, mask, count, sums, xs, ys, bad, owned_count, owned_sum, owned_x, owned_y
    return audit


if __name__ == "__main__":
    try:
        ph = list(
            csv.DictReader(
                (P / "datasets/PhenoBIC-Data/qa/image_mask_label_index.csv").open()
            )
        )
        ph = [r for r in ph if r["image_channel"] == "CD8"]
        audit = extract("PhenoBIC-Data", ph)
        (A / "features/phenobic_extraction_audit.json").write_text(
            json.dumps(audit, indent=2)
        )
        (A / "features/PHENOBIC_EXTRACTION_COMPLETE.json").write_text(
            json.dumps(
                dict(images=len(audit), cells_observed=sum(x["n_cells"] for x in audit))
            )
        )
        print("PHENOBIC EXTRACTION COMPLETE", flush=True)
    except Exception:
        traceback.print_exc()
        raise
