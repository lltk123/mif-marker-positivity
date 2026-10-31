"""Spot-check full-image preparation against original pixels and global resize coordinates."""
from common import *
from prepare_data import raw_unit
from normalization import bounds_for, normalize_pixels


def main():
    chosen = {}
    for image, es in units().items():
        group = (
            es[0]["split"]
            if es[0]["split"] != "external_val_TMA_cores"
            else es[0]["context"]
        )
        chosen.setdefault(group, (image, es[0]))
    rows = []
    for group, (image, e) in chosen.items():
        bound = bounds_for(image)
        if bound["calibration_invalid"]:
            continue
        a, m, v = raw_unit(image)
        h, w = m.shape
        hs, ws = h // 2, w // 2
        inp = cached(image, "input")
        ids = cached(image, "ids_scaled")
        norm = cached(image, "normalized_full")
        binary = cached(image, "binary_full")
        z = np.load(OLD / "bboxes" / (e["key"] + ".npz"))
        boxes = z["bbox"]
        for k in np.unique(np.linspace(0, len(boxes) - 1, 4).astype(int)):
            y1, y2, x1, x2 = boxes[k]
            cy = (float(y1 + y2) / 2 + e["offset_y"]) * hs / h
            cx = (float(x1 + x2) / 2 + e["offset_x"]) * ws / w
            yy = np.arange(max(0, int(cy) - 8), min(hs, int(cy) + 9))
            xx = np.arange(max(0, int(cx) - 8), min(ws, int(cx) + 9))
            ny = np.floor(yy * h / hs).astype(int)
            nx = np.floor(xx * w / ws).astype(int)
            center = m[ny[:, None], nx[None, :]]
            raw_support = None if v is None else v[ny[:, None], nx[None, :]]
            expected_raw = normalize_pixels(
                a[ny[:, None], nx[None, :]], bound["L"], bound["U"], raw_support
            ).astype(np.float32)
            actual_norm = norm[ny[:, None], nx[None, :]]
            assert np.array_equal(np.isfinite(expected_raw), np.isfinite(actual_norm))
            rawerr = float(np.nanmax(np.abs(expected_raw - actual_norm)))
            neigh = [
                m[
                    np.clip(ny + dy, 0, h - 1)[:, None],
                    np.clip(nx + dx, 0, w - 1)[None, :],
                ]
                for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            ]
            expected_binary = (center > 0) & ~np.logical_or.reduce(
                [n != center for n in neigh]
            )
            assert np.array_equal(expected_binary, binary[ny[:, None], nx[None, :]])
            assert np.array_equal(center, ids[yy[:, None], xx[None, :]])
            assert np.array_equal(expected_binary, inp[0, 1][yy[:, None], xx[None, :]])
            fy = (yy + 0.5) * h / hs - 0.5
            fx = (xx + 0.5) * w / ws - 0.5
            sy = np.floor(fy).astype(int)
            sx = np.floor(fx).astype(int)
            dy = (fy - sy).astype(np.float32)
            dx = (fx - sx).astype(np.float32)
            y0 = np.clip(sy, 0, h - 1)
            y1 = np.clip(sy + 1, 0, h - 1)
            x0 = np.clip(sx, 0, w - 1)
            x1 = np.clip(sx + 1, 0, w - 1)
            top = (
                np.nan_to_num(norm[y0[:, None], x0[None, :]], nan=0) * (1 - dx)[None, :]
                + np.nan_to_num(norm[y0[:, None], x1[None, :]], nan=0) * dx[None, :]
            )
            bot = (
                np.nan_to_num(norm[y1[:, None], x0[None, :]], nan=0) * (1 - dx)[None, :]
                + np.nan_to_num(norm[y1[:, None], x1[None, :]], nan=0) * dx[None, :]
            )
            expected = top * (1 - dy)[:, None] + bot * dy[:, None]
            interp = float(
                np.max(np.abs(expected - inp[0, 0][yy[:, None], xx[None, :]]))
            )
            assert rawerr == 0 and interp < 3e-6
            rows.append(
                dict(
                    group=group,
                    image_id=image,
                    anchor_cell_id=int(z["cell_id"][k]),
                    n_checked_pixels=len(yy) * len(xx),
                    raw_normalization_max_error=rawerr,
                    global_linear_resize_max_error=interp,
                    instance_nearest_exact=True,
                    binary_boundary_exact=True,
                    passed=True,
                )
            )
    pd.DataFrame(rows).to_csv(
        R / "tests/global_preparation_pixel_checks.csv", index=False
    )
    print(
        "GLOBAL_PREPARATION_PIXEL_CHECKS_PASSED",
        len(rows),
        max(r["global_linear_resize_max_error"] for r in rows),
        flush=True,
    )


if __name__ == "__main__":
    main()
