"""Shared, label-free E1_outer_P1_P99 pixel calibration.

Bounds use equally weighted original instances, never sampled cells or pixels
pooled across units. The public normalize_pixels API is shared with Nimbus.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

METHOD_ID = "E1_outer_P1_P99"
KEYS = ["dataset_id", "image_id", "cell_id", "marker"]
KEYS3 = KEYS[:3]
EXPECTED_ROWS = 1817783
EXPECTED_UNITS = 48


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError("Refusing incompatible existing artifact: " + str(path))
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def key_hash(frame):
    if frame[KEYS].isnull().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Null or duplicate explicit cell keys")
    ordered = frame[KEYS].sort_values(KEYS).copy()
    ordered["cell_id"] = ordered.cell_id.astype("int64")
    return hashlib.sha256(
        ordered.to_csv(index=False).replace("\r\n", "\n").encode()
    ).hexdigest()


def calibration_valid(L, U):
    floor = (
        32 * np.finfo(np.float64).eps * max(abs(L), abs(U), np.finfo(np.float64).tiny)
    )
    return bool(np.isfinite(L) and np.isfinite(U) and U - L > floor)


def normalize_pixels(image, L, U, valid_mask=None):
    """Map supported finite pixels in float64; preserve invalid support as NaN."""
    a = np.asarray(image, dtype=np.float64)
    valid = np.isfinite(a)
    if valid_mask is not None:
        if np.shape(valid_mask) != a.shape:
            raise ValueError("valid_mask and image shapes differ")
        valid &= np.asarray(valid_mask, dtype=bool)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if calibration_valid(L, U):
        out[valid] = np.clip((a[valid] - L) / (U - L), 0.0, 1.0)
    return out


def fit_unit_bounds(cells):
    """Return per-unit P1(cell P10)/P99(cell P90) and interpolation witnesses."""
    if cells.duplicated(KEYS3).any():
        raise ValueError("Duplicate instance keys in raw percentiles")
    params, witnesses = [], []
    for (dataset, uid), group in cells.groupby(["dataset_id", "image_id"], sort=True):
        valid = group.finite_pixel_count >= 5
        finite_quantiles = np.isfinite(group.raw_cell_p10) & np.isfinite(
            group.raw_cell_p90
        )
        if (valid & ~finite_quantiles).any():
            raise ValueError("Eligible instance lacks recovered raw quantiles: " + uid)
        eligible = group.loc[valid & finite_quantiles]
        L, U = np.nan, np.nan
        if len(eligible):
            L = float(np.percentile(eligible.raw_cell_p10.to_numpy(), 1))
            U = float(np.percentile(eligible.raw_cell_p90.to_numpy(), 99))
            for name, column, q in [
                ("L", "raw_cell_p10", 0.01),
                ("U", "raw_cell_p90", 0.99),
            ]:
                ordered = eligible.sort_values([column, "cell_id"])
                h = (len(ordered) - 1) * q
                for rank in sorted(set([int(np.floor(h)), int(np.ceil(h))])):
                    row = ordered.iloc[rank]
                    witnesses.append(
                        dict(
                            dataset_id=dataset,
                            image_id=uid,
                            bound=name,
                            zero_based_rank=rank,
                            cell_id=int(row.cell_id),
                            value=float(row[column]),
                            interpolation_weight=float(1 - abs(h - rank)),
                        )
                    )
        params.append(
            dict(
                dataset_id=dataset,
                image_id=uid,
                split=str(group.split.iloc[0]),
                context=str(group.context.iloc[0]),
                L=L,
                U=U,
                calibration_invalid=not calibration_valid(L, U),
                n_eligible=len(eligible),
                n_instances=len(group),
                finite_instance_pixels=int(group.finite_pixel_count.sum()),
                raw_cell_p10_min=float(eligible.raw_cell_p10.min())
                if len(eligible)
                else np.nan,
                raw_cell_p90_max=float(eligible.raw_cell_p90.max())
                if len(eligible)
                else np.nan,
                low_tail_instances=int((eligible.raw_cell_p10 < L).sum()),
                high_tail_instances=int((eligible.raw_cell_p90 > U).sum()),
                percentile_method="linear",
                instance_weighting="equal",
            )
        )
    return pd.DataFrame(params), pd.DataFrame(witnesses)
