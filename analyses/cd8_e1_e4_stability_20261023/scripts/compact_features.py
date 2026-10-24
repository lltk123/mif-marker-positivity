"""Pixel normalization and fixed 9 localization descriptors. No outcome inputs."""
import numpy as np
from scipy import ndimage as ndi
from common import I8, S9

EPS = 1e-8
QC = [
    "mask_missing",
    "truncated",
    "disconnected",
    "small_whole",
    "small_edge",
    "small_inner",
    "invalid_pixels",
    "reference_insufficient",
    "reference_mad_zero",
    "no_corrected_whole_signal",
    "no_corrected_edge_signal",
    "no_contact_edge",
    "no_noncontact_edge",
    "neighbor_band_insufficient",
    "reference_tissue_unknown",
    "context_clipped",
    "calibration_invalid",
]
COUNTS = [
    "observed_pixel_count",
    "finite_pixel_count",
    "edge_pixel_count",
    "inner_pixel_count",
    "reference_pixel_count",
    "contact_pixel_count",
    "noncontact_pixel_count",
    "neighbor_band_pixel_count",
]
EXTRA = [
    "edge_width_px",
    "reference_median",
    "reference_MAD",
    "observed_raw_mean",
    "signal_edge_total",
    "signal_whole_total",
]


def calibration_valid(L, U):
    scale = max(abs(float(L)), abs(float(U)), np.finfo(np.float64).tiny)
    return bool(
        np.isfinite(L)
        and np.isfinite(U)
        and U - L > 32 * np.finfo(np.float64).eps * scale
    )


def normalize(image, L, U, valid=None):
    x = np.asarray(image, np.float64)
    v = np.isfinite(x) if valid is None else np.asarray(valid, bool) & np.isfinite(x)
    out = np.full(x.shape, np.nan)
    if calibration_valid(L, U):
        out[v] = np.clip((x[v] - L) / (U - L), 0, 1)
    return out


def intensity8(v):
    v = np.asarray(v, np.float64)
    q = np.percentile(v, [10, 25, 50, 75, 90, 95])
    return dict(
        zip(
            I8,
            [
                float(v.mean()),
                float(q[2]),
                float(v.std(ddof=0)),
                float(q[0]),
                float(q[4]),
                float(q[5]),
                float(q[3] - q[1]),
                float(v.sum(dtype=np.float64)),
            ],
        )
    )


def normalized_difference(a, b):
    den = abs(a) + abs(b)
    return (a - b) / (den + EPS) if np.isfinite(den) and den > EPS else np.nan


def extract_compact(
    image, labels, cell_id, L, U, valid=None, force_truncated=False, tissue=None
):
    im = np.asarray(image)
    lab = np.asarray(labels)
    if (
        im.ndim != 2
        or lab.shape != im.shape
        or not np.issubdtype(lab.dtype, np.integer)
        or (lab < 0).any()
    ):
        raise ValueError("Aligned 2D image/integer instances required")
    valid = np.ones(im.shape, bool) if valid is None else np.asarray(valid, bool)
    if valid.shape != im.shape:
        raise ValueError("Spatial support shape mismatch")
    if tissue is not None and np.shape(tissue) != im.shape:
        raise ValueError("Tissue shape mismatch")
    r = {k: np.nan for k in I8 + S9 + EXTRA}
    r.update({k: 0 for k in COUNTS})
    r.update({"qc_" + k: False for k in QC})
    r["intensity_input_valid"] = False

    def flag(x):
        r["qc_" + x] = True

    if tissue is None:
        flag("reference_tissue_unknown")
    if not valid.all():
        flag("context_clipped")
    if not calibration_valid(L, U):
        flag("calibration_invalid")
    M = lab == int(cell_id)
    if cell_id <= 0 or not M.any():
        flag("mask_missing")
        return r
    fin = valid & np.isfinite(im)
    n = int(M.sum())
    v = im[M & fin].astype(np.float64)
    r.update(
        observed_pixel_count=n,
        finite_pixel_count=len(v),
        observed_raw_mean=float(v.mean()) if len(v) else np.nan,
    )
    if len(v) < 5:
        flag("small_whole")
    if (M & ~fin).any():
        flag("invalid_pixels")
    truncated = (
        force_truncated
        or M[0].any()
        or M[-1].any()
        or M[:, 0].any()
        or M[:, -1].any()
        or (M & ndi.binary_dilation(~valid, structure=np.ones((3, 3), bool))).any()
    )
    if truncated:
        flag("truncated")
    disconnected = ndi.label(M, structure=np.ones((3, 3), int))[1] > 1
    if disconnected:
        flag("disconnected")
    geom_ok = not (truncated or disconnected or len(v) < 5 or r["qc_invalid_pixels"])
    d = ndi.distance_transform_edt(np.pad(M, 1))[1:-1, 1:-1]
    w = max(1, 0.2 * d.max())
    edge = M & (d <= w)
    inner = M & (d > w)
    r.update(
        edge_width_px=float(w),
        edge_pixel_count=int((edge & fin).sum()),
        inner_pixel_count=int((inner & fin).sum()),
    )
    if r["edge_pixel_count"] < 5:
        flag("small_edge")
    if r["inner_pixel_count"] < 5:
        flag("small_inner")
    dt = ndi.distance_transform_edt(~M)
    reference = (dt > 2) & (dt <= 5) & (lab == 0) & fin
    if tissue is not None:
        reference &= np.asarray(tissue, bool)
    r["reference_pixel_count"] = int(reference.sum())
    if reference.sum() < 20:
        flag("reference_insufficient")
    others = (lab > 0) & ~M
    near = (
        edge & (ndi.distance_transform_edt(~others) <= 2)
        if others.any()
        else np.zeros(M.shape, bool)
    )
    far = edge & ~near
    nv = near & fin
    fv = far & fin
    band = others & (dt <= 2) & fin
    r.update(
        contact_pixel_count=int(nv.sum()),
        noncontact_pixel_count=int(fv.sum()),
        neighbor_band_pixel_count=int(band.sum()),
    )
    if nv.sum() < 5:
        flag("no_contact_edge")
    if fv.sum() < 5:
        flag("no_noncontact_edge")
    if band.sum() < 5:
        flag("neighbor_band_insufficient")
    if r["qc_calibration_invalid"]:
        return r
    x = normalize(im, L, U, valid)
    r["intensity_input_valid"] = bool(geom_ok)
    if not geom_ok:
        return r
    r.update(intensity8(x[M & fin]))
    e = edge & fin
    inside = inner & fin
    if e.sum() >= 5 and inside.sum() >= 5:
        r["edge_inner_normalized_diff"] = normalized_difference(
            float(x[e].mean()), float(x[inside].mean())
        )
    r["neighbor_edge_area_fraction"] = (
        float(near.sum() / edge.sum()) if edge.sum() else np.nan
    )
    if nv.sum() >= 5 and fv.sum() >= 5:
        r["contact_noncontact_normalized_diff"] = normalized_difference(
            float(x[nv].mean()), float(x[fv].mean())
        )
    if nv.sum() >= 5 and band.sum() >= 5:
        r["contact_neighbor_band_normalized_diff"] = normalized_difference(
            float(x[nv].mean()), float(x[band].mean())
        )
    if reference.sum() >= 20:
        vals = x[reference]
        b = float(np.median(vals))
        mad = float(np.median(abs(vals - b)))
        r.update(reference_median=b, reference_MAD=mad)
        if mad <= EPS:
            flag("reference_mad_zero")
        elif e.sum() >= 5:
            r["high_signal_edge_coverage"] = float(
                np.mean(x[e] > (b + 3 * 1.4826 * mad))
            )
        signal = np.maximum(x - b, 0)
        total = float(signal[M & fin].sum(dtype=np.float64))
        etotal = float(signal[e].sum(dtype=np.float64))
        r.update(signal_edge_total=etotal, signal_whole_total=total)
        yy, xx = np.nonzero(M)
        cy = float(yy.mean())
        cx = float(xx.mean())
        if total > EPS * n:
            wy, wx = np.nonzero(M & fin)
            weights = signal[M & fin]
            sy = np.dot(wy, weights) / total
            sx = np.dot(wx, weights) / total
            r["signal_centroid_offset_normalized"] = float(
                np.hypot(sy - cy, sx - cx) / np.sqrt(n / np.pi)
            )
            if e.sum() >= 5 and inside.sum() >= 5:
                r["edge_signal_enrichment"] = (etotal / total) / (edge.sum() / n)
        else:
            flag("no_corrected_whole_signal")
        # Rk follows the specified sum over individual edge pixels, with no phase feature.
        if etotal > 0 and e.sum() >= 5:
            ey, ex = np.nonzero(e)
            theta = np.arctan2(-(ey - cy), ex - cx)
            weights = signal[e]
            for k in [1, 2]:
                r["angular_R" + str(k)] = float(
                    min(1.0, abs(np.sum(weights * np.exp(1j * k * theta))) / etotal)
                )
        else:
            flag("no_corrected_edge_signal")
    for k in I8 + S9 + EXTRA:
        if not np.isfinite(r[k]):
            r[k] = np.nan
    return r


def perturb_mask(labels, kind, valid=None):
    lab = np.asarray(labels)
    v = np.ones(lab.shape, bool) if valid is None else np.asarray(valid, bool)
    p = np.pad(lab, 1)
    neighbors = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    if kind == "erode_1":
        out = np.where((lab > 0) & np.all(neighbors == lab[None], axis=0), lab, 0)
    elif kind == "dilate_1":
        maximum = neighbors.max(axis=0)
        minimum = np.where(neighbors > 0, neighbors, np.iinfo(lab.dtype).max).min(
            axis=0
        )
        unique = (maximum > 0) & (minimum == maximum) & (lab == 0) & v
        out = lab.copy()
        out[unique] = maximum[unique]
    else:
        raise ValueError(kind)
    out[~v] = 0
    return out
