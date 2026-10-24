"""Interpretable instance-mask morphology and CD8 spatial features.
No labels, GMM scores, patient identifiers, or evaluation results enter extraction.
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops

MORPH = [
    "Orientation",
    "Area",
    "ConvexHullArea",
    "MajorAxisLength",
    "MinorAxisLength",
    "Perimeter",
    "Circularity",
    "Eccentricity",
    "EquivalentDiameter",
    "Extent",
    "MinorMajorAxisRatio",
    "Solidity",
] + ["HuMoments" + str(k) for k in range(1, 8)]
HTK_COLUMNS = [
    "Orientation.Orientation",
    "Size.Area",
    "Size.ConvexHullArea",
    "Size.MajorAxisLength",
    "Size.MinorAxisLength",
    "Size.Perimeter",
    "Shape.Circularity",
    "Shape.Eccentricity",
    "Shape.EquivalentDiameter",
    "Shape.Extent",
    "Shape.MinorMajorAxisRatio",
    "Shape.Solidity",
] + ["Shape.HuMoments" + str(k) for k in range(1, 8)]
WHOLE = [
    "whole_mean",
    "whole_median",
    "whole_std",
    "whole_p10",
    "whole_p90",
    "whole_p95",
    "whole_IQR",
    "integrated_intensity",
]
REGIONAL = [
    "edge_mean",
    "edge_median",
    "edge_p90",
    "inner_mean",
    "inner_median",
    "inner_p90",
    "edge_inner_mean_diff",
    "edge_inner_normalized_diff",
]
REFERENCE = [
    "local_reference_median",
    "local_reference_MAD",
    "whole_mean_minus_reference",
    "edge_mean_minus_reference",
    "inner_mean_minus_reference",
    "edge_inner_log2_ratio",
    "edge_signal_enrichment",
]
SECTORS = [
    f"edge_sector_{i:02d}_{kind}"
    for i in range(12)
    for kind in ["mean", "signal_mean", "high_fraction"]
]
ANGULAR = [
    "edge_sector_cv",
    "edge_sector_max_minus_median",
    "sector_signal_cv",
    "sector_signal_max_minus_median",
    "high_signal_edge_coverage",
]
OFFSET = ["signal_centroid_offset_px", "signal_centroid_offset_normalized"]
NEIGHBOR = [
    "neighbor_edge_mean",
    "nonneighbor_edge_mean",
    "neighbor_edge_mean_diff",
    "neighbor_edge_area_fraction",
    "nonneighbor_edge_area_fraction",
]
FEATURES = MORPH + WHOLE + REGIONAL + REFERENCE + SECTORS + ANGULAR + OFFSET + NEIGHBOR
META_STR = [
    "dataset_id",
    "image_id",
    "source_image_id",
    "cell_id_source",
    "split",
    "context",
    "patient_id",
    "batch_id",
    "mask_type",
    "morphometry_backend",
    "qc_flags",
]
QC_NAMES = [
    "truncated",
    "small_whole",
    "small_edge",
    "small_inner",
    "reference_insufficient",
    "reference_mad_zero",
    "no_signal",
    "sector_insufficient",
    "sector_partial",
    "no_neighbors",
    "no_contact_edge",
    "no_noncontact_edge",
    "invalid_pixels",
    "reference_tissue_unknown",
    "context_clipped",
    "disconnected",
    "mask_type_unknown",
]
QC = ["qc_" + x for x in QC_NAMES]
COUNTS = [
    "observed_pixel_count",
    "whole_valid_pixel_count",
    "edge_pixel_count",
    "edge_valid_pixel_count",
    "inner_pixel_count",
    "inner_valid_pixel_count",
    "local_reference_pixel_count",
    "edge_valid_sector_count",
    "signal_valid_sector_count",
    "neighbor_edge_pixel_count",
    "nonneighbor_edge_pixel_count",
] + [f"edge_sector_{i:02d}_pixel_count" for i in range(12)]
META_FLOAT = [
    "centroid_x_px",
    "centroid_y_px",
    "observed_mean_CD8",
    "edge_width_px",
    "distance_max_px",
    "reference_r_inner_px",
    "reference_r_outer_px",
    "neighbor_distance_px",
    "epsilon_intensity",
    "high_signal_k",
    "high_signal_threshold",
    "signal_centroid_x_px",
    "signal_centroid_y_px",
    "pixel_size_um",
    "Area_um2",
    "ConvexHullArea_um2",
    "MajorAxisLength_um",
    "MinorAxisLength_um",
    "Perimeter_um",
    "EquivalentDiameter_um",
    "signal_centroid_offset_um",
]
COLUMNS = META_STR + ["cell_id"] + FEATURES + COUNTS + META_FLOAT + QC


@dataclass(frozen=True)
class FeatureConfig:
    mask_type: str = "unknown"
    edge_fraction: float = 0.2
    edge_min_width: float = 1.0
    edge_width_override: float = None
    r_inner: float = 2.0
    r_outer: float = 5.0
    neighbor_distance: float = 2.0
    min_whole_pixels: int = 5
    min_region_pixels: int = 5
    min_reference_pixels: int = 20
    min_sector_pixels: int = 2
    min_valid_sectors: int = 6
    epsilon: float = 1e-8
    high_signal_k: float = 3.0
    pixel_size_um: float = None
    morphology_backend: str = "skimage"

    def validate(self):
        assert self.mask_type in [
            "nucleus",
            "expanded_nucleus",
            "whole_cell",
            "unknown",
        ]
        assert self.r_outer > self.r_inner >= 0 and self.neighbor_distance > 0
        assert self.edge_fraction > 0 and self.edge_min_width >= 1
        assert self.edge_width_override is None or self.edge_width_override >= 1
        assert (
            self.epsilon > 0
            and self.min_sector_pixels >= 1
            and 1 <= self.min_valid_sectors <= 12
        )
        assert self.pixel_size_um is None or self.pixel_size_um > 0
        assert self.morphology_backend in ["skimage", "histomicstk"]
        return self


def fixed_frame(rows):
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for c in META_STR:
        frame[c] = frame[c].fillna("").astype(str)
    for c in ["cell_id"] + COUNTS:
        frame[c] = frame[c].fillna(0).astype("int64")
    for c in FEATURES + META_FLOAT:
        frame[c] = frame[c].astype("float64")
    for c in QC:
        frame[c] = frame[c].fillna(False).astype(bool)
    return frame


def morphometry(binary, backend="skimage"):
    props = regionprops(binary.astype(np.uint8), cache=True)
    if not props:
        return {name: np.nan for name in MORPH}
    rp = props[0]
    if backend == "histomicstk":
        from histomicstk.features import compute_morphometry_features

        result = compute_morphometry_features(binary.astype(np.uint8), rprops=props)
        assert all(c in result.columns for c in HTK_COLUMNS)
        values = result.loc[:, HTK_COLUMNS].iloc[0].to_numpy(dtype=float)
        out = dict(zip(MORPH, values))
    else:
        perimeter = rp.perimeter
        major = rp.major_axis_length
        vals = [
            rp.orientation,
            rp.area,
            rp.convex_area,
            major,
            rp.minor_axis_length,
            perimeter,
            4 * np.pi * rp.area / perimeter**2 if perimeter > 0 else np.nan,
            rp.eccentricity,
            rp.equivalent_diameter,
            rp.extent,
            rp.minor_axis_length / major if major > 0 else np.nan,
            rp.solidity,
        ] + list(rp.moments_hu)
        out = dict(zip(MORPH, map(float, vals)))
    # Degenerate geometric denominators are invalid, not fictitious finite shapes.
    if rp.perimeter <= 0:
        out["Circularity"] = np.nan
    if rp.major_axis_length <= 0:
        out["MinorMajorAxisRatio"] = np.nan
    return out


def extract_roi(
    image,
    labels,
    cell_id,
    metadata=None,
    config=None,
    valid_mask=None,
    tissue_mask=None,
    origin=(0, 0),
    force_truncated=False,
    return_regions=False,
):
    """Extract one instance from a sufficiently padded ROI; origin is global (y,x).
    All neighboring instance IDs must remain in labels. Cell ID is explicit.
    valid_mask denotes observed/finite spatial support, tissue_mask is optional.
    """
    cfg = (config or FeatureConfig()).validate()
    im = np.asarray(image)
    lab = np.asarray(labels)
    if im.ndim != 2 or lab.ndim != 2 or im.shape != lab.shape:
        raise ValueError("Image/mask must have identical 2D coordinates and shape")
    if not np.issubdtype(lab.dtype, np.integer) or np.any(lab < 0):
        raise ValueError("Mask must contain nonnegative integer instance IDs")
    if cell_id <= 0:
        raise ValueError("Background ID 0 is not a cell")
    M = lab == int(cell_id)
    if not M.any():
        raise KeyError("Requested instance ID absent from ROI")
    valid = (
        np.ones(im.shape, bool) if valid_mask is None else np.asarray(valid_mask, bool)
    )
    if valid.shape != im.shape:
        raise ValueError("Invalid spatial-support dimensions")
    tissue = (
        np.ones(im.shape, bool)
        if tissue_mask is None
        else np.asarray(tissue_mask, bool)
    )
    if tissue.shape != im.shape:
        raise ValueError("Tissue mask dimensions mismatch")
    finite = valid & np.isfinite(im)
    out = {c: np.nan for c in FEATURES + META_FLOAT}
    out.update({c: 0 for c in COUNTS})
    out.update({c: False for c in QC})
    out.update({c: "" for c in META_STR})
    out.update(metadata or {})
    out.update(
        cell_id=int(cell_id),
        mask_type=cfg.mask_type,
        morphometry_backend=cfg.morphology_backend,
    )
    # Keep metadata restricted to documented columns; labels/GMM keys are rejected.
    if metadata and set(metadata) - set(META_STR):
        raise ValueError("Unsupported metadata, including any outcome labels")
    flags = set()
    if cfg.mask_type == "unknown":
        flags.add("mask_type_unknown")
    if tissue_mask is None:
        flags.add("reference_tissue_unknown")
    if np.any(M & ~finite):
        flags.add("invalid_pixels")
    if not valid.all():
        flags.add("context_clipped")
    n = int(M.sum())
    yy, xx = np.nonzero(M)
    cy = float(yy.mean())
    cx = float(xx.mean())
    out.update(
        observed_pixel_count=n,
        centroid_y_px=cy + origin[0],
        centroid_x_px=cx + origin[1],
        reference_r_inner_px=cfg.r_inner,
        reference_r_outer_px=cfg.r_outer,
        neighbor_distance_px=cfg.neighbor_distance,
        epsilon_intensity=cfg.epsilon,
        high_signal_k=cfg.high_signal_k,
        pixel_size_um=cfg.pixel_size_um if cfg.pixel_size_um is not None else np.nan,
    )
    truncated = (
        force_truncated
        or M[0].any()
        or M[-1].any()
        or M[:, 0].any()
        or M[:, -1].any()
        or np.any(M & ndi.binary_dilation(~valid, structure=np.ones((3, 3), bool)))
    )
    if truncated:
        flags.add("truncated")
    disconnected = ndi.label(M, structure=np.ones((3, 3), int))[1] > 1
    if disconnected:
        flags.add("disconnected")
    # Padding is essential even when the instance fills its crop.
    distance = ndi.distance_transform_edt(np.pad(M, 1))[1:-1, 1:-1]
    dmax = float(distance.max())
    width = (
        cfg.edge_width_override
        if cfg.edge_width_override is not None
        else max(cfg.edge_min_width, cfg.edge_fraction * dmax)
    )
    edge = M & (distance <= width)
    inner = M & (distance > width)
    outside_distance = ndi.distance_transform_edt(~M)
    reference = (
        (outside_distance > cfg.r_inner)
        & (outside_distance <= cfg.r_outer)
        & (lab == 0)
        & finite
        & tissue
    )
    evalid = edge & finite
    ivalid = inner & finite
    wvalid = M & finite
    v = im[wvalid].astype(float)
    out.update(
        whole_valid_pixel_count=len(v),
        edge_pixel_count=int(edge.sum()),
        inner_pixel_count=int(inner.sum()),
        edge_valid_pixel_count=int(evalid.sum()),
        inner_valid_pixel_count=int(ivalid.sum()),
        local_reference_pixel_count=int(reference.sum()),
        edge_width_px=width,
        distance_max_px=dmax,
        observed_mean_CD8=float(v.mean()) if len(v) else np.nan,
    )
    if len(v) < cfg.min_whole_pixels:
        flags.add("small_whole")
    if evalid.sum() < cfg.min_region_pixels:
        flags.add("small_edge")
    if ivalid.sum() < cfg.min_region_pixels:
        flags.add("small_inner")
    if reference.sum() < cfg.min_reference_pixels:
        flags.add("reference_insufficient")
    geometry_valid = not (truncated or disconnected or "small_whole" in flags)
    whole_valid = geometry_valid and "invalid_pixels" not in flags
    regional_valid = (
        whole_valid and "small_edge" not in flags and "small_inner" not in flags
    )
    if geometry_valid:
        out.update(morphometry(M, cfg.morphology_backend))
    if whole_valid:
        q = np.percentile(v, [10, 25, 50, 75, 90, 95])
        out.update(
            whole_mean=float(v.mean()),
            whole_median=q[2],
            whole_std=float(v.std(ddof=0)),
            whole_p10=q[0],
            whole_p90=q[4],
            whole_p95=q[5],
            whole_IQR=q[3] - q[1],
            integrated_intensity=float(v.sum(dtype=np.float64)),
        )
    for name, region in [("edge", evalid), ("inner", ivalid)]:
        if whole_valid and region.sum() >= cfg.min_region_pixels:
            vals = im[region].astype(float)
            out[name + "_mean"] = float(vals.mean())
            out[name + "_median"] = float(np.median(vals))
            out[name + "_p90"] = float(np.percentile(vals, 90))
    if regional_valid:
        diff = out["edge_mean"] - out["inner_mean"]
        den = abs(out["edge_mean"]) + abs(out["inner_mean"])
        out["edge_inner_mean_diff"] = diff
        if den > cfg.epsilon:
            out["edge_inner_normalized_diff"] = diff / (den + cfg.epsilon)
        else:
            flags.add("no_signal")
    ref_ok = reference.sum() >= cfg.min_reference_pixels
    S = np.full(im.shape, np.nan, dtype=float)
    if ref_ok:
        ref = im[reference].astype(float)
        median = float(np.median(ref))
        mad = float(np.median(np.abs(ref - median)))
        out.update(local_reference_median=median, local_reference_MAD=mad)
        if mad <= cfg.epsilon:
            flags.add("reference_mad_zero")
        S[finite] = np.maximum(im[finite].astype(float) - median, 0)
        for name in ["whole", "edge", "inner"]:
            if np.isfinite(out[name + "_mean"]):
                out[name + "_mean_minus_reference"] = out[name + "_mean"] - median
        total = float(np.sum(S[wvalid], dtype=np.float64))
        if whole_valid and total > cfg.epsilon * n:
            if regional_valid:
                se = float(S[evalid].mean())
                si = float(S[ivalid].mean())
                out["edge_inner_log2_ratio"] = float(
                    np.log2((se + cfg.epsilon) / (si + cfg.epsilon))
                )
                out["edge_signal_enrichment"] = (float(S[evalid].sum()) / total) / (
                    edge.sum() / n
                )
            wy, wx = np.nonzero(wvalid)
            weights = S[wvalid]
            scy = float(np.dot(wy, weights) / total)
            scx = float(np.dot(wx, weights) / total)
            offset = float(np.hypot(scy - cy, scx - cx))
            out.update(
                signal_centroid_y_px=scy + origin[0],
                signal_centroid_x_px=scx + origin[1],
                signal_centroid_offset_px=offset,
                signal_centroid_offset_normalized=offset / np.sqrt(n / np.pi),
            )
        else:
            flags.add("no_signal")
    # Angle zero points right; angles increase counter-clockwise in Cartesian x-right/y-up coordinates.
    Y, X = np.indices(im.shape)
    angles = np.mod(np.arctan2(-(Y - cy), X - cx), 2 * np.pi)
    sector = np.minimum((angles / (2 * np.pi) * 12).astype(int), 11)
    rawmeans = []
    signalmeans = []
    threshold_ok = ref_ok and out["local_reference_MAD"] > cfg.epsilon and whole_valid
    if threshold_ok:
        T = (
            out["local_reference_median"]
            + cfg.high_signal_k * 1.4826 * out["local_reference_MAD"]
        )
        out["high_signal_threshold"] = T
        if evalid.sum() >= cfg.min_region_pixels:
            out["high_signal_edge_coverage"] = float(np.mean(im[evalid] > T))
    for k in range(12):
        region = evalid & (sector == k)
        num = int(region.sum())
        out[f"edge_sector_{k:02d}_pixel_count"] = num
        if whole_valid and num >= cfg.min_sector_pixels:
            value = float(im[region].mean(dtype=np.float64))
            out[f"edge_sector_{k:02d}_mean"] = value
            rawmeans.append(value)
            if ref_ok:
                value = float(S[region].mean())
                out[f"edge_sector_{k:02d}_signal_mean"] = value
                signalmeans.append(value)
            if threshold_ok:
                out[f"edge_sector_{k:02d}_high_fraction"] = float(
                    np.mean(im[region] > T)
                )
    out["edge_valid_sector_count"] = len(rawmeans)
    out["signal_valid_sector_count"] = len(signalmeans)
    if len(rawmeans) < 12:
        flags.add("sector_partial")
    if len(rawmeans) < cfg.min_valid_sectors:
        flags.add("sector_insufficient")
    for values, cvname, diffname in [
        (rawmeans, "edge_sector_cv", "edge_sector_max_minus_median"),
        (signalmeans, "sector_signal_cv", "sector_signal_max_minus_median"),
    ]:
        if len(values) >= cfg.min_valid_sectors:
            values = np.array(values)
            den = float(np.mean(np.abs(values)))
            if den > cfg.epsilon:
                out[cvname] = float(values.std(ddof=0) / den)
                out[diffname] = float(values.max() - np.median(values))
            else:
                flags.add("no_signal")
    others = (lab > 0) & ~M
    if others.any():
        near = edge & (ndi.distance_transform_edt(~others) <= cfg.neighbor_distance)
    else:
        near = np.zeros(M.shape, bool)
        flags.add("no_neighbors")
    far = edge & ~near
    nv = near & finite
    fv = far & finite
    out.update(
        neighbor_edge_pixel_count=int(nv.sum()),
        nonneighbor_edge_pixel_count=int(fv.sum()),
    )
    if whole_valid and edge.sum():
        out["neighbor_edge_area_fraction"] = float(near.sum() / edge.sum())
        out["nonneighbor_edge_area_fraction"] = float(far.sum() / edge.sum())
        if nv.sum() >= cfg.min_region_pixels:
            out["neighbor_edge_mean"] = float(im[nv].mean(dtype=np.float64))
        else:
            flags.add("no_contact_edge")
        if fv.sum() >= cfg.min_region_pixels:
            out["nonneighbor_edge_mean"] = float(im[fv].mean(dtype=np.float64))
        else:
            flags.add("no_noncontact_edge")
        if np.isfinite(out["neighbor_edge_mean"]) and np.isfinite(
            out["nonneighbor_edge_mean"]
        ):
            out["neighbor_edge_mean_diff"] = (
                out["neighbor_edge_mean"] - out["nonneighbor_edge_mean"]
            )
    if cfg.pixel_size_um is not None:
        s = cfg.pixel_size_um
        for name in ["Area", "ConvexHullArea"]:
            out[name + "_um2"] = out[name] * s * s
        for name in [
            "MajorAxisLength",
            "MinorAxisLength",
            "Perimeter",
            "EquivalentDiameter",
        ]:
            out[name + "_um"] = out[name] * s
        out["signal_centroid_offset_um"] = out["signal_centroid_offset_px"] * s
    for name in flags:
        out["qc_" + name] = True
    out["qc_flags"] = ";".join(sorted(flags))
    for key in FEATURES + META_FLOAT:
        if out[key] is not None and not np.isfinite(out[key]):
            out[key] = np.nan
    if return_regions:
        return out, dict(
            whole=M,
            edge=edge,
            inner=inner,
            reference=reference,
            neighbor_edge=near,
            sector=sector,
            signal=S,
            valid=finite,
        )
    return out


def extract_image(
    image,
    labels,
    dataset_id,
    image_id,
    config=None,
    valid_mask=None,
    tissue_mask=None,
    metadata=None,
):
    """Reusable small-image API. Bounding boxes are found once, not per cell."""
    cfg = (config or FeatureConfig()).validate()
    im = np.asarray(image)
    lab = np.asarray(labels)
    if im.ndim != 2 or im.shape != lab.shape:
        raise ValueError("Image/mask shape mismatch")
    if not np.issubdtype(lab.dtype, np.integer) or lab.min() < 0:
        raise ValueError("Integer instance mask required")
    if not lab.any():
        return fixed_frame([])
    boxes = ndi.find_objects(lab)
    halo = int(np.ceil(max(cfg.r_outer, cfg.neighbor_distance))) + 2
    rows = []
    for cid, box in enumerate(boxes, 1):
        if box is None:
            continue
        y1 = max(0, box[0].start - halo)
        y2 = min(im.shape[0], box[0].stop + halo)
        x1 = max(0, box[1].start - halo)
        x2 = min(im.shape[1], box[1].stop + halo)
        sl = (slice(y1, y2), slice(x1, x2))
        info = dict(
            dataset_id=dataset_id,
            image_id=image_id,
            cell_id_source="integer_instance_mask",
        )
        info.update(metadata or {})
        rows.append(
            extract_roi(
                im[sl],
                lab[sl],
                cid,
                info,
                cfg,
                None if valid_mask is None else valid_mask[sl],
                None if tissue_mask is None else tissue_mask[sl],
                origin=(y1, x1),
            )
        )
    return fixed_frame(rows)
