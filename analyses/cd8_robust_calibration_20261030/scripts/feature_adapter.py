"""Reuse unchanged legacy feature code with explicit-key morphology caching.

The cache substitutes only the pure morphometry call in a private function globals
copy. All geometry QC, intensity and spatial calculations remain original bytecode.
It never modifies a legacy module, source file, or process-wide global binding.
"""
import importlib.util
import sys
import types
from pathlib import Path
import numpy as np
from calibration_core import normalize_pixels, calibration_valid, METHOD_ID, KEYS3


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_legacy(legacy_dir):
    return load_module(
        "robust_legacy_mif_features", Path(legacy_dir) / "scripts/mif_features.py"
    )


def load_reader(legacy_dir, legacy):
    previous = sys.modules.get("mif_features")
    sys.modules["mif_features"] = legacy
    try:
        return load_module(
            "robust_legacy_batch_features",
            Path(legacy_dir) / "scripts/batch_features.py",
        )
    finally:
        if previous is None:
            sys.modules.pop("mif_features", None)
        else:
            sys.modules["mif_features"] = previous


class MorphologyCache:
    """Immutable original morphology bound to explicit keys and source file hash."""

    def __init__(self, frame, columns, source_sha256):
        if not source_sha256 or frame.duplicated(KEYS3).any():
            raise ValueError("Morphology cache requires unique keys and source hash")
        self.frame = frame.set_index(KEYS3).sort_index()
        self.columns = list(columns)
        self.source_sha256 = source_sha256

    def lookup(self, metadata, cell_id):
        key = (metadata["dataset_id"], metadata["image_id"], int(cell_id))
        values = self.frame.loc[key, self.columns]
        return dict(zip(self.columns, values.to_numpy(dtype=float)))


def extract_normalized_roi(
    legacy,
    image,
    labels,
    cell_id,
    metadata,
    config,
    L,
    U,
    valid_mask=None,
    origin=(0, 0),
    force_truncated=False,
    morphology_cache=None,
):
    """Re-extract pixels; optional cache changes only the 19 geometric outputs."""
    valid = (
        np.ones(np.shape(image), bool)
        if valid_mask is None
        else np.asarray(valid_mask, bool)
    )
    raw = np.asarray(image)
    selected = (labels == int(cell_id)) & valid & np.isfinite(raw)
    values = raw[selected].astype(np.float64)
    normalized = normalize_pixels(raw, L, U, valid)
    function = legacy.extract_roi
    cached = None
    if morphology_cache is not None:
        cached = morphology_cache.lookup(metadata, cell_id)
        globals_copy = dict(function.__globals__)
        globals_copy["morphometry"] = lambda *args, **kwargs: dict(cached)
        function = types.FunctionType(
            function.__code__,
            globals_copy,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
    result = function(
        normalized,
        labels,
        cell_id,
        metadata,
        config,
        valid_mask=valid,
        origin=origin,
        force_truncated=force_truncated,
    )
    if cached is None and not calibration_valid(L, U):
        raw_result = legacy.extract_roi(
            raw,
            labels,
            cell_id,
            metadata,
            config,
            valid_mask=valid,
            origin=origin,
            force_truncated=force_truncated,
        )
        cached = {name: raw_result[name] for name in legacy.MORPH}
    if cached is not None:
        # Geometry does not depend on whether the calibration unit is valid.
        result.update(cached)
    result.update(
        marker="CD8",
        calibration_valid=calibration_valid(L, U),
        L=float(L),
        U=float(U),
        normalization_method=METHOD_ID,
        raw_observed_mean_CD8=float(values.mean()) if len(values) else np.nan,
        raw_finite_pixel_count=len(values),
        raw_observed_min_CD8=float(values.min()) if len(values) else np.nan,
        raw_observed_max_CD8=float(values.max()) if len(values) else np.nan,
        raw_below_L_count=int((values < L).sum()),
        raw_above_U_count=int((values > U).sum()),
        raw_at_or_below_L_count=int((values <= L).sum()),
        raw_at_or_above_U_count=int((values >= U).sum()),
    )
    return result


def rows_frame(legacy, rows):
    import pandas as pd

    base = legacy.fixed_frame(rows)
    extra = (
        [k for k in rows[0] if k not in legacy.COLUMNS]
        if rows
        else [
            "marker",
            "calibration_valid",
            "L",
            "U",
            "normalization_method",
            "raw_observed_mean_CD8",
            "raw_finite_pixel_count",
            "raw_observed_min_CD8",
            "raw_observed_max_CD8",
            "raw_below_L_count",
            "raw_above_U_count",
            "raw_at_or_below_L_count",
            "raw_at_or_above_U_count",
        ]
    )
    for name in extra:
        base[name] = pd.Series([r[name] for r in rows], index=base.index)
    return base
