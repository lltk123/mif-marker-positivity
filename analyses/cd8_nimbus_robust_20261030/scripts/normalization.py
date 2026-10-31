"""Shared, label-free outer-P1/P99 calibration; no fitting on a crop or tile."""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common import R, P, sha, units

SHARED = P / "analyses/cd8_robust_calibration_20261030"


def normalize_pixels(image, lower, upper, valid=None):
    """Map finite observed pixels in float64; preserve unsupported pixels as NaN."""
    x = np.asarray(image, dtype=np.float64)
    support = np.isfinite(x)
    if valid is not None:
        if np.shape(valid) != x.shape:
            raise ValueError("valid mask shape differs from image")
        support &= np.asarray(valid, bool)
    scale = max(abs(float(lower)), abs(float(upper)), np.finfo(np.float64).tiny)
    if (
        not (np.isfinite(lower) and np.isfinite(upper))
        or upper - lower <= 32 * np.finfo(np.float64).eps * scale
    ):
        return np.full(x.shape, np.nan, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    out[support] = np.clip((x[support] - lower) / (upper - lower), 0.0, 1.0)
    return out


@lru_cache(maxsize=1)
def calibration():
    """Read one frozen bound per full original WSI/core, never derive it from labels."""
    paths = {
        "bounds": SHARED / "calibration/unit_bounds.csv",
        "manifest": SHARED / "calibration/calibration_manifest.json",
        "protocol": SHARED / "PROTOCOL.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"shared calibration is not ready: {path}")
    table = pd.read_csv(paths["bounds"], float_precision="round_trip")
    needed = {
        "dataset_id",
        "image_id",
        "split",
        "context",
        "L",
        "U",
        "calibration_invalid",
        "n_eligible",
        "n_instances",
    }
    if not needed.issubset(table.columns):
        raise ValueError(
            f"calibration missing columns: {sorted(needed - set(table.columns))}"
        )
    if table.image_id.duplicated().any() or set(table.image_id) != set(units()):
        raise ValueError("calibration units must match all original image IDs exactly")
    if not table.dataset_id.eq("PhenoBIC-Data").all():
        raise ValueError("unexpected calibration dataset")
    bad = (
        table.calibration_invalid.astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )
    if bad.isna().any():
        raise ValueError("calibration_invalid must be explicitly boolean")
    table["calibration_invalid"] = bad
    manifest = json.loads(paths["manifest"].read_text())
    signature = {name + "_sha256": sha(path) for name, path in paths.items()}
    if (
        manifest.get("status") != "complete"
        or manifest.get("method_id") != "E1_outer_P1_P99"
    ):
        raise ValueError(
            "shared calibration manifest is incomplete or names another method"
        )
    if (
        manifest.get("unit_bounds_sha256") != signature["bounds_sha256"]
        or manifest.get("protocol_sha256") != signature["protocol_sha256"]
    ):
        raise ValueError("shared calibration manifest checksum mismatch")
    if manifest.get("rows") != 1817783 or manifest.get("n_units") != len(table):
        raise ValueError("shared calibration manifest population mismatch")
    return table.set_index("image_id", drop=False), signature, manifest


def bounds_for(image_id):
    row = calibration()[0].loc[image_id]
    result = row.to_dict()
    result["L"], result["U"] = float(row.L), float(row.U)
    return result


def calibration_signature():
    return calibration()[1]
