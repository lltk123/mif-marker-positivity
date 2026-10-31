"""CPU-only numeric checks; optional real crops never stand in for full-image calibration."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from common import R, sha
from normalization import normalize_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-dir", type=Path)
    args = parser.parse_args()
    rows = []

    def record(name, passed, **details):
        rows.append(dict(test=name, passed=bool(passed), **details))
        if not passed:
            raise AssertionError(name)

    x = np.array([[-3, 0, 1, 2], [4, 8, np.nan, 100]], dtype=np.float64)
    valid = np.ones_like(x, bool)
    valid[1, 3] = False
    out = normalize_pixels(x, 0, 4, valid)
    record(
        "shared_pixel_formula_and_zero",
        np.allclose(out[0], [0, 0, 0.25, 0.5]) and out[1, 0] == 1 and out[1, 1] == 1,
    )
    record("invalid_support_preserved", np.isnan(out[1, 2:]).all())
    record("degenerate_bounds_unscorable", np.isnan(normalize_pixels(x, 4, 4)).all())
    record("negative_range_unscorable", np.isnan(normalize_pixels(x, 5, 4)).all())
    record(
        "no_eligible_instance_NaN_bounds",
        np.isnan(normalize_pixels(x, np.nan, np.nan)).all(),
    )
    bounds_json = {
        name: value if np.isfinite(value) else None
        for name, value in {"L": np.nan, "U": np.nan}.items()
    }
    record(
        "invalid_bounds_strict_JSON",
        json.loads(json.dumps(bounds_json, allow_nan=False)) == {"L": None, "U": None},
    )
    record("float64_computation", out.dtype == np.float64)
    # Tail sensitivity is a method check, not an assertion of classification benefit.
    lows = np.arange(1000, dtype=np.float64) / 1000
    highs = lows + 2
    old_max = highs.max()
    highs[-1] = 1e9
    record(
        "outer_upper_limits_single_extreme",
        np.percentile(highs, 99, method="linear") < old_max,
    )
    shared = R.parent / "cd8_robust_calibration_20261030/scripts/calibration_core.py"
    if shared.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("shared_calibration_core", shared)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = module.normalize_pixels(x, 0, 4, valid_mask=valid)
        record(
            "shared_implementation_exact",
            np.array_equal(out, expected, equal_nan=True),
            shared_code_sha256=sha(shared),
        )
    if args.crop_dir:
        for path in sorted(args.crop_dir.glob("*.npz"))[:5]:
            with np.load(path, allow_pickle=False) as content:
                candidate = next(
                    (
                        k
                        for k in ["CD8_original", "image", "raw", "cd8"]
                        if k in content
                    ),
                    None,
                )
                if candidate is None:
                    raise ValueError(
                        f"no recognizable raw CD8 array: {path.name}; keys={content.files}"
                    )
                image = content[candidate]
                finite = image[np.isfinite(image)]
                lower, upper = float(finite.min()), float(finite.max())
                if upper <= lower:
                    continue
                support = content["valid"] if "valid" in content else np.isfinite(image)
                normalized = normalize_pixels(image, lower, upper, support)
                observed = normalized[np.isfinite(image) & support]
                record(
                    "real_crop_numeric_only",
                    np.isfinite(observed).all()
                    and observed.min() >= 0
                    and observed.max() <= 1
                    and np.isnan(normalized[~support]).all(),
                    crop=path.name,
                    pixels=image.size,
                    scope="synthetic test bounds on existing development crop; not full-unit calibration or inference",
                )
    for path in (R / "scripts").glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    (R / "tests").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(R / "tests/offline_numeric_checks.csv", index=False)
    (R / "tests/OFFLINE_CHECKS.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "checks": len(rows),
                "full_images_used": False,
                "model_inference_run": False,
                "new_training_run": False,
            },
            indent=2,
        )
    )
    print("OFFLINE_CHECKS_PASSED", len(rows))


if __name__ == "__main__":
    main()
