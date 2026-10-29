"""Review development-set CD8 errors using frozen image-property strata.

The figures display unmodified CD8 crop pixels and existing expanded-nucleus
instance boundaries. A spatial proxy is not evidence of causal contamination.
Only training and internal-validation labels are read for case selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd

KEYS = ["dataset_id", "image_id", "cell_id", "marker"]
DEVELOPMENT = ["training_WSIs", "internal_val_WSIs"]
FEATURES = [
    "whole_mean", "neighbor_edge_mean_diff", "neighbor_edge_area_fraction",
    "local_reference_median", "edge_inner_normalized_diff", "sector_signal_cv",
    "edge_inner_mean_diff", "qc_truncated", "qc_context_clipped", "qc_disconnected",
]
MODELS = ["TabM_A_s0", "TabM_B_s0", "TabM_C_s0", "HGB_C_s0"]
CATEGORY_NAMES = {
    "weak_positive": ("弱阳性候选", "Weak-positive candidate"),
    "bright_neighbor_side_proxy": ("邻居接触侧亮信号候选", "Bright neighbor-side proxy"),
    "high_local_reference_proxy": ("高局部参照候选", "High local-reference proxy"),
    "localized_edge_signal_proxy": ("边缘局部亮信号候选", "Localized edge-signal proxy"),
    "incomplete_support": ("掩码或上下文不完整候选", "Incomplete mask or context"),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_keys(frame):
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate explicit cell keys")
    if not frame.dataset_id.eq("PhenoBIC-Data").all():
        raise ValueError("Only PhenoBIC-Data is permitted")
    if not frame.marker.eq("CD8").all() or not frame.cell_id.gt(0).all():
        raise ValueError("Expected CD8 and positive integer instance IDs")


def finite_cutoff(rules, key):
    value = rules.get(key)
    if value is None or not np.isfinite(float(value)):
        return None
    return float(value)


def category_masks(data, rules):
    """Use frozen training cutoffs; do not estimate cutoffs from selected cases."""
    masks, missing = {}, {}
    specifications = {
        "weak_positive": (
            ["whole_mean"], "weak_positive_whole_mean_q25",
            lambda d, c: d.ground_truth_binary.eq(1) & d.whole_mean.le(c),
        ),
        "bright_neighbor_side_proxy": (
            ["neighbor_edge_area_fraction", "neighbor_edge_mean_diff"],
            "contact_bright_diff_q75",
            lambda d, c: d.neighbor_edge_area_fraction.gt(0)
            & d.neighbor_edge_mean_diff.gt(c),
        ),
        "high_local_reference_proxy": (
            ["local_reference_median"], "local_reference_q75",
            lambda d, c: d.local_reference_median.gt(c),
        ),
        "localized_edge_signal_proxy": (
            ["sector_signal_cv", "edge_inner_mean_diff"], "sector_cv_q75",
            lambda d, c: d.sector_signal_cv.gt(c) & d.edge_inner_mean_diff.gt(0),
        ),
    }
    for name, (columns, cutoff_name, function) in specifications.items():
        absent = [c for c in columns if c not in data]
        cutoff = finite_cutoff(rules, cutoff_name)
        if absent or cutoff is None:
            masks[name] = pd.Series(False, index=data.index)
            missing[name] = "missing_features:" + ",".join(absent) if absent else (
                "missing_or_nonfinite_frozen_cutoff:" + cutoff_name
            )
        else:
            masks[name] = function(data, cutoff).fillna(False)
            missing[name] = ""
    qc = ["qc_truncated", "qc_context_clipped", "qc_disconnected"]
    absent = [c for c in qc if c not in data]
    if absent:
        masks["incomplete_support"] = pd.Series(False, index=data.index)
        missing["incomplete_support"] = "missing_features:" + ",".join(absent)
    else:
        masks["incomplete_support"] = data[qc].fillna(False).astype(bool).any(axis=1)
        missing["incomplete_support"] = ""
    return masks, missing


def label_changes(data):
    """Evaluate fixed score > 0.5; comparison is descriptive, seed 0 only."""
    data = data.copy()
    y = data.ground_truth_binary.to_numpy(dtype=int)
    finite = np.isfinite(data.score_TabM_B_s0) & np.isfinite(data.score_TabM_C_s0)
    b = data.score_TabM_B_s0.gt(0.5).to_numpy()
    c = data.score_TabM_C_s0.gt(0.5).to_numpy()
    b_wrong, c_wrong = b != y, c != y
    outcome = np.where(y == 1, "FN", "FP")
    data["change_type"] = np.select(
        [finite & b_wrong & ~c_wrong, finite & ~b_wrong & c_wrong,
         finite & b_wrong & c_wrong, ~finite],
        ["corrected_", "introduced_", "persistent_", "unscorable_"],
        default="both_correct_",
    )
    data["change_type"] = data.change_type + outcome
    data["comparison_score_finite"] = finite
    for model in MODELS:
        col = "score_" + model
        if col in data:
            finite_model = np.isfinite(data[col].to_numpy(float))
            data["prediction_" + model] = np.where(
                finite_model, data[col].gt(0.5).to_numpy(dtype=int), np.nan
            )
    return data


def choose_cases(data, rules, seed=0):
    """At most two unique instances per stratum; prefer corrected/new errors."""
    data = label_changes(data).sort_values(KEYS).reset_index(drop=True)
    masks, missing = category_masks(data, rules)
    rng = np.random.default_rng(seed)
    selected, availability, used = [], [], set()
    for category in CATEGORY_NAMES:
        candidates = data.loc[masks[category] & data.comparison_score_finite].copy()
        errors = candidates.loc[
            candidates.change_type.str.startswith(("corrected_", "introduced_", "persistent_"))
        ]
        counts = errors.change_type.value_counts().to_dict()
        chosen = []
        # One improvement and one deterioration, when each is available.
        for prefix in ["corrected_", "introduced_"]:
            pool = errors.loc[errors.change_type.str.startswith(prefix)]
            pool = pool.loc[
                [tuple(row) not in used for row in pool[KEYS].itertuples(index=False, name=None)]
            ]
            if len(pool):
                hit = pool.iloc[int(rng.integers(len(pool)))].copy()
                used.add(tuple(hit[k] for k in KEYS))
                hit["selection_slot"] = prefix.rstrip("_")
                chosen.append(hit)
        # Fill remaining slots with persistent errors or remaining changed errors.
        while len(chosen) < 2:
            pool = errors.loc[
                [tuple(row) not in used for row in errors[KEYS].itertuples(index=False, name=None)]
            ]
            if not len(pool):
                break
            hit = pool.iloc[int(rng.integers(len(pool)))].copy()
            used.add(tuple(hit[k] for k in KEYS))
            hit["selection_slot"] = "remaining_development_error"
            chosen.append(hit)
        for hit in chosen:
            hit["category"] = category
            selected.append(hit)
        availability.append(dict(
            category=category, n_stratum=len(candidates), n_errors=len(errors),
            n_corrected_FP=int(counts.get("corrected_FP", 0)),
            n_corrected_FN=int(counts.get("corrected_FN", 0)),
            n_introduced_FP=int(counts.get("introduced_FP", 0)),
            n_introduced_FN=int(counts.get("introduced_FN", 0)),
            n_persistent_FP=int(counts.get("persistent_FP", 0)),
            n_persistent_FN=int(counts.get("persistent_FN", 0)),
            n_selected=len(chosen), requested=2,
            missing_reason=missing[category],
            shortfall_reason=(
                "" if len(chosen) == 2 else
                "No further unique development errors; instances are not reused across categories"
            ),
        ))
    columns = list(data.columns) + ["selection_slot", "category"]
    frame = pd.DataFrame(selected).reset_index(drop=True) if selected else pd.DataFrame(columns=columns)
    return frame, pd.DataFrame(availability)


def load_inputs(root):
    import pyarrow.parquet as pq

    path = root / "data/cells.parquet"
    names = pq.read_schema(path).names
    required = KEYS + ["split", "context", "ground_truth_binary"]
    absent = [c for c in required if c not in names]
    if absent:
        raise ValueError("Required cell metadata missing: " + ", ".join(absent))
    columns = required + [c for c in FEATURES if c in names]
    if "ground_truth_original" in names:
        columns.append("ground_truth_original")
    # Push down the split filter: external labels are not read for visual review.
    data = pd.read_parquet(path, columns=columns, filters=[("split", "in", DEVELOPMENT)])
    data = data.loc[data.ground_truth_binary.isin([0, 1])].copy()
    validate_keys(data)
    labels = data[KEYS + ["ground_truth_binary"]].copy()
    model_inputs = {}
    for model in MODELS:
        pred_path = root / "predictions" / (model + ".parquet")
        if not pred_path.exists():
            if model == "HGB_C_s0":
                model_inputs[model] = {"status": "optional prediction not available"}
                continue
            raise FileNotFoundError(pred_path)
        pred = pd.read_parquet(
            pred_path, columns=KEYS + ["score"],
            filters=[("image_id", "in", data.image_id.unique().tolist())],
        )
        validate_keys(pred)
        pred = pred.rename(columns={"score": "score_" + model})
        data = data.merge(pred, on=KEYS, how="left", validate="one_to_one", indicator=True)
        if not data["_merge"].eq("both").all():
            raise ValueError("Missing development prediction rows: " + model)
        data = data.drop(columns="_merge")
        model_inputs[model] = dict(
            path=str(pred_path), bytes=pred_path.stat().st_size,
            mtime_ns=pred_path.stat().st_mtime_ns,
            score_key_hash=hashlib.sha256(
                pred.sort_values(KEYS).to_csv(index=False, float_format="%.17g").encode()
            ).hexdigest(),
        )
    rules_path = root / "data/risk_strata.json"
    rules = read_json(rules_path)
    if rules.get("fit_split") != "training_WSIs":
        raise ValueError("Risk cutoffs must explicitly record fit_split=training_WSIs")
    return data, labels, rules, model_inputs


def load_original_io(project):
    """Load the existing audited FOV-context reconstruction without editing it."""
    earlier = project / "analyses/cd8_e1_e4_stability_20261023"
    path = earlier / "scripts/io_adapter.py"
    previous = sys.modules.get("common")
    stub = types.ModuleType("common")
    constants = dict(
        A=earlier,
        OLD=project / "analyses/cd8_morphospatial_20261023",
        G=project / "analyses/cd8_scimap_gmm_20261023",
        DATA=project / "datasets/PhenoBIC-Data",
    )
    for name, value in constants.items():
        setattr(stub, name, value)
    sys.modules["common"] = stub
    try:
        spec = importlib.util.spec_from_file_location("tabm_original_io", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = previous
    groups = {}
    for entry in module.AUDIT:
        if entry["split"] in DEVELOPMENT:
            groups.setdefault(module.image_id(entry), []).append(entry)
    return module, groups, path


def configure_plotting(language, font_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.ft2font import FT2Font

    family = "DejaVu Sans"
    if language == "zh":
        candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]
        try:
            result = subprocess.run(
                ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"],
                capture_output=True, text=True, check=True,
            )
            candidates.insert(0, Path(result.stdout.strip()))
        except (OSError, subprocess.CalledProcessError):
            pass
        candidates.extend(Path(f.fname) for f in fm.fontManager.ttflist if "CJK" in f.name)
        if font_path is not None:
            explicit_font = Path(font_path).expanduser().resolve()
            if not explicit_font.is_file():
                raise FileNotFoundError(explicit_font)
            candidates = [explicit_font]
        for font in candidates:
            if font.is_file() and ord("细") in FT2Font(str(font)).get_charmap():
                fm.fontManager.addfont(str(font))
                family = fm.FontProperties(fname=str(font)).get_name()
                break
        else:
            raise RuntimeError("Chinese figure export needs a CJK font (e.g. Noto Sans CJK SC)")
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "svg.fonttype": "none",
        "pdf.fonttype": 42, "axes.unicode_minus": False, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def draw_case(image, mask, valid, row, output, language, font_path=None):
    from skimage.segmentation import find_boundaries

    plt = configure_plotting(language, font_path=font_path)
    cid = int(row["cell_id"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.65))
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#626A73")
    for ax in axes:
        ax.imshow(
            np.ma.array(image, mask=~valid | ~np.isfinite(image)), cmap=cmap,
            vmin=row["display_vmin"], vmax=row["display_vmax"],
            interpolation="nearest",
        )
        ax.axis("off")
    overlay = np.zeros((*mask.shape, 4))
    overlay[find_boundaries(mask, mode="inner") & (mask != cid)] = [0.65, 0.75, 0.85, 0.6]
    overlay[find_boundaries(mask == cid, mode="inner")] = [1.0, 0.65, 0.10, 1.0]
    axes[1].imshow(overlay)
    # Keep every boundary, but restrict text to the closest visible neighbors.
    target_y, target_x = np.nonzero((mask == cid) & valid)
    target_center = (float(np.mean(target_x)), float(np.mean(target_y)))
    neighbor_centers = []
    for other in np.unique(mask):
        if other == 0 or other == cid:
            continue
        hit = (mask == other) & valid
        if not hit.any():
            continue
        yy, xx = np.nonzero(hit)
        x, y = float(np.mean(xx)), float(np.mean(yy))
        distance2 = (x - target_center[0]) ** 2 + (y - target_center[1]) ** 2
        neighbor_centers.append((distance2, int(other), x, y))
    candidates = [(cid, *target_center)] + [
        (other, x, y) for _, other, x, y in sorted(neighbor_centers)[:6]
    ]
    # Physical pixel size is unavailable; label only the true pixel length.
    bar = min(20, max(1, image.shape[1] // 4))
    for ax in axes:
        ax.plot([5, 5 + bar], [image.shape[0] - 7] * 2, color="white", lw=2)
        ax.text(5, image.shape[0] - 11, f"{bar} px", color="white", fontsize=7)
    axes[0].set_title("a  原始 CD8" if language == "zh" else "a  Original CD8", loc="left")
    axes[1].set_title(
        "b  目标与邻居实例边界" if language == "zh" else "b  Target and neighbor boundaries",
        loc="left",
    )
    catname = CATEGORY_NAMES[row["category"]][0 if language == "zh" else 1]
    error_names = {
        "corrected_FP": "纠正假阳性", "corrected_FN": "纠正假阴性",
        "introduced_FP": "新增假阳性", "introduced_FN": "新增假阴性",
        "persistent_FP": "持续假阳性", "persistent_FN": "持续假阴性",
    }
    transition = error_names.get(row["change_type"], row["change_type"]) if language == "zh" else row["change_type"].replace("_", " ")
    fig.suptitle(f"{catname} | B → C: {transition}", fontsize=10, y=0.99)
    def display_score(value):
        return f"{value:.2e}" if 0 < abs(value) < 0.001 else f"{value:.3f}"

    text = (
        f"ID {cid} | " + ("人工标签" if language == "zh" else "Label") +
        f" {int(row['ground_truth_binary'])} | " +
        " | ".join(
            f"{model.replace('TabM_', '').replace('_s0', '')}={display_score(row['score_' + model])}"
            for model in MODELS if "score_" + model in row
        )
    )
    fig.text(0.5, 0.145, text, ha="center", fontsize=8)
    note = (
        "橙线：扩张核；灰线：邻居；灰色缺块：无图像支持。共同显示灰度范围，γ=1。\n"
        "模型种子 0，门限 >0.5；候选不证明污染。其余邻居 ID 见 neighbor_context.csv。"
        if language == "zh" else
        "Orange: expanded nucleus; grey: neighbors; grey gaps: missing support. Shared display range, gamma=1.\n"
        "Seed 0, threshold >0.5; proxies do not prove contamination. Other IDs: neighbor_context.csv."
    )
    fig.text(0.5, 0.045, note, ha="center", fontsize=7, linespacing=1.5)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.20, top=0.86, wspace=0.08)
    # Check actual rendered text bounds after aspect ratio and layout are fixed.
    # This also leaves the scale bar and its label completely free of instance IDs.
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = axes[1]
    inset = 3.0 * fig.dpi / 72.0
    frame = ax.get_window_extent(renderer)
    safe = Bbox.from_extents(
        frame.x0 + inset, frame.y0 + inset, frame.x1 - inset, frame.y1 - inset
    )
    scale_points = ax.transData.transform(
        [[-0.5, image.shape[0] - 23], [bar + 12, image.shape[0] - 0.5]]
    )
    scale_region = Bbox.from_extents(
        *np.minimum(scale_points[0], scale_points[1]),
        *np.maximum(scale_points[0], scale_points[1]),
    )
    occupied = []
    for other, x, y in candidates:
        label = ax.text(
            x, y, str(other), color="#FFB347" if other == cid else "#CBD5E0",
            fontsize=6, ha="center", va="center", clip_on=True,
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 0.5},
        )
        label.set_gid("target-cell-id" if other == cid else "neighbor-cell-id")
        bounds = label.get_window_extent(renderer).padded(1.0 * fig.dpi / 72.0)
        within = (
            safe.contains(bounds.x0, bounds.y0) and safe.contains(bounds.x1, bounds.y1)
        )
        if not within or bounds.overlaps(scale_region) or any(bounds.overlaps(b) for b in occupied):
            label.remove()
            continue
        occupied.append(bounds)
    dest = output / language
    dest.mkdir(exist_ok=True)
    for ext in ["png", "svg"]:
        fig.savefig(dest / (row["case_name"] + "." + ext), dpi=300, facecolor="white")
    plt.close(fig)


def make_reviews(root, output, font_path=None):
    data, labels, rules, model_inputs = load_inputs(root)
    chosen, availability = choose_cases(data, rules)
    io, groups, io_path = load_original_io(root.parents[1])
    fingerprint = dict(
        script_sha256=file_sha256(Path(__file__)), rules_sha256=file_sha256(root / "data/risk_strata.json"),
        original_io_sha256=file_sha256(io_path), models=model_inputs,
        cells_bytes=(root / "data/cells.parquet").stat().st_size,
        cells_mtime_ns=(root / "data/cells.parquet").stat().st_mtime_ns,
        font_sha256=file_sha256(Path(font_path)) if font_path is not None else None,
    )
    manifest = output / "REVIEW_MANIFEST.json"
    if manifest.exists() and read_json(manifest)["fingerprint"] != fingerprint:
        raise RuntimeError("Existing review inputs differ; choose a new --output-dir to preserve the earlier review")
    output.mkdir(parents=True, exist_ok=True)
    availability.to_csv(output / "case_availability.csv", index=False)
    write_json(output / "case_selection_rules.json", dict(
        cutoffs=rules, selection_seed=0, model_seed=0, threshold=0.5,
        comparison="score > threshold", maximum_cases=10, splits=DEVELOPMENT,
        sampling="At most two unique instances per category; one corrected and one introduced B-to-C error when available, then other development errors",
        interpretation="All categories are image-property proxies requiring visual inspection; no causal contamination inference",
        high_local_reference_definition="local_reference_median > training q75; identical strict comparison to evaluation",
        localized_edge_definition="sector_signal_cv > training q75 AND edge_inner_mean_diff > 0; a narrower visual-case subset of the evaluation directional_signal_proxy",
    ))
    truth = {
        tuple(row[:4]): row[4]
        for row in labels.itertuples(index=False, name=None)
    }
    registry, records, neighbors = {}, [], []
    for n, series in chosen.iterrows():
        row = series.to_dict()
        cid, image_id = int(row["cell_id"]), row["image_id"]
        found = None
        for entry in groups[image_id]:
            if entry["key"] not in registry:
                registry[entry["key"]] = io.boxes(entry)
            ids, boxes = registry[entry["key"]]
            loc = np.flatnonzero(ids == cid)
            if len(loc):
                found = entry, boxes[loc[0]]
                break
        if found is None:
            raise ValueError(f"Cannot locate explicit instance ID: {image_id}, {cid}")
        entry, bbox = found
        image, mask, valid, origin, owner_truncated = io.roi(entry, bbox, halo=48)
        target = (mask == cid) & valid & np.isfinite(image)
        if not target.any():
            raise ValueError(f"No valid target pixels after context reconstruction: {cid}")
        row["case_name"] = f"case_{n:02d}_{row['category']}"
        row["case_number"] = n
        finite = image[valid & np.isfinite(image)]
        vmax = float(np.quantile(finite, 0.995)) if len(finite) else 1.0
        row.update(
            crop_origin_y=int(origin[0]), crop_origin_x=int(origin[1]),
            display_vmin=0.0, display_vmax=max(vmax, 1e-8), display_gamma=1.0,
            display_rule="Shared within each case; original finite crop p99.5 upper limit; pixels unchanged in NPZ",
            mask_semantics="expanded_nucleus", owner_fov=entry["patch"],
            owner_fov_truncated=bool(owner_truncated),
            original_image_source=str(io.DATA / entry["source"]["image"]),
            original_mask_source=str(io.DATA / entry["source"]["mask"]),
            crop_file=row["case_name"] + ".npz",
            target_valid_pixels=int(target.sum()),
            target_mean_in_crop=float(np.mean(image[target], dtype=np.float64)),
            context_valid_fraction=float(valid.mean()),
        )
        np.savez_compressed(
            output / row["crop_file"], CD8_original=image, instance_mask=mask,
            valid=valid, cell_id=cid, origin_yx=np.asarray(origin),
        )
        for neighbor_id in np.unique(mask):
            if neighbor_id == 0 or neighbor_id == cid:
                continue
            hit = (mask == neighbor_id) & valid & np.isfinite(image)
            if not hit.any():
                continue
            neighbor_key = (row["dataset_id"], image_id, int(neighbor_id), "CD8")
            neighbors.append(dict(
                case_name=row["case_name"], dataset_id=row["dataset_id"], image_id=image_id,
                target_cell_id=cid, neighbor_cell_id=int(neighbor_id), marker="CD8",
                ground_truth_binary=truth.get(neighbor_key, np.nan),
                n_visible_valid_pixels=int(hit.sum()),
                mean_CD8_in_visible_crop=float(np.mean(image[hit], dtype=np.float64)),
                touches_crop_border=bool(hit[0].any() or hit[-1].any() or hit[:, 0].any() or hit[:, -1].any()),
                interpretation="Visible-crop statistics, not full-neighbor mean and not proof of contamination",
            ))
        for language in ["zh", "en"]:
            draw_case(image, mask, valid, row, output, language, font_path=font_path)
        records.append(row)
    index = pd.DataFrame(records) if records else chosen.copy()
    index.to_csv(output / "case_index.csv", index=False)
    neighbor_columns = [
        "case_name", "dataset_id", "image_id", "target_cell_id", "neighbor_cell_id",
        "marker", "ground_truth_binary", "n_visible_valid_pixels",
        "mean_CD8_in_visible_crop", "touches_crop_border", "interpretation",
    ]
    pd.DataFrame(neighbors, columns=neighbor_columns).to_csv(output / "neighbor_context.csv", index=False)
    write_json(manifest, dict(
        fingerprint=fingerprint, status="complete", n_cases=len(index),
        strata_note="High local reference uses strict >q75. Localized edge uses sector CV >q75 plus edge-inner mean difference >0, a narrower visual subset of evaluation directional_signal_proxy.",
        n_visible_neighbor_records=len(neighbors),
        figure_contract={
            "claim": "Inspect how spatial features change individual development errors without assuming mechanism",
            "archetype": "image plate + quant", "backend": "Python/matplotlib",
            "source_data": "case_index.csv, neighbor_context.csv, original-pixel NPZ crops",
            "exports": ["Chinese and English PNG", "editable-text SVG"],
            "physical_scale": "Unavailable; scale bars in pixels",
            "display": "Global linear contrast per crop, shared between raw and overlay panels; no pixel edits",
            "statistics": "Illustrative seed-0 errors only; formal multi-seed results are separate",
        },
    ))
    print(json.dumps({"review_directory": str(output), "cases": len(index)}, ensure_ascii=False))


def self_test():
    """Exercise selection with explicit IDs, missing strata, and strict thresholds."""
    n = 40
    data = pd.DataFrame({
        "dataset_id": ["PhenoBIC-Data"] * n,
        "image_id": ["training_WSIs/test"] * n,
        "cell_id": np.arange(n) * 103 + 90000, "marker": ["CD8"] * n,
        "ground_truth_binary": np.arange(n) % 2,
        "whole_mean": np.ones(n), "neighbor_edge_area_fraction": np.ones(n),
        "neighbor_edge_mean_diff": np.ones(n), "local_reference_median": np.ones(n),
        "sector_signal_cv": np.ones(n), "edge_inner_mean_diff": np.ones(n),
        "qc_truncated": np.ones(n, bool), "qc_context_clipped": np.zeros(n, bool),
        "qc_disconnected": np.zeros(n, bool),
        "score_TabM_A_s0": np.full(n, 0.5),
        "score_TabM_B_s0": np.tile([0.8, 0.2, 0.2, 0.8], n // 4),
        "score_TabM_C_s0": np.tile([0.2, 0.8, 0.8, 0.2], n // 4),
    })
    rules = dict(
        weak_positive_whole_mean_q25=1, contact_bright_diff_q75=0,
        local_reference_q75=0.5, sector_cv_q75=0.5,
    )
    selected, availability = choose_cases(data, rules)
    again, _ = choose_cases(data.sample(frac=1, random_state=7), rules)
    pd.testing.assert_frame_equal(selected[KEYS + ["category"]], again[KEYS + ["category"]])
    assert len(selected) == 10 and not selected.duplicated(KEYS).any()
    assert availability.n_selected.eq(2).all()
    strict = label_changes(data)
    assert strict.prediction_TabM_A_s0.eq(0).all()
    absent, counts = choose_cases(data.drop(columns=FEATURES, errors="ignore"), rules)
    assert absent.empty and counts.n_selected.eq(0).all()
    assert counts.missing_reason.str.len().gt(0).all()
    all_missing = dict(rules, weak_positive_whole_mean_q25=None)
    masks, missing = category_masks(data, all_missing)
    assert not masks["weak_positive"].any() and missing["weak_positive"]
    tied_masks, _ = category_masks(data, dict(rules, local_reference_q75=1, sector_cv_q75=1))
    assert not tied_masks["high_local_reference_proxy"].any()
    assert not tied_masks["localized_edge_signal_proxy"].any()
    validate_keys(data)
    print("SELF_TEST_PASS: explicit IDs, row-order invariance, unique seed-0 sampling, strict threshold, missing strata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--font-path", type=Path, help="Optional CJK font file for Chinese review figures")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.run_dir is None:
            parser.error("--run-dir is required unless --self-test is used")
        make_reviews(
            args.run_dir.resolve(),
            (args.output_dir or args.run_dir / "development_review").resolve(),
            font_path=args.font_path.expanduser().resolve() if args.font_path else None,
        )
