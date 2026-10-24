"""Development-only FP/FN review; morphology is used for audit selection, never Nimbus input."""
from common import *
from plot_results import style, save
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import importlib.util, types


def old_io():
    previous = sys.modules["common"]
    stub = types.ModuleType("common")
    for k, v in dict(A=E, OLD=OLD, G=G, DATA=D).items():
        setattr(stub, k, v)
    sys.modules["common"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "original_io_adapter", E / "scripts/io_adapter.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.modules["common"] = previous
    return mod


def main():
    out = R / "development_review"
    out.mkdir(exist_ok=True)
    cols = KEYS[:3] + [
        "whole_mean",
        "neighbor_edge_mean_diff",
        "neighbor_edge_area_fraction",
        "local_reference_median",
        "edge_inner_normalized_diff",
        "sector_signal_cv",
        "edge_inner_mean_diff",
        "qc_truncated",
        "qc_context_clipped",
        "qc_disconnected",
    ]
    feat = pd.read_parquet(OLD / "PhenoBIC_CD8_features.parquet", columns=cols)
    assert feat.dataset_id.eq("PhenoBIC-Data").all()
    feat["marker"] = "CD8"
    cells = load_cells()
    lab = cells.loc[
        cells.split.isin(["training_WSIs", "internal_val_WSIs"])
        & cells.ground_truth_binary.isin([0, 1])
    ]
    feat = lab.merge(
        feat, on=KEYS, how="left", validate="one_to_one", suffixes=("", "_feature")
    )
    train = feat.query("split=='training_WSIs'")
    pos = train.query("ground_truth_binary==1")
    cut = dict(
        weak_mean=float(pos.whole_mean.quantile(0.25)),
        neighbor_diff=float(train.neighbor_edge_mean_diff.quantile(0.75)),
        reference=float(train.local_reference_median.quantile(0.75)),
        sector_cv=float(train.sector_signal_cv.quantile(0.75)),
    )
    dump(
        out / "case_selection_rules.json",
        dict(
            training_derived_cutoffs=cut,
            n_per_model_category_error=2,
            selection="seed0 uniform among eligible development errors; categories are image-property proxies, not proven failure mechanisms",
        ),
    )
    categories = {
        "weak_positive": (feat.ground_truth_binary == 1)
        & (feat.whole_mean <= cut["weak_mean"]),
        "bright_neighbor_side": (feat.neighbor_edge_area_fraction > 0)
        & (feat.neighbor_edge_mean_diff >= cut["neighbor_diff"]),
        "diffuse_local_reference": (feat.local_reference_median >= cut["reference"])
        & (feat.edge_inner_normalized_diff.abs() < 0.2),
        "localized_edge_brightness": (feat.sector_signal_cv >= cut["sector_cv"])
        & (feat.edge_inner_mean_diff > 0),
        "incomplete_support": feat.qc_truncated
        | feat.qc_context_clipped
        | feat.qc_disconnected,
    }
    selected = []
    counts = []
    for model in ["Nimbus_pretrained", "Nimbus_finetuned"]:
        pred = pd.read_parquet(
            R / "predictions" / model / "all_cells.parquet",
            columns=KEYS
            + [
                "score",
                "prediction_official_default",
                "prediction_validation_selected",
                "model_version",
            ],
        )
        d = feat.merge(pred, on=KEYS, validate="one_to_one")
        d["error_type"] = np.where(d.ground_truth_binary == 1, "FN", "FP")
        err = d.prediction_official_default.ne(d.ground_truth_binary)
        d.loc[
            err,
            KEYS
            + [
                "split",
                "context",
                "ground_truth_binary",
                "score",
                "prediction_official_default",
                "prediction_validation_selected",
                "model_version",
                "error_type",
            ],
        ].to_csv(out / (model + "_all_development_errors.csv.gz"), index=False)
        # Recompute proxy masks by key membership so dataframe row order is never used for linking.
        for category, mask in categories.items():
            eligible = feat.loc[mask, KEYS].merge(
                d.loc[err], on=KEYS, validate="one_to_one"
            )
            for error in ["FP", "FN"]:
                q = eligible.loc[eligible.error_type == error].sort_values(KEYS)
                counts.append(
                    dict(
                        model=model,
                        category=category,
                        error_type=error,
                        available=len(q),
                        selected=min(2, len(q)),
                    )
                )
                if len(q):
                    q = q.iloc[
                        np.sort(
                            np.random.default_rng(0).choice(
                                len(q), min(2, len(q)), replace=False
                            )
                        )
                    ].copy()
                    q["category"] = category
                    q["model"] = model
                    selected.append(q)
    chosen = pd.concat(selected, ignore_index=True)
    pd.DataFrame(counts).to_csv(out / "case_availability.csv", index=False)
    io = old_io()
    registry = {}
    images = []
    idx = []
    neighbor_rows = []
    truth_lookup = {
        (r.dataset_id, r.image_id, int(r.cell_id), r.marker): r.ground_truth_binary
        for r in lab.itertuples()
    }
    for n, row in chosen.iterrows():
        image = row.image_id
        cid = int(row.cell_id)
        found = None
        for e in units()[image]:
            if e["key"] not in registry:
                registry[e["key"]] = io.boxes(e)
            ids, boxes = registry[e["key"]]
            j = np.flatnonzero(ids == cid)
            if len(j):
                found = (e, boxes[j[0]])
                break
        assert found is not None
        e, bb = found
        a, m, valid, origin, truncated = io.roi(e, bb, halo=48)
        name = f"case_{n:03d}_{row.model}"
        for neighbor_id in np.unique(m):
            if neighbor_id == 0 or neighbor_id == cid:
                continue
            hit = (m == neighbor_id) & valid & np.isfinite(a)
            if not hit.any():
                continue
            neighbor_rows.append(
                dict(
                    case=n,
                    model=row.model,
                    dataset_id=row.dataset_id,
                    image_id=image,
                    target_cell_id=cid,
                    neighbor_cell_id=int(neighbor_id),
                    marker="CD8",
                    ground_truth_binary=truth_lookup.get(
                        (row.dataset_id, image, int(neighbor_id), "CD8"), np.nan
                    ),
                    mean_CD8_in_visible_crop=float(np.mean(a[hit], dtype=np.float64)),
                    n_visible_pixels=int(hit.sum()),
                    touches_crop_border=bool(
                        hit[0].any()
                        or hit[-1].any()
                        or hit[:, 0].any()
                        or hit[:, -1].any()
                    ),
                    interpretation="visible crop statistics, not a complete-neighbor mean or proof of contamination",
                )
            )
        np.savez_compressed(
            out / (name + ".npz"),
            CD8_original=a,
            instance_mask=m,
            valid=valid,
            cell_id=cid,
            origin_yx=origin,
        )
        finite = a[valid & np.isfinite(a)]
        vmax = float(np.quantile(finite, 0.995)) if len(finite) else 1.0
        vmax = max(vmax, 1e-8)
        images.append((a, m, valid, cid, vmax))
        idx.append(
            dict(
                case=n,
                category=row.category,
                model=row.model,
                error_type=row.error_type,
                **{k: row[k] for k in KEYS},
                split=row.split,
                context=row.context,
                ground_truth=int(row.ground_truth_binary),
                score=float(row.score),
                prediction_default=int(row.prediction_official_default),
                prediction_validation_selected=int(row.prediction_validation_selected),
                whole_mean=float(row.whole_mean),
                local_reference_median=float(row.local_reference_median),
                neighbor_edge_mean_diff=float(row.neighbor_edge_mean_diff),
                neighbor_edge_area_fraction=float(row.neighbor_edge_area_fraction),
                sector_signal_cv=float(row.sector_signal_cv),
                edge_inner_normalized_diff=float(row.edge_inner_normalized_diff),
                qc_truncated=bool(row.qc_truncated),
                qc_context_clipped=bool(row.qc_context_clipped),
                qc_disconnected=bool(row.qc_disconnected),
                raw_image_source=str(D / e["source"]["image"]),
                mask_source=str(D / e["source"]["mask"]),
                original_owner_fov=e["patch"],
                crop_origin_y=origin[0],
                crop_origin_x=origin[1],
                display_vmin=0.0,
                display_vmax=vmax,
                display_gamma=1.0,
                scale_bar_px=20,
                mask_semantics="expanded_nucleus",
                selection_interpretation="proxy category requires visual review; no causal contamination claim",
                crop_file=name + ".npz",
            )
        )
    pd.DataFrame(idx).to_csv(out / "development_cases.csv", index=False)
    pd.DataFrame(neighbor_rows).to_csv(out / "visible_neighbors.csv", index=False)
    compare = []
    for model in ["GMM", "Nimbus_pretrained", "Nimbus_finetuned"]:
        q = pd.read_parquet(
            R / "predictions" / model / "all_cells.parquet",
            columns=KEYS
            + [
                "score",
                "prediction_official_default",
                "prediction_validation_selected",
            ],
        )
        joined = pd.DataFrame(idx)[["case", *KEYS, "ground_truth"]].merge(
            q, on=KEYS, how="left", validate="many_to_one"
        )
        joined["model"] = model
        compare.append(joined)
    pd.concat(compare, ignore_index=True).to_csv(
        out / "case_model_comparison.csv", index=False
    )
    names_zh = {
        "weak_positive": "弱阳性候选",
        "bright_neighbor_side": "邻居接触侧亮信号候选",
        "diffuse_local_reference": "高局部参照候选（不等同于弥漫背景）",
        "localized_edge_brightness": "边缘局部亮斑候选",
        "incomplete_support": "掩码／上下文不完整候选",
    }
    for lang in ["zh", "en"]:
        style(lang)
        for cat in categories:
            chosen_idx = [n for n, r in enumerate(idx) if r["category"] == cat]
            if not chosen_idx:
                continue
            n = len(chosen_idx)
            nrows = (n + 3) // 4
            figh = 0.9 + 3.15 * nrows
            fig = plt.figure(figsize=(10, figh))
            for panel, j in enumerate(chosen_idx):
                row, col = divmod(panel, 4)
                row_top = figh - 0.65 - row * 3.15
                ax = fig.add_axes(
                    [
                        (0.12 + 2.5 * col) / 10,
                        (row_top - 0.45 - 2.30) / figh,
                        2.26 / 10,
                        2.30 / figh,
                    ]
                )
                ax.axis("off")
                a, m, v, cid, vmax = images[j]
                r = idx[j]
                cmap = plt.get_cmap("gray").copy()
                cmap.set_bad("#626A73")
                ax.imshow(
                    np.ma.array(a, mask=~v),
                    cmap=cmap,
                    vmin=0,
                    vmax=vmax,
                    interpolation="nearest",
                )
                edge = find_boundaries(m == cid, mode="inner")
                overlay = np.zeros((*m.shape, 4))
                overlay[find_boundaries(m, mode="inner") & (m != cid)] = [
                    0.65,
                    0.75,
                    0.85,
                    0.4,
                ]
                overlay[edge] = [1, 0.65, 0.1, 1]
                ax.imshow(overlay)
                ax.plot([6, 26], [a.shape[0] - 7] * 2, color="white", lw=2)
                ax.text(6, a.shape[0] - 12, "20 px", color="white", fontsize=6)
                title = f"C{r['case']:02d} | " + (
                    (
                        ("预训练" if r["model"] == "Nimbus_pretrained" else "微调")
                        if lang == "zh"
                        else r["model"].replace("Nimbus_", "")
                    )
                    + f" | {r['error_type']}\nID {cid} | y={r['ground_truth']} | s={r['score']:.3f}"
                )
                fig.text(
                    (1.25 + 2.5 * col) / 10,
                    row_top / figh,
                    title,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
            fig.suptitle(
                names_zh[cat]
                if lang == "zh"
                else (
                    "High local reference (diffuse background unconfirmed)"
                    if cat == "diffuse_local_reference"
                    else cat.replace("_", " ")
                ),
                fontsize=11,
            )
            fig.text(
                0.5,
                0.015,
                "橙线：扩张核掩码；灰线：邻居；每幅图单独设置显示灰度范围"
                if lang == "zh"
                else "Orange: expanded-nucleus mask; grey: neighbors; display contrast is set per panel.",
                ha="center",
                fontsize=8,
            )
            save(fig, "development_" + cat, lang)
    print("DEVELOPMENT_CASES", len(idx), flush=True)


if __name__ == "__main__":
    main()
