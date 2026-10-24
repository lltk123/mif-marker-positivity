"""Static scientific figures: source CSV, PNG, editable-text SVG; no inference changes."""
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from PIL import Image

MODELS = ["GMM", "intensity_8", "morphospatial_90", "E1", "E2", "E3", "E4"]
LABELS = ["GMM", "Raw 8", "Full 90", "E1", "E2", "E3", "E4"]
COLORS = ["#777777", "#A9A9A9", "#C9C9C9", "#0072B2", "#009E73", "#E69F00", "#CC79A7"]
P = [
    "gain_0.5",
    "gain_2",
    "bias_0.05",
    "bias_0.10",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "erode_1",
    "dilate_1",
]
PL = [
    "Gain 0.5",
    "Gain 2",
    "Bias 5%",
    "Bias 10%",
    "Rot 90",
    "Rot 180",
    "Rot 270",
    "Erode 1px",
    "Dilate 1px",
]


def main(root):
    root = Path(root)
    out = root / "figures"
    out.mkdir(exist_ok=True)
    source = out / "source_data"
    source.mkdir(exist_ok=True)
    contract = dict(
        conclusion="共同校准、固定定位特征及 GMM 分数的实际配对收益与敏感性，结论不预设 E4 最优",
        evidence="正式完整有标签集合；稳定性独立固定抽样；所有图从 CSV 绘制",
        exports=[
            "Chinese PNG",
            "Chinese editable SVG",
            "English PNG",
            "English editable SVG",
            "source CSV",
        ],
        risks=["不把细胞视作独立患者", "不把 OOF 门限开发拟合值称泛化", "稳定性失效同时呈现", "不隐藏低成绩或缺失"],
        backend="Python matplotlib",
        font="Microsoft YaHei for Chinese; DejaVu Sans for English",
        uncertainty="Descriptive paired results; no cell-level bootstrap or p-values",
    )
    (out / "FIGURE_CONTRACT.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        cnfont = font_manager.FontProperties(fname=str(font)).get_name()
    else:
        cnfont = "DejaVu Sans"
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "font.size": 10,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )
    manifest = []

    def save(fig, name, lang):
        for ext in ["png", "svg"]:
            path = out / (name + "_" + lang + "." + ext)
            fig.savefig(path, bbox_inches="tight", facecolor="white")
            manifest.append(
                dict(
                    file=path.name, language=lang, format=ext, bytes=path.stat().st_size
                )
            )
        plt.close(fig)

    m = pd.read_csv(root / "results/metrics_split.csv")
    m = m[m.split != "training_WSIs"]
    m.to_csv(source / "performance.csv", index=False)
    pairs = pd.read_csv(root / "results/paired_changes.csv")
    p = pairs[
        (pairs.level == "unit")
        & (pairs.split == "external_val_TMA_cores")
        & (pairs.threshold_mode == "fixed_0.5")
    ]
    p = p.copy()
    p["core_order"] = p.image_id.map(
        {uid: i for i, uid in enumerate(sorted(p.image_id.unique()))}
    )
    p.to_csv(source / "core_paired_changes.csv", index=False)
    cal = pd.read_csv(root / "calibration/calibration_parameters.csv")
    cal["source"] = np.where(
        cal.split == "external_val_TMA_cores", cal.context, cal.split
    )
    cal = cal.sort_values(["split", "context", "image_id"])
    cal["unit_number"] = np.arange(1, len(cal) + 1)
    cal.to_csv(source / "calibration_units.csv", index=False)
    av = pd.read_csv(root / "results/feature_availability.csv")
    av = av[av.level == "unit"].merge(
        cal[["image_id", "source"]], on="image_id", validate="many_to_one"
    )
    av = av.groupby(["source", "feature"])[["n", "n_missing"]].sum().reset_index()
    av["missing_rate"] = av.n_missing / av.n
    av.to_csv(source / "missingness.csv", index=False)
    st = pd.read_csv(root / "stability/stability_unit.csv")
    sr = []
    for key, g in st.groupby(["model", "perturbation", "threshold_mode"]):
        r = dict(zip(["model", "perturbation", "threshold_mode"], key))
        r.update(
            n_total=int(g.total_pairs.sum()),
            n_paired=int(g.valid_pairs.sum()),
            flip_rate=g.flip_count.sum() / g.valid_pairs.sum(),
            flip_or_unscoreable_rate=(g.flip_count.sum() + g.unscoreable.sum())
            / g.total_pairs.sum(),
            median_unit_p95_score_change=g.score_change_p95.median(),
        )
        sr.append(r)
    ss = pd.DataFrame(sr)
    ss.to_csv(source / "stability_plot_data.csv", index=False)
    for lang in ["zh", "en"]:
        zh = lang == "zh"
        plt.rcParams["font.family"] = cnfont if zh else "DejaVu Sans"
        for mode, suffix in [("fixed_0.5", "fixed"), ("OOF_threshold", "oof")]:
            fig, axs = plt.subplots(
                2, 3, figsize=(13, 7), sharey=True, layout="constrained"
            )
            for i, split in enumerate(["internal_val_WSIs", "external_val_TMA_cores"]):
                g = (
                    m[(m.split == split) & (m.threshold_mode == mode)]
                    .set_index("model")
                    .loc[MODELS]
                )
                for j, metric in enumerate(["F1", "precision", "recall"]):
                    ax = axs[i, j]
                    ax.bar(np.arange(7), g[metric], color=COLORS, width=0.72)
                    ax.set_xticks(range(7), LABELS, rotation=25, ha="right")
                    ax.set_ylim(0, 1.08)
                    ax.grid(axis="y", alpha=0.18)
                    ax.set_axisbelow(True)
                    for k, val in enumerate(g[metric]):
                        ax.text(k, val + 0.018, "%.3f" % val, ha="center", fontsize=8)
                    ax.set_title(
                        ("内部 WSI" if i == 0 else "外部 TMA")
                        + " · "
                        + {"F1": "F1", "precision": "查准率", "recall": "查全率"}[metric]
                        if zh
                        else ("Internal WSI" if i == 0 else "External TMA")
                        + " · "
                        + metric
                    )
            fig.suptitle(
                ("固定 0.5 主结果" if mode == "fixed_0.5" else "训练 OOF 门限补充结果；旧基线仍为 0.5")
                if zh
                else (
                    "Primary: fixed threshold 0.5"
                    if mode == "fixed_0.5"
                    else "Supplement: training OOF thresholds; old baselines remain at 0.5"
                ),
                fontsize=14,
            )
            save(fig, "01_performance_" + suffix, lang)
        fig, axs = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        for ax, comparison in zip(
            axs.flat, ["E1 - intensity_8", "E2 - E1", "E3 - E1", "E4 - E3"]
        ):
            g = p[p.comparison == comparison].sort_values(["context", "image_id"])
            x = np.arange(len(g))
            for context, color in zip(
                ["TMA1_CD8", "TMA2_CD8", "TMA3_CD8"], ["#0072B2", "#E69F00", "#009E73"]
            ):
                mask = (g.context == context).to_numpy()
                ax.scatter(
                    x[mask],
                    g.F1_delta.to_numpy()[mask] * 100,
                    c=color,
                    s=25,
                    label=context.replace("_CD8", ""),
                    alpha=0.9,
                )
            ax.axhline(0, color="#777777", lw=0.8)
            ax.set_title(comparison.replace("intensity_8", "Raw 8"))
            ax.set_xlabel("组织芯顺序（见源 CSV）" if zh else "Core order (see source CSV)")
            ax.set_ylabel("F1 变化（百分点）" if zh else "F1 change (percentage points)")
            ax.grid(alpha=0.15)
        axs[0, 0].legend(frameon=False, ncol=3, fontsize=8)
        fig.suptitle(
            "逐组织芯配对变化：固定 0.5" if zh else "Paired changes by tissue core: fixed 0.5",
            fontsize=14,
        )
        save(fig, "02_core_changes", lang)
        fig, axs = plt.subplots(
            3,
            1,
            figsize=(13, 11),
            layout="constrained",
            gridspec_kw={"height_ratios": [1, 1, 1.8]},
        )
        axs[0].bar(cal.unit_number, cal.U, color="#0072B2")
        axs[0].set_yscale("log")
        axs[0].set_ylabel("U（原始单位，对数轴）" if zh else "U (raw units, log scale)")
        axs[0].set_title(
            "共同校准与定位特征缺失"
            if zh
            else "Common calibration and localization-feature missingness"
        )
        axs[1].bar(cal.unit_number, cal.at_or_below_L_fraction * 100, color="#777777")
        axs[1].set_ylabel("边界 0 像素比例（%）" if zh else "Pixels at normalized 0 (%)")
        axs[1].set_xlabel(
            "校准单位编号；对应关系见 calibration_units.csv"
            if zh
            else "Calibration unit index; mapping in calibration_units.csv"
        )
        for ax in axs[:2]:
            ax.set_xlim(0, 49)
            ax.grid(axis="y", alpha=0.15)
        s9 = [
            "edge_inner_normalized_diff",
            "edge_signal_enrichment",
            "high_signal_edge_coverage",
            "angular_R1",
            "angular_R2",
            "signal_centroid_offset_normalized",
            "contact_noncontact_normalized_diff",
            "neighbor_edge_area_fraction",
            "contact_neighbor_band_normalized_diff",
        ]
        short = [
            "Edge-inner",
            "Enrichment",
            "High coverage",
            "R1",
            "R2",
            "Centroid offset",
            "Contact-far",
            "Contact fraction",
            "Contact-neighbor",
        ]
        heat = (
            av[av.feature.isin(s9)]
            .pivot(index="source", columns="feature", values="missing_rate")
            .reindex(columns=s9)
        )
        im = axs[2].imshow(
            heat.to_numpy() * 100, vmin=0, vmax=100, cmap="Blues", aspect="auto"
        )
        axs[2].set_yticks(range(len(heat)), heat.index)
        axs[2].set_xticks(range(9), short, rotation=30, ha="right")
        for y in range(len(heat)):
            for x in range(9):
                v = heat.iloc[y, x] * 100
                axs[2].text(
                    x,
                    y,
                    "%.1f" % v,
                    ha="center",
                    va="center",
                    color="white" if v > 55 else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=axs[2], label="缺失率（%）" if zh else "Missing (%)")
        save(fig, "03_calibration_missingness", lang)
        for mode, suffix in [("fixed_0.5", "fixed"), ("OOF_threshold", "oof")]:
            fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
            g = ss[ss.threshold_mode == mode]
            for ax, col, title, scale in zip(
                axs,
                [
                    "median_unit_p95_score_change",
                    "flip_rate",
                    "flip_or_unscoreable_rate",
                ],
                (
                    ["各单位 p95 分数变化的中位数", "有效配对翻转率（%）", "翻转或不可评分（%）"]
                    if zh
                    else [
                        "Median of unit p95 |score change|",
                        "Flip rate among valid pairs (%)",
                        "Flipped or unscoreable (%)",
                    ]
                ),
                [1, 100, 100],
            ):
                arr = (
                    g.pivot(index="model", columns="perturbation", values=col)
                    .reindex(index=MODELS, columns=P)
                    .to_numpy()
                    * scale
                )
                vmax = max(np.nanmax(arr), 0.001)
                im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
                ax.set_yticks(range(7), LABELS)
                ax.set_xticks(range(9), PL, rotation=60, ha="right", fontsize=8)
                ax.set_title(title, fontsize=11)
                for y in range(7):
                    for x in range(9):
                        v = arr[y, x]
                        ax.text(
                            x,
                            y,
                            ("%.2g" % v if scale == 1 else "%.2f" % v),
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if v > 0.55 * vmax else "black",
                        )
                for x in [3.5, 6.5]:
                    ax.axvline(x, color="black", lw=0.7)
                fig.colorbar(im, ax=ax, shrink=0.6)
            fig.suptitle(
                ("冻结模型稳定性 · " + mode) if zh else "Frozen-model stability · " + mode,
                fontsize=14,
            )
            save(fig, "04_stability_" + suffix, lang)
        examples = pd.read_csv(root / "examples/example_index.csv")
        examples.to_csv(source / "examples.csv", index=False)
        fig, axs = plt.subplots(
            len(examples), 2, figsize=(12, 3 * len(examples)), layout="constrained"
        )
        for y, r in examples.iterrows():
            z = np.load(root / "examples" / r["file"])
            raw = z["raw"]
            norm = z["normalized"]
            ax = axs[y, 0]
            im = ax.imshow(raw, cmap="gray", vmin=r.L, vmax=r.U)
            ax.contour(z["whole"], levels=[0.5], colors=["#FFFFFF"], linewidths=0.6)
            fig.colorbar(im, ax=ax, shrink=0.6)
            ax.set_title(
                str(r.image_id).replace("external_val_TMA_cores/", "")
                + " / ID "
                + str(r.cell_id),
                fontsize=9,
            )
            ax.axis("off")
            ax = axs[y, 1]
            ax.imshow(norm, cmap="gray", vmin=0, vmax=1)
            for name, color in [
                ("edge", "#00BFC4"),
                ("inner", "#E69F00"),
                ("reference", "#00AA55"),
            ]:
                overlay = np.zeros(norm.shape + (4,))
                from matplotlib.colors import to_rgb

                overlay[z[name]] = list(to_rgb(color)) + [0.4]
                ax.imshow(overlay)
            ax.set_title(
                (
                    "青：边缘；橙：内部；绿：局部参照\n"
                    if zh
                    else "Cyan: edge; orange: inner; green: local reference\n"
                )
                + "mean %.3f | R1 %.3f | R2 %.3f | offset %.3f"
                % (
                    r.norm_whole_mean,
                    r.angular_R1,
                    r.angular_R2,
                    r.signal_centroid_offset_normalized,
                ),
                fontsize=9,
            )
            ax.axis("off")
        fig.suptitle(
            "真实样例：共享单位 L/U 显示，无逐细胞归一化"
            if zh
            else "Real examples: shared unit L/U display, no per-cell normalization",
            fontsize=13,
        )
        save(fig, "05_real_examples", lang)
    pd.DataFrame(manifest).to_csv(out / "figure_manifest.csv", index=False)
    checks = []
    for r in manifest:
        p = out / r["file"]
        if r["format"] == "png":
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                checks.append(
                    dict(file=p.name, width=im.width, height=im.height, readable=True)
                )
        else:
            txt = p.read_text(encoding="utf-8")
            assert "<text" in txt
            checks.append(dict(file=p.name, editable_text=True, readable=True))
    pd.DataFrame(checks).to_csv(out / "figure_file_checks.csv", index=False)
    print("FIGURES COMPLETE", len(manifest))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    main(args.root)
