"""Render scientific figures from saved summary tables."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

A = Path(__file__).parent / "delivery"
F = A / "figures"
F.mkdir(exist_ok=True)
colors = {"GMM": "#777777", "intensity_8": "#347FA3", "morphospatial_90": "#C16C2D"}
models = list(colors)
labels = {
    "zh": ["GMM", "8 项强度", "90 项形态与空间"],
    "en": ["GMM", "8 intensity features", "90 morphology + spatial"],
}
splits = ["internal_val_WSIs", "external_val_TMA_cores"]


def save(fig, stem):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(F / (stem + "." + ext), dpi=300)
    plt.close(fig)


for lang in ["zh", "en"]:
    zh = lang == "zh"
    font = Path("C:/Windows/Fonts/msyh.ttc" if zh else "C:/Windows/Fonts/arial.ttf")
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    names = [
        "uniform",
        "ring",
        "one_sided",
        "bright_neighbor_without_spill",
        "bright_neighbor_with_spill",
        "dark",
        "tiny",
        "no_reference",
        "image_boundary",
    ]
    titles = (
        [
            "均匀亮",
            "环形亮、内部暗",
            "单侧亮点",
            "邻居亮、目标暗",
            "接触侧增加亮信号",
            "全暗",
            "极小实例",
            "无可用局部参照",
            "图像边界截断",
        ]
        if zh
        else [
            "Uniform",
            "Bright ring, dark inner",
            "One-sided spot",
            "Bright neighbor, dark target",
            "Signal at contact side",
            "All dark",
            "Tiny instance",
            "No local reference",
            "Truncated at image border",
        ]
    )
    controls = pd.read_csv(A / "tests/controlled_features.csv").set_index("image_id")
    fig, axs = plt.subplots(3, 3, figsize=(183 / 25.4, 200 / 25.4))
    fig.subplots_adjust(
        top=0.90, bottom=0.12, left=0.07, right=0.97, hspace=0.72, wspace=0.35
    )
    for i, (name, title, ax) in enumerate(zip(names, titles, axs.flat)):
        z = np.load(A / "tests" / (name + ".npz"))
        im = z["image"]
        r = controls.loc[name]
        ax.imshow(im, cmap="gray", vmin=0, vmax=30, interpolation="nearest")
        ax.contour(z["whole"], levels=[0.5], colors="#e7a73c", linewidths=0.65)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(chr(97 + i) + "  " + title, fontsize=8)
        if i < 3:
            text = f"diff={r.edge_inner_mean_diff:.2f}; enrich={r.edge_signal_enrichment:.2f}\nCV={r.sector_signal_cv:.2f}; offset/r={r.signal_centroid_offset_normalized:.2f}"
        elif i in [3, 4]:
            text = f"whole={r.whole_mean:.2f}\ncontact diff={r.neighbor_edge_mean_diff:.2f}"
        else:
            text = (
                [
                    "no_signal / MAD=0",
                    "small_whole / small_inner",
                    "reference_insufficient",
                    "truncated",
                ][i - 5]
                + "\n"
                + ("相关特征 NaN" if zh else "Affected features: NaN")
            )
        ax.text(
            0.5,
            -0.10,
            text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
        )
    fig.suptitle(
        "受控图像验证：信号分布与异常情形"
        if zh
        else "Controlled validation: signal patterns and edge cases",
        fontsize=11,
        y=0.97,
    )
    fig.text(
        0.5,
        0.025,
        "橙线：目标实例。diff：边缘−内部；enrich：面积校正富集；CV：扇区信号变异系数。\n无参照、弱信号或异常几何通过 NaN/QC 表达，不据此决定细胞阴阳性。"
        if zh
        else "Orange contour: target instance. diff: edge-inner; enrich: area-adjusted enrichment; CV: sector signal variation.\nUndefined localization and geometry are represented by NaN/QC; these are not cell-positivity rules.",
        ha="center",
        fontsize=6.5,
    )
    save(fig, "controlled_validation_" + lang)
    if not (A / "results/model_comparison.csv").exists():
        continue
    metrics = pd.read_csv(A / "results/model_comparison.csv")
    contexts = pd.read_csv(A / "results/metrics_per_context.csv")
    fig, axs = plt.subplots(2, 2, figsize=(183 / 25.4, 167 / 25.4))
    fig.subplots_adjust(
        top=0.88, bottom=0.13, left=0.10, right=0.97, hspace=0.45, wspace=0.30
    )
    source = []
    for i, split in enumerate(splits):
        title = (
            ["内部验证 WSI", "外部验证 TMA"]
            if zh
            else ["Internal WSI validation", "External TMA validation"]
        )[i]
        for model, label in zip(models, labels[lang]):
            r = metrics[(metrics.model == model) & (metrics.split == split)].iloc[0]
            z = np.load(A / "results" / (model + "_" + split + "_curves.npz"))
            for j, (xx, yy, score) in enumerate(
                [("fpr", "tpr", "ROC_AUC"), ("recall", "precision", "AP")]
            ):
                base = np.linspace(0, len(z[xx]) - 1, min(2000, len(z[xx]))).astype(int)
                jumps = np.flatnonzero(
                    (np.abs(np.diff(z[xx])) > 0.005) | (np.abs(np.diff(z[yy])) > 0.005)
                )
                idx = np.unique(np.r_[base, jumps, jumps + 1])
                xs = z[xx][idx]
                ys = z[yy][idx]
                axs[i, j].plot(
                    xs,
                    ys,
                    color=colors[model],
                    lw=1.3,
                    label=label + f" ({r[score]:.3f})",
                    drawstyle="steps-post" if j == 1 else "default",
                )
                if lang == "en":
                    source.extend(
                        dict(
                            split=split,
                            model=model,
                            curve=score,
                            x=float(x),
                            y=float(y),
                        )
                        for x, y in zip(xs, ys)
                    )
        axs[i, 0].plot([0, 1], [0, 1], color="#BBBBBB", lw=0.7, ls="--")
        axs[i, 1].axhline(r.true_prevalence, color="#BBBBBB", lw=0.7, ls="--")
        for j in [0, 1]:
            ax = axs[i, j]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.025)
            ax.set_title(title + (" · ROC" if j == 0 else " · PR"), fontsize=9)
            ax.legend(
                loc="lower right" if j == 0 else "lower left",
                fontsize=6.5,
                frameon=False,
            )
            ax.set_xlabel(
                ("假阳性率" if zh else "False-positive rate")
                if j == 0
                else ("查全率" if zh else "Recall")
            )
            ax.set_ylabel(
                ("真阳性率" if zh else "True-positive rate")
                if j == 0
                else ("查准率" if zh else "Precision")
            )
            ax.grid(alpha=0.15)
    fig.suptitle(
        "PhenoBIC：固定模型的跨图像评估"
        if zh
        else "PhenoBIC: evaluation of fixed models across images",
        fontsize=11,
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "训练：3 张原始 WSI；仅使用其 101,057 个已标注细胞；未调验证阈值"
        if zh
        else "Training: 101,057 annotated cells from 3 original WSIs; no validation-based threshold tuning",
        ha="center",
        fontsize=7,
    )
    fig.text(
        0.5,
        0.02,
        "图例括号为 AUC / AP。GMM 每图像无监督校准；分类器使用训练标签，监督条件不同。"
        if zh
        else "Legend: AUC / AP. GMM uses per-image unsupervised calibration; classifiers use training labels.",
        ha="center",
        fontsize=6.5,
    )
    save(fig, "model_curves_" + lang)
    if source:
        pd.DataFrame(source).to_csv(A / "results/plotted_curve_source.csv", index=False)
    groups = [
        ("internal_val_WSIs", None),
        ("external_val_TMA_cores", "TMA1_CD8"),
        ("external_val_TMA_cores", "TMA2_CD8"),
        ("external_val_TMA_cores", "TMA3_CD8"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(183 / 25.4, 83 / 25.4))
    fig.subplots_adjust(top=0.70, bottom=0.23, left=0.08, right=0.98, wspace=0.35)
    for j, (column, title) in enumerate(
        zip(
            ["F1", "precision", "recall"],
            ["F1", "查准率", "查全率"] if zh else ["F1", "Precision", "Recall"],
        )
    ):
        ax = axs[j]
        for model, label in zip(models, labels[lang]):
            vals = []
            for split, context in groups:
                d = (
                    metrics[(metrics.model == model) & (metrics.split == split)]
                    if context is None
                    else contexts[
                        (contexts.model == model)
                        & (contexts.split == split)
                        & (contexts.context == context)
                    ]
                )
                vals.append(float(d.iloc[0][column]))
            ax.plot(
                range(4), vals, "o-", ms=3, lw=1.2, color=colors[model], label=label
            )
        ax.set_xticks(range(4))
        ax.set_xticklabels(
            ["内部" if zh else "Internal", "TMA1", "TMA2", "TMA3"],
            rotation=30,
            ha="right",
            fontsize=7,
        )
        ax.set_ylim(0, 1.04)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "分组表现：内部提升未转化为 TMA 泛化提升"
        if zh
        else "Internal gains do not translate to better TMA generalization",
        fontsize=10,
        y=0.96,
    )
    fig.legend(
        *axs[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=3,
        fontsize=7,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "同一固定分类器与阈值应用于全部组。不同组原始强度尺度不同；该图为描述性比较。"
        if zh
        else "The same fixed classifier and threshold are used across groups; raw intensity scales vary. Descriptive comparison.",
        ha="center",
        fontsize=6.5,
    )
    save(fig, "context_comparison_" + lang)
print("FIGURES READY")
