"""Research plot examples for the frozen PhenoBIC CD8 experiment."""

from pathlib import Path
import json
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

A = Path(__file__).parent / "delivery"
F = A / "figures"
F.mkdir(exist_ok=True)
examples = json.loads((A / "examples/example_index.json").read_text())
selected = examples[::3]
for lang in ["zh", "en"]:
    font = Path(
        "C:/Windows/Fonts/msyh.ttc" if lang == "zh" else "C:/Windows/Fonts/arial.ttf"
    )
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axs = plt.subplots(5, 3, figsize=(183 / 25.4, 240 / 25.4))
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.90, bottom=0.075, hspace=0.62, wspace=0.28
    )
    for j, e in enumerate(selected):
        z = np.load(A / "examples" / (e["example_id"] + ".npz"))
        im = z["image"]
        r = e["features"]
        hi = float(np.percentile(im[z["valid"]], 99.5))
        hi = max(hi, 1e-8)
        for col in [0, 1]:
            ax = axs[j, col]
            ax.imshow(im, cmap="gray", vmin=0, vmax=hi, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            width = min(10, im.shape[1] // 3)
            ax.plot([3, 3 + width], [im.shape[0] - 4] * 2, color="white", lw=2)
            ax.text(
                3,
                im.shape[0] - 6,
                str(width) + (" 像素" if lang == "zh" else " px"),
                color="white",
                fontsize=5,
            )
        overlay = np.zeros(im.shape + (4,))
        for key, rgb, alpha in [
            ("inner", (0.15, 0.5, 0.9), 0.4),
            ("edge", (1.0, 0.55, 0.05), 0.75),
            ("reference", (0.68, 0.3, 0.84), 0.65),
        ]:
            overlay[z[key], :3] = rgb
            overlay[z[key], 3] = alpha
        axs[j, 1].imshow(overlay, interpolation="nearest")
        axs[j, 1].contour(z["whole"], levels=[0.5], colors="white", linewidths=0.6)
        axs[j, 0].set_title(
            e["image_id"]
            .replace("external_val_TMA_cores/", "")
            .replace("training_WSIs/", "Train/")
            .replace("internal_val_WSIs/", "Internal/")
            + "\nID "
            + str(e["cell_id"]),
            fontsize=7,
        )
        axs[j, 1].set_title(
            "边缘 / 内部 / 周细胞参照" if lang == "zh" else "Edge / inner / local reference",
            fontsize=7,
        )
        vals = np.array([r[f"edge_sector_{k:02d}_mean"] for k in range(12)])
        axs[j, 2].bar(np.arange(12), vals, color="#4d879c", width=0.8)
        axs[j, 2].set_xticks([0, 3, 6, 9])
        axs[j, 2].set_xlabel("角度扇区" if lang == "zh" else "Angular sector", fontsize=6)
        axs[j, 2].set_ylabel(
            "平均 CD8 强度" if lang == "zh" else "Mean CD8 intensity", fontsize=6
        )
        axs[j, 2].tick_params(labelsize=6)
        text = (
            f"均值 {r['observed_mean_CD8']:.3g}；边缘−内部 {r['edge_inner_mean_diff']:.3g}\n富集 {r['edge_signal_enrichment']:.3g}；归一化偏移 {r['signal_centroid_offset_normalized']:.3g}"
            if lang == "zh"
            else f"Mean {r['observed_mean_CD8']:.3g}; edge-inner {r['edge_inner_mean_diff']:.3g}\nEnrichment {r['edge_signal_enrichment']:.3g}; offset/r {r['signal_centroid_offset_normalized']:.3g}"
        )
        axs[j, 2].set_title(text, fontsize=6.5)
    fig.suptitle(
        "真实细胞特征核查：固定原始实例掩码"
        if lang == "zh"
        else "Real-cell feature review: original instance masks",
        fontsize=11,
    )
    note = (
        "橙色：掩码边缘；蓝色：内部；紫色：局部参照；白线：实例边界。\n逐 ROI 调整灰度仅用于显示，数值计算使用原始强度。掩码为核扩张近似；比例尺为像素。"
        if lang == "zh"
        else "Orange: mask edge; blue: inner; purple: local reference; white: instance boundary.\nPer-ROI contrast is for display only. Quantification uses original intensities. Expanded-nucleus masks; scale bars in pixels."
    )
    fig.text(0.5, 0.022, note, ha="center", fontsize=6.5)
    stem = F / ("real_examples_" + lang)
    fig.savefig(str(stem) + ".png", dpi=240)
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".svg")
    plt.close(fig)
print("EXAMPLES READY")
