"""Bilingual, editable scientific figures with CSV source data."""
from common import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

MODELS = ["GMM", "Nimbus_pretrained", "Nimbus_finetuned"]
COLORS = ["#7C8795", "#7FA5CA", "#384E78"]


def style(lang):
    if lang == "zh":
        import subprocess

        font = subprocess.check_output(
            ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True
        ).strip()
        fm.fontManager.addfont(font)
        family = fm.FontProperties(fname=font).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )


def save(fig, name, lang):
    d = R / "figures" / lang
    d.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(d / (name + "." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    t = pd.read_csv(R / "results/all_metrics.csv")
    sd = R / "figures/source_data"
    sd.mkdir(parents=True, exist_ok=True)
    primary = t.query("level=='split' and split!='training_WSIs'")
    primary.to_csv(sd / "performance_panels.csv", index=False)
    im = t.query(
        "level=='image' and split=='external_val_TMA_cores' and threshold_mode=='official_default'"
    )
    im.to_csv(sd / "external_core_panels.csv", index=False)
    for lang in ["zh", "en"]:
        style(lang)
        fig, axes = plt.subplots(2, 4, figsize=(9, 4.9), sharey=True)
        for row, split in enumerate(["internal_val_WSIs", "external_val_TMA_cores"]):
            for col, metric in enumerate(["precision", "recall", "F1", "AP"]):
                ax = axes[row, col]
                for j, model in enumerate(MODELS):
                    g = primary.query("split==@split and model==@model")
                    default = float(
                        g.loc[g.threshold_mode == "official_default", metric].iloc[0]
                    )
                    tuned = float(
                        g.loc[g.threshold_mode == "validation_selected", metric].iloc[0]
                    )
                    ax.plot(
                        [j - 0.12, j + 0.12], [default, tuned], color=COLORS[j], lw=1
                    )
                    ax.scatter(
                        j - 0.12, default, color=COLORS[j], s=28, marker="o", zorder=3
                    )
                    ax.scatter(
                        j + 0.12,
                        tuned,
                        color=COLORS[j],
                        s=28,
                        marker="s",
                        facecolors="white",
                        zorder=3,
                    )
                ax.set_xticks(
                    range(3),
                    [
                        "GMM",
                        "Nimbus\n预训练" if lang == "zh" else "Nimbus\npretrained",
                        "Nimbus\n微调" if lang == "zh" else "Nimbus\nfine-tuned",
                    ],
                )
                ax.set_ylim(0, 1.03)
                ax.set_title(
                    (
                        {"precision": "查准率", "recall": "查全率", "F1": "F1", "AP": "AP"}
                        if lang == "zh"
                        else {}
                    ).get(metric, "AP" if metric == "AP" else metric.title())
                )
                ax.grid(axis="y", color="#eeeeee", lw=0.5)
                if col == 0:
                    ax.set_ylabel(
                        ("内部验证（参与选择）" if row == 0 else "外部 TMA")
                        if lang == "zh"
                        else (
                            "Internal (used for selection)"
                            if row == 0
                            else "External TMA"
                        )
                    )
        fig.suptitle(
            "实心圆：默认 0.5；空心方块：验证集阈值"
            if lang == "zh"
            else "Filled circle: default 0.5; open square: validation-selected threshold",
            y=1.02,
            fontsize=9,
        )
        fig.tight_layout()
        save(fig, "performance", lang)
        # Core-paired changes, sorted by core ID; cores are not claimed independent patients.
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
        base = im.query("model=='GMM'")[["image_id", "F1", "context"]]
        for ax, model, color in zip(axes, MODELS[1:], COLORS[1:]):
            a = (
                im.query("model==@model")[["image_id", "F1"]]
                .merge(base, on="image_id", suffixes=("_nimbus", "_gmm"))
                .sort_values("image_id")
            )
            delta = a.F1_nimbus - a.F1_gmm
            group_positions = []
            group_labels = []
            ax.axhline(0, color="#999999", lw=0.7)
            for context, g in a.groupby("context", sort=False):
                ind = np.flatnonzero(a.context.to_numpy() == context)
                ax.scatter(ind, (g.F1_nimbus - g.F1_gmm), s=15, color=color)
                group_positions.append(float(np.mean(ind)))
                group_labels.append(context.replace("_CD8", ""))
                if ind.max() < len(a) - 1:
                    ax.axvline(ind.max() + 0.5, color="#dddddd", lw=0.6, ls=":")
            ax.set_ylim(-1, 1)
            ax.set_xticks(group_positions, group_labels)
            ax.set_title(
                ("Nimbus 预训练" if model == "Nimbus_pretrained" else "Nimbus 微调")
                if lang == "zh"
                else model.replace("_", " ")
            )
            ax.set_xlabel(
                "组织芯（按原始 ID 排序）" if lang == "zh" else "Cores ordered by original ID"
            )
        axes[0].set_ylabel(
            "F1 差值（Nimbus − GMM）" if lang == "zh" else "F1 difference (Nimbus − GMM)"
        )
        fig.suptitle(
            "默认门限 0.5；每点为一个组织芯（非独立患者）"
            if lang == "zh"
            else "Default threshold 0.5; each point is a core, not an independent patient",
            y=1.02,
            fontsize=9,
        )
        fig.tight_layout()
        save(fig, "paired_core_F1", lang)
        # Per-slide default result: separability and prevalence errors remain available in CSV.
        slide = t.query(
            "level=='slide' and split=='external_val_TMA_cores' and threshold_mode=='official_default'"
        )
        slide.to_csv(sd / "external_slide_panels.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(8, 3.1), sharey=True)
        for ax, (context, g) in zip(axes, slide.groupby("context")):
            for model, color in zip(MODELS, COLORS):
                r = g.loc[g.model == model].iloc[0]
                ax.plot(
                    range(4),
                    [r.precision, r.recall, r.F1, r.AP],
                    "-o",
                    color=color,
                    ms=3,
                    label=(
                        {
                            "GMM": "GMM",
                            "Nimbus_pretrained": "Nimbus 预训练",
                            "Nimbus_finetuned": "Nimbus 微调",
                        }[model]
                        if lang == "zh"
                        else model.replace("_", " ")
                    ),
                )
            ax.set_xticks(
                range(4),
                ["查准率", "查全率", "F1", "AP"]
                if lang == "zh"
                else ["Precision", "Recall", "F1", "AP"],
                rotation=20,
            )
            ax.set_ylim(0, 1.03)
            ax.set_title(context.replace("_CD8", ""))
            ax.grid(axis="y", color="#eeeeee")
        axes[0].set_ylabel("指标值" if lang == "zh" else "Metric")
        axes[-1].legend(fontsize=7, loc="lower right")
        fig.suptitle(
            "默认门限 0.5；各模型使用同一有标签细胞集合"
            if lang == "zh"
            else "Default threshold 0.5; identical labelled cell sets across models",
            y=1.02,
            fontsize=9,
        )
        fig.tight_layout()
        save(fig, "TMA_comparison", lang)
    dump(
        R / "figures/FIGURE_QA.json",
        dict(
            backend="Python matplotlib",
            languages=["zh", "en"],
            formats=["PNG", "SVG editable text", "PDF"],
            statistics="paired descriptive; no cell-level bootstrap or significance tests",
            patient_identity="TMA cores not independent patients",
            needs_visual_inspection=True,
        ),
    )


if __name__ == "__main__":
    main()
