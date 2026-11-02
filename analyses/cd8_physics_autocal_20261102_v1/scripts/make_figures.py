"""Bilingual scientific figures from verified output CSVs; no illustrative/generated images."""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

R = Path(__file__).resolve().parents[1]
OUT = R / "reports/figures"
OUT.mkdir(parents=True, exist_ok=True)
C = ["#6786A1", "#BF8A60", "#93A88E", "#8C7E9B"]
data = R / "results"
FONT = Path(
    __import__("os").environ.get(
        "MIF_CJK_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
)
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    ZH = font_manager.FontProperties(fname=str(FONT)).get_name()
else:
    ZH = "Noto Sans CJK SC"


def style(lang):
    plt.rcParams.update(
        {
            "font.family": ZH if lang == "zh" else "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
        }
    )


def save(fig, name, lang, caption):
    fig.savefig(OUT / (name + "_" + lang + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT / (name + "_" + lang + ".svg"), bbox_inches="tight")
    fig.savefig(OUT / (name + "_" + lang + ".pdf"), bbox_inches="tight")
    fig.savefig(
        OUT / (name + "_" + lang + ".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    (OUT / (name + "_" + lang + "_caption.txt")).write_text(caption, encoding="utf-8")
    plt.close(fig)


def labels(ax, letter):
    ax.text(-0.13, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)


def stage0(lang):
    zh = lang == "zh"
    t = pd.read_csv(data / "STAGE0_FULL_METRICS.csv")
    t = t[(t.unit == "pooled") & (t.threshold_mode == "dev_FPR01_each_WSI")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), layout="constrained")
    for row, (prefix, xs) in enumerate(
        [
            ("gain_", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]),
            ("zero_", [0.7, 0.85, 0.92, 0.95, 0.98]),
        ]
    ):
        names = [
            ("clean" if prefix == "gain_" and x == 1 else prefix + str(x)) for x in xs
        ]
        for col, metric in enumerate(["F1", "AP"]):
            ax = axes[row, col]
            mu = []
            sd = []
            ens = []
            for name in names:
                z = t[t.scenario == name]
                a = z[z.model.str.startswith("A_s")][metric]
                mu.append(a.mean())
                sd.append(a.std(ddof=1))
                ens.append(z[z.model == "ENS_A"][metric].iloc[0])
            ax.errorbar(
                xs,
                mu,
                yerr=sd,
                color=C[0],
                fmt="o-",
                ms=4,
                lw=1,
                capsize=2,
                label="种子均值 ± SD" if zh else "Seed mean ± SD",
            )
            ax.plot(
                xs,
                ens,
                "s--",
                ms=4,
                lw=1,
                color=C[1],
                label="概率集成" if zh else "Probability ensemble",
            )
            ax.set_ylabel(metric)
            ax.set_xlabel(
                ("增益系数" if zh else "Gain factor")
                if row == 0
                else ("目标零值比例" if zh else "Target zero fraction")
            )
            ax.set_xticks(xs)
            ax.set_xticklabels(
                [str(x) if row == 0 else str(int(x * 100)) + "%" for x in xs]
            )
            ax.grid(axis="y", alpha=0.15)
            labels(ax, "abcd"[row * 2 + col])
            if row == 0:
                ax.set_xscale("log", base=2)
                ax.set_xticks(xs)
                ax.set_xticklabels([str(x) for x in xs])
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "技术代理扰动下的判别与排序" if zh else "Discrimination and ranking under technical proxies",
        fontsize=11,
    )
    caption = (
        "3张内部WSI，共539,132细胞；每条件3种子。误差棒为种子指标SD，橙线为概率集成，二者不同。阈值沿用干净内部集逐WSI≤1% FPR规则；扰动后实际FPR不作保证。零值目标的逐WSI实际分母/比例见STAGE0_PIXEL_DENOMINATORS。裁剪级归一化后代理，不是物理成像复刻。"
        if zh
        else "Three internal source WSIs, 539,132 cells per condition, three model seeds. Error bars: SD of seed metrics; orange: probability ensemble. Clean internal per-WSI 1% FPR thresholds remain frozen; perturbed FPR is not guaranteed. Actual zero fractions and denominators are in STAGE0_PIXEL_DENOMINATORS.csv. Normalized crop-level proxies, not reconstructed acquisition physics."
    )
    save(fig, "F1_technical_curves", lang, caption)
    # Full16-scenario evidence, including noise/gamma/compound.
    e = t[(t.model == "ENS_A") & (t.scenario != "clean")].copy()
    order = list(dict.fromkeys(e.scenario))
    e = e.set_index("scenario").loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 6.1), sharey=True, layout="constrained")

    def pretty(s):
        for prefix, a, b in [
            ("gain_", "增益 ×", "Gain ×"),
            ("gamma_", "伽马 ", "Gamma "),
            ("zero_", "目标零值 ", "Zero target "),
        ]:
            if s.startswith(prefix):
                return (a if zh else b) + s[len(prefix) :]
        return {
            "noise_low": "低噪声" if zh else "Low noise",
            "noise_high": "高噪声" if zh else "High noise",
            "compound": "组合压力" if zh else "Compound stress",
        }[s]

    ys = np.arange(len(e))
    for k, (metric, label) in enumerate([("delta_F1", "ΔF1"), ("delta_AP", "ΔAP")]):
        ax[k].barh(ys, e[metric], color=C[k], height=0.62)
        ax[k].axvline(0, color="#555555", lw=0.8)
        ax[k].set_xlabel(label)
        ax[k].grid(axis="x", alpha=0.15)
        labels(ax[k], "ab"[k])
    ax[0].set_yticks(ys)
    ax[0].set_yticklabels([pretty(s) for s in e.index])
    ax[0].invert_yaxis()
    fig.suptitle(
        "全部预注册扰动：相对干净输入的变化"
        if zh
        else "All predeclared perturbations: change from clean inputs",
        fontsize=11,
    )
    save(
        fig,
        "F2_all_scenarios",
        lang,
        caption
        + "\n"
        + (
            "概率集成；同一细胞和冻结阈值配对比较。"
            if zh
            else "Probability ensemble; paired cells and frozen operating points."
        ),
    )


def zerogate(lang):
    zh = lang == "zh"
    t = pd.read_csv(data / "ZERO_LABEL_DEV_METRICS.csv")
    t = t[(t.unit == "pooled") & (t.threshold_mode == "source_LOO_FPR01")]
    methods = ["spatial", "mixture", "ridge"]
    names = (
        ["空间低信号候选", "非零混合分布", "代理截距回归"]
        if zh
        else ["Spatial candidates", "Nonzero mixture", "Proxy intercept ridge"]
    )
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    models = ["A_s0", "A_s1", "A_s2", "ENS_A"]
    for j, model in enumerate(models):
        for i, m in enumerate(methods):
            g = t[(t.model == model) & (t.method == m)]
            xx = i + (j - 1.5) * 0.14
            ax[0].scatter(
                xx,
                float(g[g.scenario == "clean"].delta_F1.iloc[0]),
                s=28,
                color=C[j],
                marker="s" if model == "ENS_A" else "o",
                label=("集成" if zh else "Ensemble")
                if model == "ENS_A" and i == 0
                else ("种子 " + str(j) if zh else "Seed " + str(j))
                if i == 0
                else None,
            )
            ax[1].scatter(
                xx,
                float(g.delta_AP.min()),
                s=28,
                color=C[j],
                marker="s" if model == "ENS_A" else "o",
            )
    for k, a in enumerate(ax):
        a.set_xticks(range(3))
        a.set_xticklabels(names, rotation=15, ha="right")
        a.axhline(0 if k == 0 else -0.003, color="#555555", ls="--", lw=0.8)
        a.grid(axis="y", alpha=0.15)
        labels(a, "ab"[k])
    ax[0].set_ylabel("干净输入 ΔF1" if zh else "Clean ΔF1")
    ax[1].set_ylabel("全部条件最小 ΔAP" if zh else "Minimum ΔAP across conditions")
    ax[0].legend(fontsize=7)
    fig.suptitle(
        "零标签候选的来源分组开发门槛"
        if zh
        else "Source-grouped development gates for label-free candidates",
        fontsize=11,
    )
    save(
        fig,
        "F3_zero_label_gates",
        lang,
        "3sourceWSIs; leave-one-source-out, all derived perturbations stay with source. Gate: cleanF1>=baseline; no condition AP loss>.003; all3seeds andENS mustpass. These are development results, not external validation."
        if not zh
        else "3个来源WSI留一；同来源所有派生扰动整体隔离。门槛：干净F1不降、任何条件AP下降不超过0.003；三种子和集成均须通过。此图为开发结果，不是外部验证。",
    )


def fewshot(lang):
    zh = lang == "zh"
    t = pd.read_csv(data / "ALL_FEWSHOT_POOLED.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.5), layout="constrained")
    metrics = [
        "F1",
        "AP",
        "weak_recall",
        "FPR",
        "core_F1_median",
        "core_prevalence_MAE",
    ]
    cn = ["F1", "AP", "弱观测信号召回率", "实际 FPR", "逐 core F1 中位数", "逐 core 阳性比例 MAE"]
    en = [
        "F1",
        "AP",
        "Weak observed-signal recall",
        "Actual FPR",
        "Median core F1",
        "Mean core prevalence error",
    ]
    random = t[t.strategy.isin(["random_reused", "random"])]
    active = t[t.strategy == "active_boundary"]
    for k, metric in enumerate(metrics):
        ax = axes.flat[k]
        for frame, color, marker, label in [
            (random, C[0], "o", "随机抽样" if zh else "Random"),
            (active, C[1], "s", "边界主动抽样" if zh else "Boundary active"),
        ]:
            g = frame.groupby("budget")[metric].agg(["mean", "std"])
            ax.errorbar(
                g.index,
                g["mean"],
                yerr=g["std"],
                fmt=marker + "-",
                ms=4,
                lw=1,
                capsize=2,
                color=color,
                label=label,
            )
        ax.set_xlabel("每 core 标签数" if zh else "Labels per core")
        ax.set_ylabel(cn[k] if zh else en[k])
        ax.set_xticks([0, 5, 10, 20, 50, 100])
        ax.set_xscale("symlog", linthresh=5)
        ax.set_xticks([0, 5, 10, 20, 50, 100])
        ax.set_xticklabels(
            ["0", "5", "10", "20", "50", "100"], rotation=45, ha="right", fontsize=7
        )
        ax.grid(axis="y", alpha=0.15)
        labels(ax, "abcdef"[k])
        if k == 0:
            ax.legend(fontsize=7)
    fig.suptitle(
        "共同评价子集上的少量标签校准"
        if zh
        else "Few-label calibration on common evaluation subsets",
        fontsize=11,
    )
    save(
        fig,
        "F4_few_label_calibration",
        lang,
        "每次433,280细胞；42core，各排除100预留细胞；3次预留池重复。误差棒为重复间SD，不是患者不确定性。模型为3种子概率集成；随机0/20/50/100直接复用旧计算，随机5/10及主动5/10/20为新增。确定性topK无逆倾向权重，存在选择偏差假设。"
        if zh
        else "433,280 evaluation cells per repeat; 42 cores, 100 reserved and excluded per core; three pool repeats. Error bars: repeat SD, not patient uncertainty. Frozen three-seed probability ensemble. Random0/20/50/100 reused; random5/10 and active5/10/20 newlycomputed. Deterministic topK uses no inversepropensity weights; selection-bias assumptions remain.",
    )


def trainfig(lang):
    p = data / "CONTINUATION_CV.csv"
    if not p.exists():
        return
    zh = lang == "zh"
    t = pd.read_csv(p)
    variants = ["control", "physics", "sdf", "ranking"]
    names = (
        ["配对继续训练", "技术一致性", "距离场与边缘", "排序损失"]
        if zh
        else [
            "Matched continuation",
            "Technical consistency",
            "SDF + edge",
            "Ranking loss",
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), layout="constrained")
    for k, metric in enumerate(["F1", "AP"]):
        for j, v in enumerate(variants):
            y = t[t.variant == v][metric].to_numpy()
            axes[k].scatter(j + np.linspace(-0.09, 0.09, len(y)), y, color=C[j], s=28)
            axes[k].plot([j - 0.17, j + 0.17], [np.mean(y)] * 2, color="black", lw=1.5)
        axes[k].set_xticks(range(4))
        axes[k].set_xticklabels(names, rotation=18, ha="right")
        axes[k].set_ylabel(metric)
        axes[k].grid(axis="y", alpha=0.15)
        labels(axes[k], "ab"[k])
    fig.suptitle(
        "单变量新训练：完整训练 WSI 留一评价"
        if zh
        else "Single-variable continuation: held-out training WSIs",
        fontsize=11,
    )
    save(
        fig,
        "F5_continuation_development",
        lang,
        "3个训练WSI留一，每点一个来源WSI，横线为均值；不是独立患者。复用旧折权重继续训练2个epoch，配对相同采样与额外更新数。外部标签未参与筛选。"
        if zh
        else "Three held-out training source WSIs; each point one WSI, line mean; not independent patients. Two additional epochs from existingfoldweights, pairedsamples andupdatenumbers. External labels excluded fromselection.",
    )


def main():
    for lang in ["zh", "en"]:
        style(lang)
        for fn in [stage0, zerogate, fewshot, trainfig]:
            fn(lang)
    print("BILINGUAL_FIGURES_COMPLETE")


if __name__ == "__main__":
    main()
