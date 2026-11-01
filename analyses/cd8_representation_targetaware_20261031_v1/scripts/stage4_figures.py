"""Python-only bilingual figures, with locked source data and editable text."""
import os

os.environ.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
import json
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment_common import *
from stage4_review import cjk_font

GROUPS = [
    "image",
    "image_target",
    "image_target_other",
    "tabular_only",
    "image_target_other_tabular",
]
COLORS = ["#8e9ba9", "#7ca5bf", "#5d9c98", "#b6a3bf", "#be8b6b"]


def main():
    assert (RUN / "results/EVALUATION_COMPLETE.json").exists()
    out = RUN / "reports/figures"
    out.mkdir(parents=True, exist_ok=True)
    font = cjk_font()
    m = pd.read_csv(RUN / "results/METRICS_BY_UNIT.csv")
    r = pd.read_csv(RUN / "results/RISK_STRATA.csv")
    cal = pd.read_csv(RUN / "results/TARGET_LABEL_CALIBRATION_SIMULATION.csv")
    comp = pd.read_csv(RUN / "results/COMPOSITION_SUMMARY.csv")
    selected = json.loads((RUN / "models/MODEL_SELECTION_FROZEN.json").read_text())[
        "selected_model"
    ]

    def one(name, mode="fixed_0.5", level="split", unit=TEST):
        return m[
            (m.model == name)
            & (m.split == TEST)
            & (m.threshold_mode == mode)
            & (m.level == level)
            & (m.unit == unit)
        ].iloc[0]

    def risk(name, stratum):
        return r[
            (r.model == name)
            & (r.split == TEST)
            & (r.level == "split")
            & (r.stratum == stratum)
            & (r.threshold_mode == "dev_recall_90")
        ].iloc[0]

    h0 = one("H0")
    h1 = one("H1")
    h2 = one("H2")
    contract = {
        "backend": "Python/matplotlib only",
        "languages": ["zh", "en"],
        "formats": ["PNG", "SVG with editable text", "PDF TrueType"],
        "statistics": "HGB one deterministic fit; neural paired seeds 0/1/2, ensemble curves explicitly separate; label-calibration SD is across 3 sampling repeats, not patient CI",
        "figures": {
            "stage1": {
                "claim": f"Normalization-only delta F1={h1.F1-h0.F1:+.6f}; representation-only delta F1={h2.F1-h1.F1:+.6f}; source recall remains a separate outcome.",
                "archetype": "quantitative grid",
                "panels": [
                    "controlled HGB AP/F1",
                    "achieved source recall at internal R90",
                ],
                "source": ["EXTERNAL_MAIN.csv", "EXTERNAL_BY_TMA.csv"],
            },
            "target_aware": {
                "claim": "Target-aware inputs are evaluated jointly for proxy false positives, weak recall and achieved external recall.",
                "archetype": "quantitative grid",
                "panels": [
                    "diagnostic PR curves with frozen operating points",
                    "proxy tradeoff",
                    "three paired seeds",
                ],
                "source": [
                    "curves/*TEST_DIAGNOSTIC_PR.npz",
                    "RISK_STRATA.csv",
                    "METRICS_BY_UNIT.csv",
                ],
            },
            "calibration": {
                "claim": "Fixed per-core target labels are evaluated on a strictly isolated, identical held-out population across budgets.",
                "archetype": "quantitative grid",
                "panels": ["held-out F1", "source recall", "Brier score"],
                "source": [
                    "TARGET_LABEL_CALIBRATION_SIMULATION.csv",
                    "CALIBRATION_RESERVED_INSTANCES.csv",
                ],
            },
            "composition": {
                "claim": "Only the normalization estimation pool changes; target raw pixel hashes remain identical. This is diagnostic.",
                "archetype": "quantitative grid",
                "panels": ["absolute score change", "fraction changed at .5"],
                "source": ["COMPOSITION_SUMMARY.csv", "COMPOSITION_DIAGNOSTIC.csv"],
            },
        },
    }
    savej(RUN / "reports/FIGURE_CONTRACT.json", contract)
    exports = []

    def save(fig, name, lang, footer):
        fig.text(0.02, -0.02, footer, fontsize=6)
        for ext in ["png", "svg", "pdf"]:
            p = out / (name + "_" + lang + "." + ext)
            fig.savefig(p, dpi=350, bbox_inches="tight")
            exports.append(
                {
                    "figure": name,
                    "language": lang,
                    "format": ext,
                    "path": str(p.relative_to(RUN)),
                    "sha256": sha(p),
                }
            )
        plt.close(fig)

    for lang in ["zh", "en"]:
        zh = lang == "zh"
        plt.rcParams.update(
            {
                "font.family": font if zh else "DejaVu Sans",
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "legend.frameon": False,
            }
        )
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
        names = ["H0", "H1", "H2", "H3", "H4", "H5"]
        v = [one(n) for n in names]
        y = np.arange(6)
        ax[0].scatter([a.F1 for a in v], y - 0.08, s=25, color=COLORS[1], label="F1")
        ax[0].scatter(
            [a.AP for a in v], y + 0.08, s=22, marker="s", color=COLORS[2], label="AP"
        )
        ax[0].set_yticks(y, names)
        ax[0].invert_yaxis()
        ax[0].set_xlim(max(0, min(min(a.F1, a.AP) for a in v) - 0.04), 1)
        ax[0].set_xlabel("外部总体指标" if zh else "Pooled external metric")
        ax[0].set_title(
            "a 逐项改变的HGB对照" if zh else "a Controlled HGB changes",
            loc="left",
            fontweight="bold",
        )
        ax[0].legend(fontsize=7)
        z = np.array(
            [
                [
                    one(n, "dev_recall_90", "context", t).recall
                    for t in ["TMA1_CD8", "TMA2_CD8", "TMA3_CD8"]
                ]
                for n in names
            ]
        )
        im = ax[1].imshow(z, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax[1].set_xticks(range(3), ["TMA1", "TMA2", "TMA3"])
        ax[1].set_yticks(y, names)
        ax[1].set_title(
            "b 内部R90门限下实际外部召回" if zh else "b Actual external recall at internal R90",
            loc="left",
            fontweight="bold",
        )
        for i in range(6):
            for j in range(3):
                ax[1].text(
                    j,
                    i,
                    f"{z[i,j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if z[i, j] > 0.55 else "#243442",
                    fontsize=7,
                )
        fig.colorbar(im, ax=ax[1], shrink=0.8, label="召回率" if zh else "Recall")
        save(
            fig,
            "stage1",
            lang,
            "外部n=437,480；HGB每组1个固定拟合；内部R90不保证外部R90。"
            if zh
            else "External n=437,480; one fixed HGB fit per condition; internal R90 does not guarantee external R90.",
        )
        fig = plt.figure(figsize=(7.2, 5.0), layout="constrained")
        gs = fig.add_gridspec(2, 2)
        a = fig.add_subplot(gs[0, :])
        b = fig.add_subplot(gs[1, 0])
        c = fig.add_subplot(gs[1, 1])
        labels = (
            ["图像", "图像+目标", "图像+目标+其他", "精简特征", "端到端融合"]
            if zh
            else [
                "Image",
                "Image + target",
                "Image + target + others",
                "Compact features",
                "Joint fusion",
            ]
        )
        for g, color, label in zip(GROUPS, COLORS, labels):
            name = "ENS_" + g
            p = np.load(RUN / "results/curves" / (name + "_TEST_DIAGNOSTIC_PR.npz"))
            ix = np.linspace(
                0, len(p["recall"]) - 1, min(1200, len(p["recall"]))
            ).astype(int)
            a.plot(
                p["recall"][ix], p["precision"][ix], color=color, label=label, lw=1.4
            )
            op = one(name, "dev_recall_90")
            a.scatter(op.recall, op.precision, color=color, marker="*", s=55, zorder=4)
            fp = risk(name, "contact_bright_proxy_negative").FPR
            wr = risk(name, "weak_positive").recall
            b.scatter(fp, wr, color=color, s=35)
            b.annotate(
                label,
                (fp, wr),
                xytext={
                    "image": (0.65, 0.92),
                    "image_target": (0.18, 0.52),
                    "image_target_other": (0.18, 0.75),
                    "tabular_only": (0.40, 0.34),
                    "image_target_other_tabular": (0.18, 0.10),
                }[g],
                textcoords="axes fraction",
                fontsize=6,
                arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color},
            )
        a.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="召回率" if zh else "Recall",
            ylabel="精确率" if zh else "Precision",
        )
        a.set_title(
            "a 测试PR仅作诊断；星号为内部冻结门限"
            if zh
            else "a Test PR is diagnostic; stars use frozen internal thresholds",
            loc="left",
            fontweight="bold",
        )
        a.legend(ncol=3, fontsize=6, loc="lower left")
        b.set_xlabel("接触侧亮阴性代理组FPR" if zh else "Contact-bright negative proxy FPR")
        b.set_ylabel("弱阳性代理组召回率" if zh else "Weak-positive proxy recall")
        b.set_title(
            "b 冻结内部R90判定点" if zh else "b Frozen internal R90 points",
            loc="left",
            fontweight="bold",
        )
        for i, (g, color) in enumerate(zip(GROUPS, COLORS)):
            v = [
                one("N_" + g + "_s" + str(seed), "dev_recall_90").recall
                for seed in [0, 1, 2]
            ]
            c.scatter(np.full(3, i) + np.linspace(-0.1, 0.1, 3), v, color=color, s=18)
            c.plot([i - 0.2, i + 0.2], [np.mean(v)] * 2, color=color, lw=2)
        c.set_xticks(range(5), labels, rotation=25, ha="right")
        c.set_ylabel("实际外部总体召回" if zh else "Achieved external recall")
        c.set_ylim(0, 1)
        c.set_title(
            "c 三固定种子（线为均值）" if zh else "c Three fixed seeds (line: mean)",
            loc="left",
            fontweight="bold",
        )
        save(
            fig,
            "target_aware",
            lang,
            "外部n=437,480；曲线为三种子平均概率集成，散点为单种子；无患者显著性检验。"
            if zh
            else "External n=437,480; curves: 3-seed probability ensembles; dots: individual seeds; no patient-level significance claim.",
        )
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), layout="constrained")
        base = cal[(cal.level == "split") & (cal.unit == TEST) & cal.stratum.isna()]
        plotnames = list(
            dict.fromkeys(["H0", "H2", selected, "ENS_image_target_other_tabular"])
        )
        budgets = [0, 20, 50, 100]
        for i, name in enumerate(plotnames):
            q = base[base.model == name].groupby("budget_per_core")
            axes[0].errorbar(
                budgets,
                q.F1.mean().reindex(budgets),
                yerr=q.F1.std().reindex(budgets),
                color=COLORS[i],
                marker="o",
                ms=3,
                lw=1,
                label=name,
                capsize=2,
            )
            axes[2].plot(
                budgets,
                q.Brier.mean().reindex(budgets),
                color=COLORS[i],
                marker="o",
                ms=3,
                lw=1,
            )
        for i, t in enumerate(["TMA1_CD8", "TMA2_CD8", "TMA3_CD8"]):
            q = (
                cal[
                    (cal.model == selected)
                    & (cal.level == "context")
                    & (cal.unit == t)
                    & cal.stratum.isna()
                ]
                .groupby("budget_per_core")
                .recall
            )
            axes[1].errorbar(
                budgets,
                q.mean().reindex(budgets),
                yerr=q.std().reindex(budgets),
                color=COLORS[i],
                marker="o",
                ms=3,
                capsize=2,
                label=t.replace("_CD8", ""),
            )
        axes[1].axhline(0.9, color="#aaa", ls="--", lw=0.7)
        for ax in axes:
            ax.set_xlabel("每芯标注预算" if zh else "Labels per core")
            ax.set_xticks(budgets)
        axes[0].set_ylabel("留出集合F1" if zh else "Held-out F1")
        axes[1].set_ylabel("选定候选实际召回" if zh else "Selected candidate recall")
        axes[2].set_ylabel("Brier分数（越低越好）" if zh else "Brier score (lower is better)")
        axes[0].set_title(
            "a 隔离的目标标签模拟" if zh else "a Held-out calibration",
            loc="left",
            fontweight="bold",
        )
        axes[1].set_title(
            "b 逐来源表现" if zh else "b Source variation", loc="left", fontweight="bold"
        )
        axes[2].set_title(
            "c 概率误差" if zh else "c Probability error", loc="left", fontweight="bold"
        )
        axes[0].legend(fontsize=5)
        axes[1].legend(fontsize=6)
        save(
            fig,
            "calibration",
            lang,
            "每次留出n=433,280；误差条为3次标签抽样的标准差，不是患者置信区间。"
            if zh
            else "Held-out n=433,280 per repeat; error bars: SD of 3 label-sampling repeats, not patient confidence intervals.",
        )
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
        pools = ["drop_top_1pct", "lowest_70pct", "highest_70pct"]
        plabels = (
            ["去除其他最亮1%", "其他低70%", "其他高70%"]
            if zh
            else ["Drop brightest 1%", "Lowest other 70%", "Highest other 70%"]
        )
        for k, name in enumerate(["H0", "H1", "H2"]):
            q = comp[comp.model == name].set_index("pool").reindex(pools)
            axes[0].plot(
                range(3),
                q.mean_abs_score_delta,
                marker="o",
                label=name,
                color=COLORS[k],
            )
            axes[1].plot(
                range(3),
                q.prediction_change_fraction,
                marker="o",
                label=name,
                color=COLORS[k],
            )
        for ax in axes:
            ax.set_xticks(range(3), plabels, rotation=10)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("平均绝对评分变化" if zh else "Mean absolute score change")
        axes[1].set_ylabel("固定0.5预测改变比例" if zh else "Fraction changed at fixed 0.5")
        axes[0].set_title(
            "a 目标原像素不变" if zh else "a Target raw pixels unchanged",
            loc="left",
            fontweight="bold",
        )
        axes[1].set_title(
            "b 仅改变边界估计组成" if zh else "b Only estimation pools change",
            loc="left",
            fontweight="bold",
        )
        save(
            fig,
            "composition",
            lang,
            "n=300个训练/内部目标；原像素哈希不变；本图是测量诊断，不是独立泛化评价。"
            if zh
            else "n=300 training/internal targets; raw hashes unchanged; measurement diagnostic, not independent generalization.",
        )
    pd.DataFrame(exports).to_csv(RUN / "reports/FIGURE_EXPORTS.csv", index=False)
    savej(
        RUN / "reports/FIGURES_COMPLETE.json",
        {
            "figures": 4,
            "exports": len(exports),
            "backend": "Python only",
            "languages": ["zh", "en"],
            "code_sha256": sha(__file__),
            "visual_QA": "pending local inspection",
        },
    )
    print("FIGURES_COMPLETE", len(exports), flush=True)


if __name__ == "__main__":
    main()
