"""Additional external and internal figures; purely descriptive, no selection."""
from make_figures import *


def external(lang):
    zh = lang == "zh"
    t = pd.read_csv(data / "ALL_EXTERNAL_SIX_METRICS.csv")
    t = t[
        (t.threshold_mode == "dev_FPR01_each_WSI")
        & t.model.str.contains("ENS")
        & (~t.model.str.startswith("OLD_"))
    ]
    sources = sorted(t[t.level == "source"].unit.unique())
    groups = list(
        t[["model", "method"]].drop_duplicates().itertuples(index=False, name=None)
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), layout="constrained")
    metrics = ["F1", "AP", "weak_recall", "FPR"]
    cn = ["F1", "AP", "弱观测信号召回率", "实际 FPR"]
    en = ["F1", "AP", "Weak observed-signal recall", "Actual FPR"]
    for k, metric in enumerate(metrics):
        ax = axes.flat[k]
        for j, (model, method) in enumerate(groups):
            g = (
                t[(t.level == "source") & (t.model == model) & (t.method == method)]
                .set_index("unit")
                .reindex(sources)
            )
            label = (
                model
                if method in ["new_continuation", "reused_baseline"]
                else method + " / " + model
            )
            ax.plot(
                range(len(sources)),
                g[metric],
                marker=["o", "s", "^", "D", "v"][j % 5],
                ms=4,
                lw=1,
                color=C[j % 4],
                label=label,
            )
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([x.split("/")[-1] for x in sources], rotation=15)
        ax.set_ylabel(cn[k] if zh else en[k])
        ax.grid(axis="y", alpha=0.15)
        labels(ax, "abcd"[k])
        if k == 3:
            ax.axhline(0.01, color="#777777", ls=":", lw=0.8)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(
        "已知探索性外部基准：分来源概率集成"
        if zh
        else "Known exploratory external benchmark: source-level ensembles",
        fontsize=10,
    )
    save(
        fig,
        "F6_external_sources",
        lang,
        "全量437,480个已标注细胞，3来源42core。仅开发通过的分支进入外部。各模型沿用自己冻结的内部逐WSI≤1%FPR工作点，虚线不是外部保证。TMA1/2同样本，core不是独立患者；线仅连接离散来源便于辨识。"
        if zh
        else "All437,480 labelled cells, three sources and42cores. Development-passed branches only. Each model retains its own frozen internal per-WSI1%FPR operating point; the dotted line is not an external guarantee. TMA1/2 share samples; cores are not independent patients. Lines only join discrete sources for identification.",
    )


def internal(lang):
    p = data / "CONTINUATION_INTERNAL_METRICS.csv"
    if not p.exists():
        return
    zh = lang == "zh"
    t = pd.read_csv(p)
    t = t[(t.level == "pooled") & (t.threshold_mode == "dev_FPR01_each_WSI")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), layout="constrained")
    variants = ["control", "physics"]
    metrics = ["F1", "AP", "weak_recall"]
    for k, metric in enumerate(metrics):
        ax = axes[k]
        for j, v in enumerate(variants):
            y = t[t.model.str.startswith("NEW_" + v + "_s")][metric].to_numpy()
            ens = t[t.model == "NEW_ENS_" + v][metric].iloc[0]
            ax.scatter(
                j + np.linspace(-0.07, 0.07, len(y)),
                y,
                c=C[j],
                s=22,
                label="种子" if zh and j == 0 else "Seeds" if j == 0 else None,
            )
            ax.plot([j - 0.15, j + 0.15], [y.mean()] * 2, color="black", lw=1)
            ax.scatter(
                j,
                ens,
                color=C[j],
                marker="D",
                s=40,
                facecolors="none",
                label="概率集成" if zh and j == 0 else "Ensemble" if j == 0 else None,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["配对继续训练", "技术一致性"] if zh else ["Matched control", "Consistency"],
            rotation=15,
        )
        ax.set_ylabel(
            ["F1", "AP", "弱观测信号召回率"][k] if zh else ["F1", "AP", "Weak-signal recall"][k]
        )
        labels(ax, "abc"[k])
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(fontsize=6)
    fig.suptitle(
        "完整内部开发检查：种子与集成分开呈现"
        if zh
        else "Full internal development check: seeds and ensemble",
        fontsize=10,
    )
    save(
        fig,
        "F7_continuation_internal",
        lang,
        "539,132细胞；各模型的冻结内部FPR工作点。3点是模型种子，横线为均值，空菱形是概率集成；不是患者重复。内部集用于预注册门槛，不能称未接触验证集。"
        if zh
        else "539,132cells at eachmodel frozen internal FPR operating point. Three dots:modelseeds;bar:mean;open diamond:probabilityensemble, not patient repeats. Internal data serve predeclared development gates, not untouched validation.",
    )


if __name__ == "__main__":
    for lang in ["zh", "en"]:
        style(lang)
        external(lang)
        internal(lang)
    print("ADDITIONAL_FIGURES_COMPLETE")
