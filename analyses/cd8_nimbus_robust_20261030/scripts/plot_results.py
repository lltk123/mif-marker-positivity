"""Editable figures backed by explicit CSV source tables."""
from common import *
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def style(lang="en"):
    family = "DejaVu Sans"
    font = os.environ.get("NIMBUS_FONT_PATH")
    if lang == "zh" and not font:
        import subprocess

        font = subprocess.check_output(
            ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True
        ).strip()
    if font:
        font_manager.fontManager.addfont(font)
        family = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )


def save(fig, name, lang="en"):
    destination = R / "figures" / lang
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        fig.savefig(
            destination / (name + "." + extension), dpi=220, bbox_inches="tight"
        )
    plt.close(fig)


def main():
    data = pd.read_csv(R / "results/all_metrics.csv")
    output = R / "figures/source_data"
    output.mkdir(parents=True, exist_ok=True)
    style()
    for mode in ["official_default", "validation_selected"]:
        table = data.query(
            "level=='split' and split!='training_WSIs' and threshold_mode==@mode"
        )
        table.to_csv(output / (mode + "_performance.csv"), index=False)
        models = [
            m
            for m in ["GMM", MODEL_PRETRAINED, MODEL_FINETUNED]
            if m in set(table.model)
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
        for ax, split in zip(axes, ["internal_val_WSIs", "external_val_TMA_cores"]):
            for j, model in enumerate(models):
                row = table.query("split==@split and model==@model").iloc[0]
                xx = np.arange(4) + (j - (len(models) - 1) / 2) * 0.24
                ax.bar(
                    xx,
                    [row[k] for k in ["precision", "recall", "F1", "AP"]],
                    width=0.23,
                    label=model.replace("Nimbus_outerP1P99_", "Nimbus "),
                )
            ax.set_xticks(range(4), ["Precision", "Recall", "F1", "AP"])
            ax.set_ylim(0, 1)
            ax.set_title(
                "Internal development"
                if split.startswith("internal")
                else "External TMA"
            )
        axes[-1].legend(fontsize=8)
        fig.suptitle(mode + " — outer P1/P99 input")
        save(fig, "performance_" + mode)
    units = data.query(
        "level=='image' and split=='external_val_TMA_cores' and threshold_mode=='official_default'"
    )
    units.to_csv(output / "external_core_performance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    for model in [MODEL_PRETRAINED, MODEL_FINETUNED]:
        subset = units.query("model==@model").sort_values("image_id")
        ax.plot(
            np.arange(len(subset)),
            subset.F1,
            marker=".",
            label=model.replace("Nimbus_outerP1P99_", "Nimbus "),
        )
    ax.set(
        xlabel="TMA core (sorted image_id; see source CSV)", ylabel="F1", ylim=(0, 1)
    )
    ax.legend()
    save(fig, "per_core_F1")
    dump(
        R / "figures/FIGURES_MANIFEST.json",
        dict(
            status="complete",
            source_csvs=[str(p.relative_to(R)) for p in output.glob("*.csv")],
            figures=[str(p.relative_to(R)) for p in (R / "figures/en").glob("*")],
        ),
    )


if __name__ == "__main__":
    main()
