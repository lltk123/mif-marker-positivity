"""Move performance legends outside data; preserve source tables and original figures."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    marker = R / "provenance/figure_layout_finalization.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if all(
            sha(R / path) == value for path, value in previous["output_sha256"].items()
        ):
            assert previous["script_sha256"] == sha(Path(__file__))
            print("VERIFIED_FIGURE_LAYOUT")
            return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = R / "provenance/figure_layout_original" / stamp
    backup.mkdir(parents=True, exist_ok=False)
    if marker.exists():
        shutil.copy2(marker, backup / marker.name)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    inputs, outputs, before = {}, {}, {}
    labels = {
        "official_default": "Fixed score > 0.5",
        "validation_selected": "Validation-selected threshold",
    }
    for mode, title in labels.items():
        source = R / "figures/source_data" / (mode + "_performance.csv")
        inputs[str(source.relative_to(R))] = sha(source)
        table = pd.read_csv(source)
        models = [
            m
            for m in [
                "GMM",
                "Nimbus_outerP1P99_pretrained",
                "Nimbus_outerP1P99_finetuned",
            ]
            if m in set(table.model)
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, split in zip(axes, ["internal_val_WSIs", "external_val_TMA_cores"]):
            for index, model in enumerate(models):
                row = table.loc[(table.model == model) & (table.split == split)]
                assert len(row) == 1
                values = row.iloc[0][["precision", "recall", "F1", "AP"]].to_numpy(
                    float
                )
                ax.bar(
                    np.arange(4) + (index - (len(models) - 1) / 2) * 0.24,
                    values,
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
        handles, names = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            names,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(models),
            frameon=False,
        )
        fig.suptitle(title + " — outer P1/P99 input", y=1.045, fontsize=11)
        fig.subplots_adjust(top=0.79, bottom=0.14, wspace=0.18)
        for suffix in ["png", "svg"]:
            path = R / "figures/en" / ("performance_" + mode + "." + suffix)
            before[str(path.relative_to(R))] = sha(path)
            shutil.copy2(path, backup / path.name)
            fig.savefig(path, dpi=220, bbox_inches="tight")
            outputs[str(path.relative_to(R))] = sha(path)
        plt.close(fig)
    readme = R / "README.md"
    note = "\n性能图最终排版复现：`python audit_scripts/finalize_figure_layout.py`；仅将图例移出数据区，数值源表不变。\n"
    original = readme.read_text(encoding="utf-8")
    if note.strip() not in original:
        shutil.copy2(readme, backup / "README.before_figure_layout.md")
        readme.write_text(original + note, encoding="utf-8")
    state = dict(
        status="complete",
        created=datetime.datetime.now().isoformat(),
        script_sha256=sha(Path(__file__)),
        source_sha256=inputs,
        original_figure_sha256=before,
        output_sha256=outputs,
        readme_sha256_after=sha(readme),
        backup=str(backup.relative_to(R)),
        scope="Legend layout only. No numerical source table, model, prediction, protocol or computational script changed.",
    )
    marker.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("FIGURE_LAYOUT_FINALIZED")


if __name__ == "__main__":
    main()
