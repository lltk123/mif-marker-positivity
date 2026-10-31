"""Add missing Chinese quantitative figures from the existing, unchanged source CSVs."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

R = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "GMM": "GMM",
    "Nimbus_outerP1P99_pretrained": "Nimbus 预训练",
    "Nimbus_outerP1P99_finetuned": "Nimbus 微调",
}
FORMAL = [
    "performance_official_default",
    "performance_validation_selected",
    "per_core_F1",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-path", type=Path)
    parser.add_argument(
        "--confirm-visual-review",
        action="store_true",
        help="Use only after opening and inspecting the three generated Chinese figures.",
    )
    args = parser.parse_args()
    marker = R / "provenance/bilingual_figures_complete.json"
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        assert state["script_sha256"] == sha(Path(__file__))
        for path, expected in {
            **state["source_sha256"],
            **state["figure_sha256"],
        }.items():
            assert sha(R / path) == expected, path
        if args.confirm_visual_review:
            state.update(
                status="complete",
                visual_review_completed=True,
                visual_review_time=datetime.datetime.now().isoformat(),
                visual_review="All three Chinese quantitative PNGs inspected: legible Chinese text, complete labels, no data-obscuring legends; SVG text remains editable.",
            )
            save_json(marker, state)
        print(state["status"])
        return
    assert not args.confirm_visual_review, "Generate figures before visual review."
    source_paths = [
        R / "figures/source_data" / x
        for x in [
            "official_default_performance.csv",
            "validation_selected_performance.csv",
            "external_core_performance.csv",
        ]
    ]
    source_hashes = {str(p.relative_to(R)): sha(p) for p in source_paths}
    metrics_hash = sha(R / "results/all_metrics.csv")
    report_hash = sha(R / "ANALYSIS_REPORT.md")
    english_hashes = {
        str(p.relative_to(R)): sha(p)
        for p in (R / "figures/en").glob("*")
        if p.suffix in [".png", ".svg"]
    }
    contract = dict(
        archetype="quantitative grid",
        backend="Python/matplotlib",
        conclusion="Present the existing pooled performance and per-core variation without changing model selection or numerical values.",
        evidence={
            "performance_official_default": "GMM and two Nimbus models at strict score > 0.5; internal and external panels.",
            "performance_validation_selected": "Two Nimbus models at thresholds selected on the internal development set.",
            "per_core_F1": "The same 42 external cores in image_id order; descriptive paired model performance.",
        },
        export={
            "formats": ["PNG", "editable-text SVG"],
            "languages": ["zh", "en"],
            "dpi": 220,
            "dimensions_inches": {"performance": [10, 4], "per_core": [9, 4]},
        },
        statistics="One Nimbus seed; descriptive cell-pooled metrics, no confidence intervals or cell-level significance tests. Cores are not claimed to be independent patients.",
        risks=[
            "Do not confuse the internal threshold-development result with unbiased validation.",
            "No numerical recomputation, model change, external threshold selection or image manipulation.",
        ],
        correction="The original plotting script produced quantitative figures only in English. The missing Chinese versions are added; prior English figures remain byte-identical.",
    )
    save_json(R / "provenance/BILINGUAL_FIGURE_CONTRACT.json", contract)
    font = args.font_path or Path(
        subprocess.check_output(
            ["fc-match", "-f", "%{file}", "Noto Sans CJK SC"], text=True
        ).strip()
    )
    assert font.is_file(), font
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=font).get_name(),
            "font.size": 9,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )
    target = R / "figures/zh"

    def save_figure(fig, name):
        for extension in ["png", "svg"]:
            path = target / (name + "." + extension)
            assert (
                not path.exists()
            ), "Refusing to replace an untracked existing figure: " + str(path)
            fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    titles = {"official_default": "固定评分 > 0.5", "validation_selected": "内部验证所选门限"}
    for mode, title in titles.items():
        table = pd.read_csv(R / "figures/source_data" / (mode + "_performance.csv"))
        models = [m for m in MODEL_NAMES if m in set(table.model)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, split in zip(axes, ["internal_val_WSIs", "external_val_TMA_cores"]):
            for index, model in enumerate(models):
                row = table.loc[(table.model == model) & (table.split == split)]
                assert len(row) == 1
                ax.bar(
                    np.arange(4) + (index - (len(models) - 1) / 2) * 0.24,
                    row.iloc[0][["precision", "recall", "F1", "AP"]].to_numpy(float),
                    width=0.23,
                    label=MODEL_NAMES[model],
                )
            ax.set_xticks(range(4), ["查准率", "查全率", "F1", "平均精确率"])
            ax.set_ylim(0, 1)
            ax.set_title("内部开发集" if split.startswith("internal") else "外部组织芯")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(models),
            frameon=False,
        )
        fig.suptitle(title + " — 外层 P1/P99 共同归一化", y=1.045, fontsize=11)
        fig.subplots_adjust(top=0.79, bottom=0.14, wspace=0.18)
        save_figure(fig, "performance_" + mode)
    cores = pd.read_csv(source_paths[-1])
    fig, ax = plt.subplots(figsize=(9, 4))
    for model in ["Nimbus_outerP1P99_pretrained", "Nimbus_outerP1P99_finetuned"]:
        subset = cores.loc[cores.model == model].sort_values("image_id")
        assert len(subset) == 42
        ax.plot(np.arange(len(subset)), subset.F1, marker=".", label=MODEL_NAMES[model])
    ax.set(xlabel="组织芯序号（按图像 ID 排序；见绘图源表）", ylabel="F1", ylim=(0, 1))
    ax.legend()
    save_figure(fig, "per_core_F1")
    assert all(sha(R / path) == value for path, value in english_hashes.items())
    assert (
        sha(R / "results/all_metrics.csv") == metrics_hash
        and sha(R / "ANALYSIS_REPORT.md") == report_hash
    )
    groups = {p.stem for p in (R / "figures/en").glob("*.png")}
    assert len(groups) == 8
    rows, output_hashes = [], {}
    for language in ["zh", "en"]:
        assert {p.stem for p in (R / "figures" / language).glob("*.png")} == groups
        assert {p.stem for p in (R / "figures" / language).glob("*.svg")} == groups
        for group in sorted(groups):
            for extension in ["png", "svg"]:
                path = R / "figures" / language / (group + "." + extension)
                if extension == "png":
                    with Image.open(path) as raster:
                        raster.verify()
                else:
                    assert "<text" in path.read_text(encoding="utf-8")
                relative = str(path.relative_to(R))
                output_hashes[relative] = sha(path)
                rows.append(
                    dict(
                        language=language,
                        figure=group,
                        format=extension,
                        path=relative,
                        passed=True,
                        bytes=path.stat().st_size,
                        sha256=output_hashes[relative],
                    )
                )
    pd.DataFrame(rows).to_csv(R / "figures/bilingual_coverage.csv", index=False)
    manifest = R / "figures/FIGURES_MANIFEST.json"
    backup = R / "provenance/FIGURES_MANIFEST.before_bilingual_completion.json"
    assert not backup.exists()
    shutil.copy2(manifest, backup)
    save_json(
        manifest,
        dict(
            status="complete",
            source_csvs=list(source_hashes),
            figures=sorted(output_hashes),
            languages=["zh", "en"],
            groups_per_language=8,
            bilingual_qa="provenance/bilingual_figures_complete.json",
        ),
    )
    readme = R / "README.md"
    note = (
        "\n中文正式图件补齐与两语覆盖检查：`python audit_scripts/complete_bilingual_figures.py`。"
        "检查三张中文 PNG 后运行 `python audit_scripts/complete_bilingual_figures.py --confirm-visual-review`。"
        "其后可重新运行 `python audit_scripts/validate_delivery.py` 与审阅包生成入口。\n\n"
        "中文：[固定0.5性能图](figures/zh/performance_official_default.svg)、"
        "[验证门限性能图](figures/zh/performance_validation_selected.svg)、"
        "[逐组织芯F1](figures/zh/per_core_F1.svg)。对应英文文件保存在 `figures/en`；"
        "两语完整性见 `figures/bilingual_coverage.csv`。\n"
    )
    readme.write_text(readme.read_text(encoding="utf-8") + note, encoding="utf-8")
    save_json(
        marker,
        dict(
            status="generated_pending_visual_review",
            visual_review_completed=False,
            created=datetime.datetime.now().isoformat(),
            script_sha256=sha(Path(__file__)),
            source_sha256=source_hashes,
            figure_sha256=output_hashes,
            english_files_unchanged=True,
            groups_per_language=8,
            formats=["png", "svg"],
            generated_chinese_groups=FORMAL,
            font_file=str(font),
            metric_table_sha256_unchanged=metrics_hash,
            report_sha256_unchanged=report_hash,
            coverage_csv_sha256=sha(R / "figures/bilingual_coverage.csv"),
            manifest_sha256=sha(manifest),
        ),
    )
    print("GENERATED_CHINESE_FIGURES_AND_VERIFIED_BILINGUAL_COVERAGE")


if __name__ == "__main__":
    main()
