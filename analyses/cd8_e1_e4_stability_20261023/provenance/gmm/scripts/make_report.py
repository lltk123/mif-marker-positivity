"""Build the evidence-linked analysis report from saved numerical results."""

from pathlib import Path
import argparse, json, hashlib, html, platform, sys
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
A = args.root
R = A / "results/phenobic"
F = A / "figures"
F.mkdir(exist_ok=True)
metrics = pd.read_csv(R / "metrics_summary.csv")
units = pd.read_csv(R / "metrics_per_unit.csv")
macro = pd.read_csv(R / "metrics_macro.csv")
coverage = pd.read_csv(R / "annotation_coverage.csv")
status = json.loads((R / "STATUS.json").read_text())
verify = json.loads((R / "VERIFICATION.json").read_text())
curves = json.loads((R / "curve_index.json").read_text())
splits = ["training_WSIs", "internal_val_WSIs", "external_val_TMA_cores"]
colors = ["#416C9B", "#3F918A", "#B48552"]
styles = ["-", "--", "-."]
labels = {
    "zh": ["原始训练 WSI", "内部验证 WSI", "外部验证 TMA"],
    "en": [
        "Original training WSI",
        "Internal validation WSI",
        "External validation TMA",
    ],
}
contract = {
    "question": "Quantify ranking quality and fixed automatic-gate errors for CD8 in the original PhenoBIC partitions.",
    "conclusion": "Strong discrimination coexists with excess positive calls; external TMA3 shows the largest precision shortfall.",
    "archetype": "quantitative grid",
    "backend": "Python / matplotlib only",
    "panels": {
        "a": "ROC curves: ranking across original splits",
        "b": "PR curves with prevalence reference: performance under class imbalance",
        "c": "F1 per independent gating unit, equal-unit mean and cell-pooled F1",
        "d-f": "Confusion matrices: false-positive and false-negative counts and within-truth-class percentages",
    },
    "size_mm": [183, 140],
    "score": "SCIMAP rescaled continuous score, not probability",
    "threshold": "score > 0.5",
    "n_definition": "Cells for pooled metrics; WSI/core for unit scatter. Cores are not asserted to be independent patients.",
    "error_bars": "None. Each dot is one gating unit; diamond is arithmetic unit mean.",
    "inferential_statistics": "No p-values or confidence intervals; descriptive comparison only.",
    "source_data": [
        "results/phenobic/metrics_summary.csv",
        "results/phenobic/metrics_per_unit.csv",
        "results/phenobic/*_curves.npz",
    ],
    "image_integrity": "No raw image pixels displayed or modified.",
    "formats": [
        "SVG editable text",
        "PDF embedded TrueType text",
        "PNG 300 dpi",
        "TIFF 600 dpi",
    ],
}
(A / "provenance/figure_contract.json").write_text(json.dumps(contract, indent=2))
font_paths = {
    "zh": Path("C:/Windows/Fonts/msyh.ttc"),
    "en": Path("C:/Windows/Fonts/arial.ttf"),
}
for lang in ["zh", "en"]:
    if font_paths[lang].exists():
        font_manager.fontManager.addfont(str(font_paths[lang]))
        family = font_manager.FontProperties(fname=str(font_paths[lang])).get_name()
    else:
        family = "Noto Sans CJK SC" if lang == "zh" else "DejaVu Sans"
        font_manager.findfont(family, fallback_to_default=False)
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 5.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )
    fig, axs = plt.subplots(2, 3, figsize=(183 / 25.4, 140 / 25.4))
    fig.subplots_adjust(
        left=0.082, right=0.985, bottom=0.115, top=0.91, wspace=0.42, hspace=0.52
    )
    axroc, axpr, axf1 = axs[0]
    for j, sp in enumerate(splits):
        m = metrics[(metrics.level == "split") & (metrics.group == sp)].iloc[0]
        ci = next(c for c in curves if c["level"] == "split" and c["group"] == sp)
        z = np.load(R / ci["file"])
        # Keep endpoint and quantile-spaced curve vertices for compact editable vector exports. Metrics use all observations.
        ridx = np.unique(
            np.linspace(0, len(z["fpr"]) - 1, min(2500, len(z["fpr"]))).astype(int)
        )
        pidx = np.unique(
            np.linspace(0, len(z["recall"]) - 1, min(3000, len(z["recall"]))).astype(
                int
            )
        )
        axroc.plot(
            z["fpr"][ridx],
            z["tpr"][ridx],
            color=colors[j],
            ls=styles[j],
            lw=1.2,
            label=labels[lang][j] + f" ({m.ROC_AUC:.3f})",
        )
        axpr.plot(
            z["recall"][pidx],
            z["precision"][pidx],
            color=colors[j],
            ls=styles[j],
            lw=1.2,
            label=labels[lang][j] + f" ({m.AP:.3f})",
        )
        axpr.axhline(m.true_prevalence, color=colors[j], ls=":", lw=0.6, alpha=0.65)
        v = units.loc[units.split == sp, "F1"].to_numpy()
        jitter = np.random.RandomState(100 + j).uniform(-0.15, 0.15, len(v))
        axf1.scatter(
            j + jitter,
            v,
            s=12,
            alpha=0.65,
            color=colors[j],
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )
        axf1.scatter(
            j,
            v.mean(),
            marker="D",
            s=32,
            facecolor=colors[j],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        axf1.plot([j - 0.24, j + 0.24], [m.F1, m.F1], color="black", lw=1, zorder=4)
        axf1.text(j, 1.035, f"n = {len(v)}", ha="center", fontsize=6)
        ax = axs[1, j]
        cm = np.array([[m.TN, m.FP], [m.FN, m.TP]], float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for y in range(2):
            for x in range(2):
                ax.text(
                    x,
                    y,
                    f"{int(cm[y,x]):,}\n({100*norm[y,x]:.1f}%)",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if norm[y, x] > 0.55 else "#17212B",
                )
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(length=0)
        ax.set_xticklabels(
            ["预测阴性", "预测阳性"] if lang == "zh" else ["Predicted −", "Predicted +"]
        )
        ax.set_yticklabels(
            ["人工阴性", "人工阳性"] if lang == "zh" else ["Reference −", "Reference +"]
        )
        ax.set_title(
            labels[lang][j] + f"\nn = {int(m.n_evaluated):,}", fontsize=7, pad=8
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
    axroc.plot([0, 1], [0, 1], color="#aaaaaa", ls=":", lw=0.7)
    for ax in [axroc, axpr]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    axroc.set_title("ROC 曲线（AUC）" if lang == "zh" else "ROC curves (AUC)")
    axroc.set_xlabel("假阳性率" if lang == "zh" else "False-positive rate")
    axroc.set_ylabel("真阳性率" if lang == "zh" else "True-positive rate")
    axroc.legend(loc="lower right", handlelength=1.7)
    axpr.set_title("PR 曲线（AP）" if lang == "zh" else "PR curves (AP)")
    axpr.set_xlabel("查全率 / 召回率" if lang == "zh" else "Recall")
    axpr.set_ylabel("查准率 / 精确率" if lang == "zh" else "Precision")
    axpr.legend(loc="lower left", bbox_to_anchor=(0, 0.18), handlelength=1.7)
    axf1.set_title("各 WSI / 组织芯的 F1" if lang == "zh" else "F1 across WSI / cores")
    axf1.set_ylabel("F1")
    axf1.set_ylim(0, 1.10)
    axf1.set_yticks([0, 0.5, 1])
    axf1.set_xlim(-0.5, 2.5)
    axf1.set_xticks(range(3))
    axf1.set_xticklabels(
        ["训练", "内部验证", "外部 TMA"]
        if lang == "zh"
        else ["Training", "Internal", "External"]
    )
    axf1.text(
        0.5,
        -0.26,
        "点：各单位  ◆：单位均值  —：细胞汇总"
        if lang == "zh"
        else "Dots: units; diamonds: unit mean\nBars: cell-pooled F1",
        transform=axf1.transAxes,
        ha="center",
        fontsize=5.4,
    )
    for i, ax in enumerate(axs.flat):
        ax.text(
            -0.19,
            1.08,
            chr(97 + i),
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
        )
    fig.suptitle(
        "PhenoBIC · CD8 自动 GMM 阈值评估"
        if lang == "zh"
        else "PhenoBIC · CD8 automatic GMM gating",
        fontsize=11,
        y=0.984,
        fontweight="bold",
    )
    caption = (
        "3 分量 GMM；每个 WSI / 组织芯独立拟合；score > 0.5 为阳性。混淆矩阵百分比按人工类别归一化。"
        if lang == "zh"
        else "3-component GMM per WSI / core; positive if score > 0.5. Confusion percentages are normalized within reference class."
    )
    fig.text(0.5, 0.025, caption, ha="center", fontsize=6.2)
    stem = F / ("CD8_GMM_summary_" + lang)
    fig.savefig(str(stem) + ".svg")
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png", dpi=300)
    fig.savefig(str(stem) + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("FIGURE", lang, flush=True)

friendly = {
    "training_WSIs": "原始训练 WSI",
    "internal_val_WSIs": "内部验证 WSI",
    "external_val_TMA_cores": "PhenoBIC 外部 TMA",
    "phenobic": "全部 PhenoBIC",
}
summary = metrics[metrics.level.isin(["split", "all"])].copy()
summary["划分"] = summary.group.map(friendly)
show = summary[
    [
        "划分",
        "n_evaluated",
        "n_positive",
        "ROC_AUC",
        "AP",
        "F1",
        "precision",
        "recall",
        "specificity",
    ]
].rename(
    columns={
        "n_evaluated": "评估细胞",
        "n_positive": "人工阳性",
        "ROC_AUC": "ROC-AUC",
        "precision": "查准率",
        "recall": "查全率",
        "specificity": "特异度",
    }
)
context = metrics[metrics.level == "context"][
    [
        "group",
        "n_evaluated",
        "ROC_AUC",
        "AP",
        "F1",
        "precision",
        "recall",
        "true_prevalence",
        "predicted_prevalence",
    ]
].copy()
context.columns = [
    "WSI / TMA",
    "评估细胞",
    "ROC-AUC",
    "AP",
    "F1",
    "查准率",
    "查全率",
    "人工阳性比例",
    "预测阳性比例",
]
macroshow = macro[
    (macro.level == "split")
    & macro.metric.isin(["ROC_AUC", "AP", "F1", "precision", "recall"])
][["group", "metric", "n_units", "mean", "std", "min", "max"]].copy()
macroshow["group"] = macroshow.group.map(friendly)
macroshow.columns = ["划分", "指标", "单位数", "等权均值", "单位间标准差", "最小值", "最大值"]
allrow = metrics[metrics.level == "all"].iloc[0]
server = (
    "/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_scimap_gmm_20261023"
)
readme = f"""PhenoBIC CD8 SCIMAP GMM analysis
Server analysis root: {server}
Primary report: CD8_GMM_ANALYSIS_REPORT.html
Human-readable annotation links: annotations/<original_split>/<context>/<unit>_CD8.csv.gz
Complete cell table index: ANNOTATION_INDEX.csv (server); results/phenobic/cell_table_index.csv
Original per-cell gzip tables are retained on the server and are not included in the compact local report bundle.
Key columns: cell_id (original segmentation ID), mean_CD8, rescale_score, prediction_CD8, ground_truth_original, eligible_binary_evaluation, record_status.
All segmented cells have predictions. Unannotated cells remain in the tables. No human labels entered GMM fitting.
Continuous rescale_score is a ranking score, not a calibrated probability. Binary CD8+ is strictly score > 0.5.
Reproduce using the analysis-root .venv/bin/python with scripts/extract_cd8_v2.py, scripts/analyze_cd8_v2.py, and scripts/verify_results.py. The verify script creates an exclusive ZIP; use a fresh output name on rerun.
Plots: python scripts/make_report.py --root <analysis-root>. Requires matplotlib and a Chinese font (Microsoft YaHei on Windows or Noto Sans CJK SC on Linux).
Pan-Multiplex Gold-Standard is sealed and excluded from this analysis. No encryption or filesystem access changes are implied.
"""
(A / "README_ANALYSIS.txt").write_text(readme, encoding="utf-8")
report = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PhenoBIC CD8 GMM 评估报告</title><style>body{{font:16px/1.7 'Microsoft YaHei',sans-serif;color:#21313f;margin:36px auto;max-width:1100px;padding:0 24px;background:white}}h1{{font-size:30px}}h2{{font-size:22px;margin-top:34px}}table{{border-collapse:collapse;font-size:13px;width:100%;margin:18px 0}}th,td{{padding:8px 10px;text-align:right;border-bottom:1px solid #dce3e9}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef3f6}}.lead{{font-size:19px}}.note{{padding:16px 20px;background:#f0f5f8;border-left:4px solid #416c9b}}code,pre{{font-size:13px;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;background:#f6f7f8;padding:14px}}img{{max-width:100%}}a{{color:#316594}}.scroll{{overflow-x:auto}}small{{color:#607080}}</style>
<h1>PhenoBIC：CD8 自动 GMM 阈值评估</h1><p><small>2026-10-23 · 仅 PhenoBIC · 固定流程基线 · Pan-M Gold 封存，留待后续盲测</small></p>
<p class="lead">连续强度评分具有较强判别能力，但自动阈值偏向较高召回率，并产生较多假阳性。全部 PhenoBIC 的 AUC 为 <b>{allrow.ROC_AUC:.4f}</b>，AP 为 <b>{allrow.AP:.4f}</b>，F1 为 <b>{allrow.F1:.4f}</b>；查准率 <b>{allrow.precision:.4f}</b>，查全率 <b>{allrow.recall:.4f}</b>。</p>
<p>共对 <b>1,817,783</b> 个分割细胞生成 CD8 注释，使用其中全部 <b>1,077,669</b> 条人工标签评估（阳性 57,488，阴性 1,020,181）。740,114 个未标注细胞保留预测。未发现标签缺失匹配、含糊标签或无效强度；48 个 GMM 均收敛。</p>
<div class="note">本报告中的“外部 TMA”是 PhenoBIC 自带的原始划分，并非 Pan-M Gold。本轮后续分析完全排除 Pan-M Gold。各目标 WSI / 组织芯使用自身未标注强度分布拟合 GMM，属于逐图像无监督校准；不是只在训练集学习一个固定数值阈值后迁移。</div>
<h2>1. 主要结果</h2><p>下表为细胞汇总指标。AUC 指 ROC-AUC，AP 指 average precision；二者使用连续的 SCIMAP 重缩放分数。F1、查准率、查全率与特异度使用固定规则 <code>rescale_score &gt; 0.5</code>，没有利用人工标签选择阈值。</p><div class="scroll">{show.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)}</div>
<img src="figures/CD8_GMM_summary_zh.png" alt="CD8 GMM ROC、PR、各单位 F1 与混淆矩阵"><p><small>图 a–b：曲线图例括号内分别为 AUC、AP；PR 点线为各划分阳性率。图 c：每点为一张 WSI 或一个组织芯，菱形为单位等权均值，横线为细胞汇总 F1。图 d–f：数量及按人工类别归一化的百分比。n 为细胞数或门控单位数，不等于独立患者数。</small></p>
<p><a href="figures/CD8_GMM_summary_zh.pdf">中文 PDF</a> · <a href="figures/CD8_GMM_summary_zh.svg">中文 SVG</a> · <a href="figures/CD8_GMM_summary_en.pdf">English PDF</a> · <a href="figures/CD8_GMM_summary_en.svg">English SVG</a> · <a href="figures/CD8_GMM_summary_en.png">English PNG</a></p>
<h2>2. WSI / TMA 分层比较</h2><div class="scroll">{context.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)}</div>
<p>TMA3 的查全率为 0.9933，但查准率为 0.5087：预测阳性比例 11.98%，高于人工阳性比例 6.14%。这说明强度排序良好并不保证自动阈值准确。整体预测阳性比例为 7.15%，人工阳性比例为 5.33%，相差 +1.82 个百分点。TMA1 则有较低召回率（0.8135），提示各批次的阈值偏差并不完全同向。</p>
<p>原始训练 3 张 WSI、内部验证 3 张 WSI、外部验证 42 个组织芯的单位等权统计如下。不同单位的大小与阳性比例不同，等权指标与细胞汇总指标回答不同问题。</p><div class="scroll">{macroshow.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)}</div>
<h2>3. 固定分析方法</h2><ol><li>按原始 cell ID，计算原始 CD8 通道在细胞 mask 内的像素算术均值；采用 float64 累加。未缩放图像、转换位深或修改 mask ID。</li><li>原始训练 3 张 WSI 分别门控。内部验证 131 个视野按原始 3 张 WSI 汇总；依据文件名中的全局坐标检查重叠，并使用非零 mask 的并集与唯一像素计数。实际检查未发现跨视野重复的非零细胞 ID，重叠差异均为单侧背景；未因重叠删除标注记录。外部 42 个组织芯分别门控。</li><li>调用固定 commit 的官方 <code>rescale</code> 原函数：<code>gate=None, log=True, gmm_components=3, random_state=0</code>。拟合前在 0.01 与 99.99 百分位截尾，再做 log1p；阈值为最高两个高斯分量均值的中点。使用全部分割细胞拟合，不使用人工类别。</li><li>逐个门控单位调用官方函数（单单位 <code>method='all'</code>）。源码多图像 <code>by_image</code> 的 gate 映射存在跨图像复用问题，已用合成数据复现；单单位调用规避此问题，源码未修改。</li><li>官方重缩放分数严格大于 0.5 标记为阳性，等于 0.5 标记为阴性。源码使用最接近理论阈值的观测值作为缩放分界，因此少量边界细胞与直接比较原始强度阈值不同，数量记录在阈值表中。</li><li>完成评分后才读取人工标签用于评估。<code>label</code> 是细胞 ID，<code>CD8</code> 的 pos/neg 是人工类别；关联键包括原始划分、WSI / 组织芯与 cell ID。</li></ol>
<p>这是单次固定 seed 的描述性基线，不包含监督模型训练、人工阈值调优或数据重划分。SCIMAP 分数不是校准后的阳性概率；跨单位汇总 AUC/AP 受单位内缩放影响，应结合单位指标解释。未将百万级细胞当作独立生物重复计算显著性；当前不报告 p 值或置信区间。</p>
<h2>4. 完整性与独立核查</h2><ul><li>1,077,669 / 1,077,669 条人工标签均与 mask 匹配，所有原始标签和未标注细胞均保留。</li><li>48 / 48 个 GMM 收敛，无无效强度导致的排除。</li><li>独立抽查 48 个门控单位各 1 个阴性和 1 个阳性细胞，共 96 个细胞：直接读取原始像素重新计数与计算均值，像素数完全一致，均值在 rtol=atol=1e-10 内一致。</li><li>核验混淆矩阵计数、评估总量与标注覆盖表一致。未修改原始 TIFF、mask 或 CSV。</li></ul>
<h2>5. 文件与复现</h2><pre>{server}</pre><p>逐细胞结果在服务器的 <code>annotations/&lt;原始划分&gt;/&lt;WSI或TMA&gt;/&lt;单位&gt;_CD8.csv.gz</code>；本地报告包只包含汇总与绘图数据。<code>ANNOTATION_INDEX.csv</code> 提供完整服务器路径。</p><ul><li><a href="results/phenobic/metrics_summary.csv">汇总指标 CSV</a>、<a href="results/phenobic/metrics_per_unit.csv">48 单位指标 CSV</a>、<a href="results/phenobic/metrics_macro.csv">单位等权统计 CSV</a></li><li><a href="results/phenobic/gmm_gates.csv">阈值表</a>、<a href="results/phenobic/gmm_gates.json">GMM 分量参数与诊断</a>、<a href="results/phenobic/intensity_by_ground_truth.csv">人工阴阳性强度统计</a></li><li><a href="results/phenobic/annotation_coverage.csv">标注覆盖审计</a>、<a href="results/phenobic/independent_pixel_checks.csv">96 细胞独立像素核查</a></li><li><a href="README_ANALYSIS.txt">复现说明</a>、<a href="provenance/runtime_versions.json">运行版本</a>、<a href="provenance/upstream_multi_image_gate_audit.json">上游映射问题复现实验</a></li></ul>
<p>关键逐细胞字段：<code>cell_id</code>、<code>mean_CD8</code>、<code>rescale_score</code>、<code>prediction_CD8</code>、<code>ground_truth_original</code>、<code>eligible_binary_evaluation</code>、<code>record_status</code>。</p>
<h2>6. 来源</h2><p>方法：<a href="https://scimap.xyz/Functions/pp/rescale/">SCIMAP rescale 官方文档</a>；<a href="https://github.com/ajitjohnson/scimap/blob/e8f1609fbcf838e265541983af7e63490e5c7a13/scimap/preprocessing/rescale.py">固定源码 e8f1609f…</a>（网页函数与源码 AST 一致）。数据：<a href="https://huggingface.co/datasets/mittal-research-lab/PhenoBIC-Data/tree/aeffd288d922ee5df4cbb414f3c1f77507ba0001">PhenoBIC-Data 固定版本 aeffd288…</a>。下载与原始文件完整性记录见项目 datasets/DATA_DOWNLOAD_REPORT.md。</p></html>"""
(A / "CD8_GMM_ANALYSIS_REPORT.html").write_text(report, encoding="utf-8")
(A / "provenance/plot_runtime.json").write_text(
    json.dumps(
        {
            "python": sys.version,
            "matplotlib": mpl.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "figure_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        indent=2,
    )
)
print("REPORT", A / "CD8_GMM_ANALYSIS_REPORT.html")
