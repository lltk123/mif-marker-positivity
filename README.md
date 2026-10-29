# mIF Marker Positivity — CD8

基于固定实例分割的逐细胞 CD8 阴阳识别：从原始荧光图像提取强度与空间表达特征，比较 SCIMAP GMM、HGB 特征模型、Nimbus 和 TabM。当前研究数据为 **PhenoBIC-Data**。

## Overview

细胞通过 `dataset_id + image_id + cell_id + marker` 对齐；沿用官方训练、内部验证与外部 TMA 划分。mask 为 **expanded_nucleus**，边缘只代表近似细胞边缘。原始数据、权重和全量逐细胞结果保存在研究服务器；本仓库提供可读代码、固定配置、中文 MD 报告、汇总 CSV 和图表。

## Experiment Status

| 实验 | 状态 | 报告 |
|---|---|---|
| SCIMAP GMM | 已完成；后续比较复用原分数 | [GMM](analyses/cd8_scimap_gmm_20261023/ANALYSIS_REPORT.md) |
| 形态与空间特征、原 8/90 项模型 | 已完成 | [形态与空间分析](analyses/cd8_morphospatial_20261023/ANALYSIS_REPORT.md) |
| E1–E4 与亮度／旋转／mask 稳定性 | 已完成 | [E1–E4](analyses/cd8_e1_e4_stability_20261023/ANALYSIS_REPORT.md) |
| Nimbus_pretrained、Nimbus_finetuned | 已完成全量推理、微调与评价 | [Nimbus](analyses/cd8_nimbus_20261024/ANALYSIS_REPORT.md) |
| TabM A/B/C 与匹配 HGB | 已完成每组 3 种子、全量评价与开发案例核对 | [TabM](analyses/cd8_tabm_20261029/ANALYSIS_REPORT.md) |

**Nimbus 状态更新（2026-10-24）：** 首轮预训练和微调基线已经完成，当前没有正在运行的训练。预训练来源独立性、跨平台输入尺度和进一步任务适配仍是后续待研究事项，未写成已完成实验。Pan-M 继续封存，本仓库不提供或使用其图像、标签、分布及预测。

**TabM 状态更新（2026-10-29）：** 已完成官方 TabM + PLE 从头训练及 A/B/C 消融。固定门限下 C 的外部平均 F1 为 0.5423，低于 GMM 的 0.7610；空间特征的收益集中于部分来源，TMA1/2 和弱阳性漏检仍有明显问题。见[关键结论与独立核算](analyses/cd8_tabm_20261029/independent_review/INDEPENDENT_REVIEW.md)。

## Data and Model Access

- [PhenoBIC-Data](https://huggingface.co/datasets/mittal-research-lab/PhenoBIC-Data)：自行取得原始图像、实例 mask 与 ground_truth；本仓库不重新分发。
- [Nimbus 官方实现](https://github.com/angelolab/Nimbus-Inference/tree/241281ef85ba34d9857cc03886cc45aedd0f0b34)与[固定权重版本](https://huggingface.co/JLrumberger/Nimbus-Inference/tree/3affc9e774d68408701e44ed6136b1e13503c639)：版本和 SHA-256 见 [Nimbus 协议](analyses/cd8_nimbus_20261024/NIMBUS_PROTOCOL.json)。
- 大型产物的位置与排除规则见 [数据与发布范围](docs/DATA_POLICY.md)。

## Installation

不同实验使用独立环境，避免把旧统计环境和 Nimbus 的 CUDA 依赖混装。完整环境记录与服务器复现步骤见 [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。

仅使用单图特征接口时，原验证环境为 Python 3.8.20：

```bash
python3.8 -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements-features.txt
```

## Example Usage

以已有 CD8 图像和整数实例 mask 提取特征；不重新分割：

```bash
python analyses/cd8_morphospatial_20261023/scripts/extract_single.py \
  --image /path/to/CD8.tiff \
  --mask /path/to/instances.tiff \
  --dataset-id PhenoBIC-Data \
  --image-id example/core_01 \
  --config docs/feature_config.example.json \
  --output /path/to/output/features.parquet --csv
```

输出一行一个原实例，包含数值特征、QC、显式 ID 与运行配置。默认不会覆盖已有输出。GMM、E1–E4、Nimbus 的批处理和微调命令分别保留在实验 README 中；这些完整流程依赖服务器上的数据、上游实现和缓存，克隆仓库本身不等于已具备全部输入。

## Results

外部 CD8 TMA 的 **437,480** 个有标签细胞，统一严格 `score > 0.5`：

| 方法 | F1 | AP |
|---|---:|---:|
| GMM | 0.7610 | 0.8302 |
| 原 8 项强度 | 0.5333 | 0.5476 |
| 原 90 项特征 | 0.5453 | 0.6134 |
| E1 | 0.8490 | 0.9487 |
| E2 | 0.8576 | 0.9584 |
| E3 | 0.8377 | 0.9400 |
| E4 | 0.8536 | 0.9493 |
| Nimbus_pretrained | 0.5885 | 0.6400 |
| Nimbus_finetuned | 0.5611 | 0.5579 |
| TabM A（3 种子均值） | 0.5325 | 0.7590 |
| TabM B（3 种子均值） | 0.5328 | 0.6877 |
| TabM C（3 种子均值） | 0.5423 | 0.6706 |

TabM 行为三个固定随机种子的指标均值，标准差在源表中；没有按测试表现选择种子。来源：[results_overview.csv](docs/results_overview.csv)。每行保留源 CSV 与筛选条件；完整报告另外给出查准率、查全率、混淆计数、逐图结果和门限开发方式。

当前所列实验中 E2 的外部 F1 最高；Nimbus 首轮微调没有改善外部总体 F1。模型监督条件不同：GMM 为无监督基线，HGB 使用 PhenoBIC 训练标签，Nimbus 继承作者监督预训练；Nimbus 内部验证还参与检查点和补充门限选择。结果是本任务的配对描述，不代表论文全部指标复现或无条件的跨域优越性。

![E1–E4 performance](analyses/cd8_e1_e4_stability_20261023/figures/01_performance_fixed_zh.png)

## Directory Structure

```text
analyses/
  cd8_scimap_gmm_20261023/      # GMM 提取、评分与汇总
  cd8_morphospatial_20261023/   # 19 项形态、CD8 空间特征、8/90 项模型
  cd8_e1_e4_stability_20261023/ # 共同校准、E1–E4、冻结模型扰动检查
  cd8_paper_baseline_20261024/  # 统一键、标签对齐与评估接口
  cd8_nimbus_20261024/         # 官方 Nimbus 适配、推理、微调与评价
  cd8_tabm_20261029/           # 官方 TabM、A/B/C 三种子消融、匹配 HGB
docs/                         # 复现、来源、发布清单与结果索引
tools/                        # 发布边界与仓库检查
tests/                        # 发布工具的独立检查
```

原实验脚本的发布副本经过格式化并补充说明，计算逻辑未改；AST 对照与文件哈希见 [代码整理核验](docs/publication/code_readability_audit.csv)。服务器原代码、缓存和结果保持原样。历史脚本包含固定路径，使用前请阅读复现说明，不要用格式化文件覆盖已冻结运行来强行复用缓存。

## References

- [PhenoBIC](https://github.com/Shachi-Mittal-Lab/PhenoBIC-qupath-extension)
- [SCIMAP rescale](https://scimap.xyz/Functions/pp/rescale/)
- [Nimbus paper](https://www.nature.com/articles/s41592-025-02826-9)
- [TabM 固定官方版本](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4)

第三方源码与权重遵循各自许可，见 [THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。仓库说明结构参考 [ROSIE](https://gitlab.com/enable-medicine-public/rosie)；未引入 ROSIE 模型或源码。
