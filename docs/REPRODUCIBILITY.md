# 复现与运行入口

## 两类文件

`analyses/*/scripts` 是已完成研究运行的可读源码副本。除排版和文档字符串外，发布时与服务器原源码的 AST 一致；[核验表](publication/code_readability_audit.csv)逐文件记录。

已有运行按源代码、配置、输入和权重哈希检查缓存。格式化也会改变源文件哈希，因此**不要把发布副本覆盖回已完成运行**。服务器原工程和独立 Git 发布工作区分开保存。历史 provenance 中的代码 commit/哈希描述当时实验；当前 Git commit 描述本次公开整理。

## 轻量特征接口

按根 README 安装 `requirements-features.txt`，调用 `extract_single.py`。这组依赖来自原 Python 3.8.20 特征环境（RUN_CONFIG.json 的 software 字段）；Nimbus 另用 Python 3.11.13 和自身依赖锁定记录。输入需要同尺寸的二维 CD8 与非负整数 mask，cell_id 必须来自 mask。可选 valid/tissue mask 通过相应参数传入；默认不从 CD8 反推组织区域。

## 已有服务器上的完整实验

以下命令针对已存在的数据和对应独立环境。它们不是本次发布再次执行的训练命令：

```bash
PROJECT=/mnt/disk3/mIHC/mif_marker_positivity_20261022

# E1-E4: compatibility checks run before reusing completed results.
cd "$PROJECT/analyses/cd8_e1_e4_stability_20261023"
"$PROJECT/analyses/cd8_morphospatial_20261023/.venv/bin/python" scripts/resume_checked.py evaluate

# Nimbus: retain the frozen official code, weight revision and normalization.
cd "$PROJECT/analyses/cd8_nimbus_20261024"
.venv/bin/python scripts/run_pipeline.py
```

- [GMM 用法](../analyses/cd8_scimap_gmm_20261023/README_ANALYSIS.txt)：`extract_cd8_v2.py` 和 `analyze_cd8_v2.py` 是最终版本；未加 v2 的脚本用于历史追溯。
- [形态与空间接口](../analyses/cd8_morphospatial_20261023/USAGE.md)：原始 19 项形态与 CD8 定位特征；最终训练脚本 `train_compare_v2.py`。
- [E1–E4 用法](../analyses/cd8_e1_e4_stability_20261023/README.md)：提取、OOF、最终模型、门限与固定扰动。
- [Nimbus 用法](../analyses/cd8_nimbus_20261024/README.md)：`prepare_data.py`、`inference.py`、`finetune.py`、`evaluate.py`。
- [Nimbus 环境记录](../analyses/cd8_nimbus_20261024/provenance/requirements.lock.txt)：原环境的完整记录，包含本地 editable 源码路径，不应直接当成任意机器可安装的 requirements 文件。

## 新机器或新运行

完整流程需要：官方 PhenoBIC 文件、旧阶段的全量特征／键表、固定第三方源码与权重、以及对应独立环境。Git 仓库不包含这些大型输入。历史脚本保留原路径与运行契约；迁移时需要在**新的运行目录**统一设置路径，重新冻结输入／代码／配置清单，重新准备兼容缓存。不要仅改一个路径后复用其他运行的完成标记。

Nimbus 直接推理与微调均已完成，但预训练图像来源独立性和物理像素尺寸仍存在报告所列限制。未执行 Pan-M、ROSIE 或论文全部队列实验。

## 发布验证

```bash
python tools/check_publication.py
python -m unittest discover -s tests
```

上述检查验证发布边界、链接和源码结构；历史数值正确性由各实验已保存的验证表和报告提供。本次整理不重训模型、不重算性能。

## 2026-10-29 新增入口

- [TabM A/B/C](../analyses/cd8_tabm_20261029/README.md)：固定官方实现、三种子训练和全量评价。
- [已完成校准实验汇总](../analyses/cd8_calibrated_results_summary_20261029/README.md)：只依赖 Python 标准库，可从仓库现有汇总表重建排名，无需原图或 GPU。
- [TabM 发布源码排版核验](publication/tabm_readability_20261029.csv)：当前发布副本与原运行代码 AST 一致，保留两者的独立哈希。

## 2026-10-31：共同校准与双模型重训练

本轮全量计算已完成。统一入口、分阶段命令和恢复条件见[本轮 README](../analyses/cd8_robust_comparison_20261030/README.md)。它使用原服务器上两个固定环境：形态环境计算共享特征，Nimbus 环境运行新模型及报告；不会自动取得全部数据或权重。

```bash
# 只重建仓库性能总览与固定门限排名；不访问原图或 GPU。
python tools/update_robust_overview.py

# 以下仅在原服务器、原冻结代码与依赖齐备时用于核验／恢复。
PROJECT=/mnt/disk3/mIHC/mif_marker_positivity_20261022
"$PROJECT/analyses/cd8_nimbus_20261024/.venv/bin/python" \
  "$PROJECT/analyses/cd8_robust_comparison_20261030/scripts/run_retraining.py" \
  --stage all --workers 4 --dry-run
```

- [校准接口](../analyses/cd8_robust_calibration_20261030/README.md)：原像素 P10/P90、跨实例 P1/P99、完整单位去重及特征重提。
- [TabM 训练／预测](../analyses/cd8_tabm_robust_20261030/README.md)：A/B/C 三种子、训练集预处理、成员概率聚合与匹配 HGB。
- [Nimbus 推理／微调](../analyses/cd8_nimbus_robust_20261030/README.md)：固定官方 V1 和任务适配输入尺度、全实例评分与切块核验。
- [正式参数与版本](../analyses/cd8_robust_comparison_20261030/RETRAIN_PROTOCOL.json)及各子目录冻结协议；[独立数值核查](../analyses/cd8_robust_comparison_20261030/provenance/independent_tabm_audit/AUDIT_REPORT.md)。

本轮发布副本仅按 Black 排版，并逐文件核对完全相同的 Python AST；[robust_readability_20261031.csv](publication/robust_readability_20261031.csv)记录原始文件、LF 规范化文件和发布源码 SHA。冻结配置及完成清单仍记录**实际运行时源码**的哈希。不要把格式化副本覆盖原运行目录后复用原完成标记；新机器运行须先配置数据和固定上游依赖，在新目录重新冻结源码与输入。服务器原运行与原审阅包均保持原样。

所有训练与性能计算均发生在本次发布之前；发布阶段只读取配置、报告、汇总表及图件，不重新训练、不读取 Pan-M。
