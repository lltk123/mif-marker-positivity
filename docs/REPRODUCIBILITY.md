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
