# 已完成校准实验汇总

本目录汇总现有 PhenoBIC CD8 实验，不训练模型、不运行新预测、不读取 Pan-M。

主要阅读 [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)。数值排名在 [CALIBRATED_RANKINGS.csv](CALIBRATED_RANKINGS.csv)，完整实验范围在 [EXPERIMENT_INVENTORY.csv](EXPERIMENT_INVENTORY.csv)。

主排名按外部 437,480 个有标签细胞、固定严格 score > 0.5 的 F1 降序排列。补充面板只使用训练 OOF 选定的 E1–E4 门限；GMM 保持原 0.5，未把它重新调成 OOF 门限。所有排名是已有冻结实验的回顾性描述。

## 文件

| 文件 | 内容 |
| --- | --- |
| CALIBRATED_RANKINGS.csv | 内外部总体指标、各指标排名、实际门限与监督方式 |
| CALIBRATED_CONTEXT_METRICS.csv | 逐 WSI/TMA 来源汇总 |
| CALIBRATED_UNIT_METRICS.csv | 逐 WSI/组织芯结果，保留混淆计数和完整集合覆盖率 |
| CALIBRATED_MACRO_METRICS.csv | 单位等权汇总、分布和无阳性预测单位数 |
| PAIRED_VS_GMM.csv | E1–E4 相对原 GMM 的指标及 TP/FP/FN 变化 |
| EXPERIMENT_INVENTORY.csv | 所有已完成实验及进入共同校准主榜的资格说明 |
| SOURCE_MANIFEST.csv | 汇总读取的原文件及 SHA-256，未包含任何图像或逐细胞数据 |
| AUDIT_CHECKS.csv | 142 项混淆矩阵重算、完整集合、门限及逐单位加总检查 |
| LEAKAGE_AUDIT.json | 独立服务器记录审计、哈希对应及尚未核实的独立性边界 |
| SUMMARY_VALIDATION.json | CSV 导出后数值回读和原文件哈希未变检查 |

CSV 采用 UTF-8 BOM，指标保持原始精度，显示四位小数仅用于报告。空白为不适用/缺失，未填成零。原 3 张训练 WSI 的重新代入成绩不进入本排名。单种子 E1–E4 不提供未经支持的置信区间。

## 重建摘要

需要 Python 3.8+ 和本仓库已有汇总文件。主入口只使用 Python 标准库，不加载原图、模型或逐细胞 Parquet：

```bash
python analyses/cd8_calibrated_results_summary_20261029/prepare_summary.py --output-dir /path/to/new_summary
```

原始表格导出工具 export_tables.mjs 也保留，适用于已有 @oai/artifact-tool 环境；通常无需使用。它接收 prepare_summary.py --payload 输出的中间 JSON。

Python 入口拒绝覆盖已有目录；重复执行请指定新的目录。报告和独立泄露审计是人工核对的解释材料，不由上述重建命令重新认证。CSV 中的 source_path 相对于仓库根目录；全量结果保留在服务器 `/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/` 下对应分析目录。

## 发布副本

DELIVERY_MANIFEST.json 和 SUMMARY_CONFIG.json 保留首次生成摘要时的记录。本次发布增加纯 Python CSV 重建入口，并整理报告表述；结果数值未变，见 [重建核验](PYTHON_REBUILD_VALIDATION.json)。当前发布文件的 LF 归一化 SHA-256 见 [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)。
