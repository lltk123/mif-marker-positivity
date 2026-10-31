# 双层分位数校准后重新训练 TabM 与 Nimbus

本轮采用 E1_outer_P1_P99：先计算每实例内 CD8 像素 P10/P90，再跨同一原始 WSI／组织芯取 P1/P99，所有有效像素共享 float64 clip((I-L)/(U-L),0,1)。完整规则见 RETRAIN_PROTOCOL.json；文献和适配边界见 ../../docs/normalization_percentiles_20261030/ANALYSIS_REPORT.md。

## 目录

- ../cd8_robust_calibration_20261030：48 个单位的尺度参数、像素重提 90 特征、QC、固定键及哈希。
- ../cd8_tabm_robust_20261030：A/B/C 各 3 种子 TabM，匹配 HGB 对照、完整预测和报告。
- ../cd8_nimbus_robust_20261030：原 V1 在新输入上的推理，以及从 V1 单独微调后的推理和报告。
- 本目录：固定总协议、统一恢复入口、新旧实验比较。

Pan-M 保持封存；原分割、原 ID、标签和划分保持不变。旧实验结果只读复用。全量特征、模型及逐细胞预测保留在服务器。

## 在服务器执行或恢复

项目路径：/mnt/disk3/mIHC/mif_marker_positivity_20261022。

运行下面的 Python 入口。它依次执行共享特征、Nimbus、TabM、结果比较及综合报告；共享形态提取复用原 morphology 环境，其余阶段复用原 Nimbus 环境。各子流程校验配置、代码、输入和产物哈希后才复用完成缓存。此处“恢复”指已完成阶段复用，不把中断的优化器状态称为精确续训。

    /mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_nimbus_20261024/.venv/bin/python \
      /mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_comparison_20261030/scripts/run_retraining.py \
      --stage all --workers 4

可选 --stage shared、nimbus、tabm、compare 或 report 单独运行。--dry-run 只打印命令。--feature-python 与 --python 可分别指定兼容的固定环境。正式入口面向 Linux 服务器，不会自动下载原始数据、模型或修改旧环境。

## 比较表

统一入口要求两套新流程通过正式评价及交付核验，并校验旧结果的对应完成记录后，比较脚本才生成：

- results/MODEL_COMPARISON.csv：逐种子、划分、WSI／组织芯、新旧模型的指标和来源。
- results/SUMMARY_METRICS.csv：各模型种子均值、标准差、种子数及完整评分状态。计数均值不代表合并多个种子的细胞。
- results/PAIRED_CHANGES.csv：同模型、同特征组、同单位的新旧指标及 TP/FP/FN 变化。
- results/RANKINGS_FIXED_0.5.csv：固定严格大于 0.5 的排名；缺失评分的组合不进入完整集合排名。
- results/COMPARISON_COMPLETE.json：输入与输出哈希、运行状态。
- ANALYSIS_REPORT.md、report_tables/、figures/、figure_source/：中文综合报告、正式数值表、中英文 PNG/SVG 和绘图源数据。REPORT_COMPLETE.json 绑定产物哈希。

主指标使用完整预定有标签集合。额外门限由内部验证集确定；不同门限模式分开。新旧 TabM 的弱阳性和接触侧风险分层沿用旧训练集定义，通过显式键附加原始审计字段，不进入新模型输入。

本轮已完成全量校准、TabM A/B/C 三种子及匹配 HGB、Nimbus 预训练推理与独立微调、全量评价和新旧配对。正式结果见 [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)，固定门限外部排序见 [RANK_external_val_TMA_cores.csv](report_tables/RANK_external_val_TMA_cores.csv)。完成证据由各子流程清单、COMPARISON_COMPLETE.json 和 REPORT_COMPLETE.json 提供；图件核查见 FIGURE_QA.json。

## 审阅包

四个新目录的完成记录和关键报告、指标哈希均通过后，可运行标准库打包入口：

    python scripts/build_review_bundle.py --project-root /path/to/project --output-zip /path/to/new_review.zip

它保留项目相对目录，收录代码、配置、MD、汇总 CSV 与 PNG/SVG，排除原始数据、全量逐细胞文件、权重、缓存、环境和密码密钥。输出 ZIP、独立文件清单及 ZIP 校验值，拒绝覆盖已有包。清单写明全量文件的服务器路径；打包不会重新执行模型或读取封存数据。
