# Nimbus：outer P1/P99 共同校准实验

本次已完成原 V1 权重在新输入上的推理，以及从原 V1 开始的独立微调和全量评价。原 Nimbus 和原始数据未改写，Pan-M 保持封存。此为输入归一化的研究适配，不是作者论文全部实验复现。

## 主要结果

Nimbus_outerP1P99_pretrained 外部总体 F1 相对旧 Nimbus 提高 0.2613；AP 变化 +0.2857，TP/FP/FN 变化分别为 +7821/-590/-7821。 Nimbus_outerP1P99_finetuned 外部总体 F1 相对旧 Nimbus 提高 0.2543；AP 变化 +0.3331，TP/FP/FN 变化分别为 +8938/-107/-8938。 对应表为 results/report_vs_previous.csv，筛选 level=split、threshold_mode=official_default、split=external_val_TMA_cores，查看 delta_F1、delta_AP、delta_TP、delta_FP、delta_FN。不能仅据 F1 变化推断邻居污染减少。

固定 score > 0.5 的主结果来自 results/report_primary.csv；两模型在同一完整标签集合比较，未按特征或预测成功情况删行。

| model | split | precision | recall | F1 | AP | ROC_AUC | coverage | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | external_val_TMA_cores | 0.6547 | 0.9086 | 0.7610 | 0.8302 | 0.9903 | 1.0000 | 20290 | 10703 | 2041 |
| GMM | internal_val_WSIs | 0.7513 | 0.9382 | 0.8344 | 0.9481 | 0.9974 | 1.0000 | 17789 | 5890 | 1171 |
| Nimbus_outerP1P99_pretrained | external_val_TMA_cores | 0.8717 | 0.8291 | 0.8499 | 0.9257 | 0.9956 | 1.0000 | 18514 | 2724 | 3817 |
| Nimbus_outerP1P99_pretrained | internal_val_WSIs | 0.8453 | 0.9130 | 0.8779 | 0.9438 | 0.9978 | 1.0000 | 17311 | 3167 | 1649 |
| Nimbus_outerP1P99_finetuned | external_val_TMA_cores | 0.7273 | 0.9278 | 0.8154 | 0.8909 | 0.9939 | 1.0000 | 20718 | 7769 | 1613 |
| Nimbus_outerP1P99_finetuned | internal_val_WSIs | 0.7094 | 0.9594 | 0.8157 | 0.9118 | 0.9967 | 1.0000 | 18191 | 7451 | 769 |

## 验证集门限

补充门限仅在内部三个 WSI 的完整标签上，按等权平均 F1 选择 0.01–0.99 网格；并列优先距离0.5更近，再取较高者。微调检查点也由内部验证选择，因此内部为开发拟合成绩。本轮没有新 OOF 或额外参数搜索。GMM 仅保留原分数和固定0.5，不重新调参。

来源 results/report_validation_threshold.csv；实际门限和网格分别见 models/*_thresholds.json、results/*_validation_threshold_grid.csv。

| model | split | precision | recall | F1 | AP | ROC_AUC | coverage | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nimbus_outerP1P99_pretrained | external_val_TMA_cores | 0.8912 | 0.7897 | 0.8374 | 0.9257 | 0.9956 | 1.0000 | 17635 | 2154 | 4696 |
| Nimbus_outerP1P99_pretrained | internal_val_WSIs | 0.8667 | 0.8891 | 0.8778 | 0.9438 | 0.9978 | 1.0000 | 16857 | 2592 | 2103 |
| Nimbus_outerP1P99_finetuned | external_val_TMA_cores | 0.8354 | 0.8003 | 0.8175 | 0.8909 | 0.9939 | 1.0000 | 17871 | 3521 | 4460 |
| Nimbus_outerP1P99_finetuned | internal_val_WSIs | 0.8250 | 0.8675 | 0.8457 | 0.9118 | 0.9967 | 1.0000 | 16448 | 3489 | 2512 |

## 外部各 TMA 与组织芯

来源 results/report_external_TMA.csv，筛选 level=slide、split=external_val_TMA_cores、threshold_mode=official_default。

| model | context | precision | recall | F1 | AP | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GMM | TMA1_CD8 | 0.8975 | 0.8135 | 0.8534 | 0.9399 | 5919 | 676 | 1357 |
| GMM | TMA2_CD8 | 0.7416 | 0.9084 | 0.8166 | 0.8935 | 6238 | 2173 | 629 |
| GMM | TMA3_CD8 | 0.5087 | 0.9933 | 0.6728 | 0.9232 | 8133 | 7854 | 55 |
| Nimbus_outerP1P99_pretrained | TMA1_CD8 | 0.8561 | 0.8543 | 0.8552 | 0.9278 | 6216 | 1045 | 1060 |
| Nimbus_outerP1P99_pretrained | TMA2_CD8 | 0.8781 | 0.7591 | 0.8143 | 0.9065 | 5213 | 724 | 1654 |
| Nimbus_outerP1P99_pretrained | TMA3_CD8 | 0.8812 | 0.8653 | 0.8732 | 0.9413 | 7085 | 955 | 1103 |
| Nimbus_outerP1P99_finetuned | TMA1_CD8 | 0.7126 | 0.9461 | 0.8129 | 0.8876 | 6884 | 2777 | 392 |
| Nimbus_outerP1P99_finetuned | TMA2_CD8 | 0.7229 | 0.9014 | 0.8023 | 0.8741 | 6190 | 2373 | 677 |
| Nimbus_outerP1P99_finetuned | TMA3_CD8 | 0.7448 | 0.9336 | 0.8286 | 0.9095 | 7644 | 2619 | 544 |

逐组织芯完整指标在 results/all_metrics.csv，筛选 level=image；等权均值、中位数、最低F1及无阳性预测单位数在 results/unit_equal_weight_summary.csv。病例身份缺失，组织芯不等同独立患者。无真阳性时 recall/AP/ROC-AUC 按定义输出 NA，且无阳性预测时 precision=0；无真阳性且无预测阳性时 F1=NA。未进行细胞级显著性检验。

## 输入定义与训练

共享校准由 cd8_robust_calibration_20261030 提供：每个至少5个有限有效像素的原实例先计算P10/P90，再按完整原WSI或单个TMA芯的全部合格实例等权计算外层P1/P99。包含无标签实例；不读标签或模型预测估计边界。计算float64，裁剪到[0,1]后转float32网络输入；不再除以17.2643。校准表在 results/normalization_parameters.csv，按image_id查看L/U、n_eligible、calibration_invalid。当前校准单位数为 48。原始图像、实例ID及掩码未改动。

官方源码固定 241281ef85ba34d9857cc03886cc45aedd0f0b34，V1权重revision 3affc9e774d68408701e44ed6136b1e13503c639、SHA256 0651a3852fbafc03be415a1ec80ef947cc1913a081cf0bc649301ade887a82a9。模型、既有0.5缩放、全图网格与完整原实例聚合保持不变。固定源码的实际TTA包含四次270°及上下/左右翻转；保持该行为并验证流式等价。模型只输入CD8与掩码内部二值图，不输入GMM、人工标签、形态表或身份编号。

微调seed=0，最多10epochs，patience=5，batch=8，学习率3e-6。初始化为原V1，不加载旧微调权重。选择记录见 models/finetuned_checkpoint.json；当前选定validation_call=3。训练与验证的几何tile集合在无校准失败时强制与旧实验一致。患者信息缺失，保留官方WSI划分。掩码为expanded_nucleus，边缘是扩张核近似边缘，不能解释成验证过的真实细胞膜。

## 核验、耗时与错误审阅

tests/pilot_reference_comparison.csv 给出真实训练、内部及TMA1/2/3样例对官方推理的数值对照；tests/global_preparation_pixel_checks.csv 检查归一化和全图缩放。tests/local_real_crop_numeric_checks.csv 的人工bounds真实crop检查仅验证局部算术，不代表完整单位校准；tests/offline_numeric_checks.csv 为服务器受控数值检查。results/prediction_integrity.csv 和 results/seam_cell_audit.csv 检查原ID、一行一细胞及接缝覆盖。

校准失败时保留全部ID、score=NaN，不能用网络对NaN的有限输出代替有效观测。主表失败计数合计为 0（按模型/划分相加，不是去重细胞数）；具体n_failed/coverage查看主表。局部未观测像素在normalized_full中保留NaN及有效支持；仅网络张量使用零哨兵表示未观测上下文，不将其计为真实观测。所有目标实例像素必须有效。

资源来源 results/runtime_summary.csv；fit_seconds包含优化及验证，RSS为运行进程高水位，不能解释为纯模型权重内存。

| model | inference_seconds | inference_peak_gpu_allocated_gib | inference_process_peak_rss_gib | fit_seconds | fit_peak_gpu_allocated_gib | fit_process_peak_rss_gib |
| --- | --- | --- | --- | --- | --- | --- |
| Nimbus_outerP1P99_pretrained | 2745.0040 | 0.9810 | 17.5301 | 0.0000 | 0.0000 | 0.0000 |
| Nimbus_outerP1P99_finetuned | 2643.5792 | 0.9810 | 17.6108 | 531.5023 | 5.8368 | 9.1096 |

开发集病例索引、原始CD8图、掩码叠加与模型对照见 development_review/development_cases.csv、case_model_comparison.csv 和 figures/*/development_*.png/svg。弱表达、邻居接触侧亮、局部高参照等均为预定图像属性候选，不能单凭这些代理确认污染原因。图件的数值源表位于 figures/source_data。

## 复现与完整文件

运行入口：`python scripts/run_pipeline.py --stage all`。服务器目录：`/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_nimbus_robust_20261030`。完整逐细胞结果在 predictions/Nimbus_outerP1P99_pretrained/all_cells.parquet 和 predictions/Nimbus_outerP1P99_finetuned/all_cells.parquet；含原键、标签、split/context、连续评分、两套预测及QC。曲线和threshold数组在 results/curves；微调权重在 models/Nimbus_outerP1P99_finetuned.pt。旧结果比较表为 results/paired_vs_previous_nimbus.csv，GMM比较为 results/paired_vs_GMM.csv。代码/输入/校准/权重哈希变更拒绝复用缓存。

上述归一化是项目适配，文献没有证明双层P1/P99最优。极亮实例可是真阳性；新裁剪可能压缩强阳性差异或放大稀少阳性图的背景。当前结果仅支持此固定协议的观测变化。

## 审计补充

预推理实现修订及兼容缓存核验见 REPRODUCIBILITY_NOTES.md、provenance/IMPLEMENTATION_FIX_001.json 和 provenance/CACHE_COMPATIBILITY_AUDIT.json。校准覆盖率单独见 results/calibration_coverage.csv，按 split、level、context、image_id 筛选 n_cells、n_calibration_invalid、calibration_coverage；不以模型有限评分代替有效输入。

runtime_summary.csv 的 inference_seconds 是各完整单位 infer_one 的总耗时，包括缓存读取、模型推理、完整实例聚合及文件写入；fit_seconds 包括优化、验证、初始化及检查点保存。两者不是纯GPU算子计时，也不构成隔离机器条件下的速度基准。
