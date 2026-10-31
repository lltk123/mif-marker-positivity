# PhenoBIC CD8：TabM A/B/C 与匹配 HGB 的完整实验报告

本报告仅依据已冻结的完整实验。所有数值表的来源、筛选条件、聚合方式及指标列见 [report_tables.csv](results/report_tables.csv)。均值 ± 标准差是种子间描述性波动，不是患者置信区间。

## 1. 主要结论与完成边界

数值依据：[report_performance_fixed_0.5.csv](results/report_performance_fixed_0.5.csv)；筛选：split=external_val_TMA_cores,group_type=split,model_family=TabM；指标列：`f1_mean,f1_std`。

固定 0.5 门限的完整外部细胞集合上，TabM A/B/C 中平均 F1 最高为 **B**，F1 为 **0.8518 ± 0.0009**。该排序是冻结后的描述，不用于重选模型、特征或门限。

- TabM_B_minus_A：总体 F1 提高，ΔF1=0.0124 ± 0.0065；平均 ΔTP=523.0、ΔFP=126.0、ΔFN=-523.0。

- TabM_C_minus_B：总体 F1 降低，ΔF1=-0.0111 ± 0.0058；平均 ΔTP=-605.7、ΔFP=-299.3、ΔFN=605.7。

- TabM_C_minus_A：总体 F1 提高，ΔF1=0.0013 ± 0.0117；平均 ΔTP=-82.7、ΔFP=-173.3、ΔFN=82.7。

数值依据：[report_paired_deltas.csv](results/report_paired_deltas.csv)；筛选：split=external_val_TMA_cores,group_type=split,threshold_mode=fixed_0.5,comparison=上述三项；指标列：`delta_f1_mean/std,delta_tp_mean,delta_fp_mean,delta_fn_mean`。

已完成官方 TabM 实现的逐细胞 CD8 任务适配、从头训练、三个随机种子和匹配 HGB 对照；不宣称复现 TabM 原论文全部数据集或指标。Pan-M 持续封存，本轮没有读取其图像、标签、分布或预测。

## 2. 数据、划分与输入

数值依据：[report_data_inventory.csv](results/report_data_inventory.csv)；筛选：scope=各官方划分及all_instances_including_unlabelled；指标列：`n_labelled_cells,n_all_cells,n_units`。

| 集合 | 有标签细胞 | 全部实例 | WSI/组织芯单位数 |
| --- | --- | --- | --- |
| external_val_TMA_cores | 437480.0 | NA | 42.0 |
| internal_val_WSIs | 539132.0 | NA | 3.0 |
| training_WSIs | 101057.0 | NA | 3.0 |
| all_instances_including_unlabelled | NA | 1817783.0 | NA |

统一键为 dataset_id + image_id + cell_id + marker，通过显式 ID 对齐。保留官方训练 WSI、内部 WSI 与外部 TMA 划分；不随机打散细胞。缺少一致患者映射时按原 WSI/组织芯分组，不将组织芯当作独立患者。仅原训练划分有标签细胞用于监督优化和拟合预处理；无标签实例保留预测，不进入分类指标。

数值依据：[report_calibration_status.csv](results/report_calibration_status.csv)；筛选：all original instances；指标列：`n_cells,n_calibration_invalid_cells,n_calibration_invalid_units,calibration_coverage`。

校准失败实例 0 个、单位 0 个；所有 ID 保留。正式指标依旧使用完整有标签集合；模型可通过原缺失处理产生分数，但有限分数不等于校准观测有效。逐 split/单位指标的 n_calibration_invalid 与 calibration_coverage 单独报告。

数值依据：[report_feature_sets.csv](results/report_feature_sets.csv)；筛选：feature_set=A/B/C；指标列：`n_requested,n_effective,excluded`。

| 组合 | 请求维数 | 有效维数 | 排除原因 |
| --- | --- | --- | --- |
| A | 8 | 8 | {} |
| B | 27 | 27 | {} |
| C | 90 | 90 | {} |

A 为 E1_outer_P1_P99 像素校准后重新提取的 8 项 CD8 强度；B 加入原有 19 项形态；C 加入同一归一化像素重提的 63 项空间特征。完整名称和顺序见 FEATURE_SETS.json、results/feature_manifest.csv。标识、患者/批次、真值和 GMM 不进入模型。当前 mask 为 expanded_nucleus；其边缘是扩张核近似边缘，不是经过验证的真实膜。长度和面积保持像素尺度。

图像共同边界为 L=P1(实例内P10)、U=P99(实例内P90)，每个单位独立使用本图至少含 5 个有限像素的全部实例估计边界，不读取标签；所有有效像素使用共同浮点映射并 clip 到 [0,1]。中位数插补、带拟合噪声的正态分位数变换、PLE 分箱均仅训练拟合。HGB 使用相同有效特征和预处理，便于区分特征组合与学习器收益；它与历史未经该预处理的 HGB 是不同实验条件。

## 3. 官方实现、有限调参与冻结

数值依据：[report_provenance.csv](results/report_provenance.csv)；筛选：item=official_commit/tabm_version/rtdl_num_embeddings/hardware；指标列：`value,source`。

官方 commit 为 `28e47ae301c92ec37787dde1ce923a0793f405b4`；TabM 0.0.3，rtdl_num_embeddings 0.0.12。采用 TabM with piecewise-linear embeddings (TabM dagger)。解析后的结构、优化器、嵌入参数与随机种子见 TABM_PROTOCOL.json；下面模型表记录实际训练状态。

数值依据：[report_training_configuration.csv](results/report_training_configuration.csv)；筛选：全部冻结参数；指标列：`parameter,value`。

| 参数 | 冻结配置 |
| --- | --- |
| arch_type | "tabm" |
| k | 32 |
| n_blocks | 2 |
| d_block | 512 |
| dropout | 0.1 |
| embedding | {"type": "PiecewiseLinearEmbeddings", "n_bins": 48, "d_embedding": 16, "activation": false, "version": "B", "fitting": "train-only quantiles"} |
| optimizer | {"type": "AdamW", "weight_decay": 0.0003} |
| candidates | [{"name": "official_default", "learning_rate": 0.002}, {"name": "lower_lr", "learning_rate": 0.0005}] |
| batch_size | 256 |
| max_epochs | 100 |
| patience | 16 |
| seeds | [0, 1, 2] |
| training_loss | "mean cross entropy over B*k separate member outputs" |
| inference | "mean(softmax(logits, class_dim)[positive], member_dim)" |
| precision | "float32" |
| selection_metric | "equal-WSI mean AP on original internal validation; max" |
| HGB | {"learning_rate": 0.06, "max_iter": 150, "max_leaf_nodes": 15, "min_samples_leaf": 80, "l2_regularization": 2.0, "early_stopping": false} |

使用两个 logits；每个成员分别计算交叉熵再平均。推理先成员 softmax，再平均阳性概率，没有平均 logits。每组种子 0 比较预定两个学习率，其他结构固定；按内部 WSI 等权 AP 早停和选择配置，再完成种子 1、2。内部还用于全局门限开发，其成绩属于开发拟合表现。GMM 保持原固定 0.5，不重新拟合或调门限。

数值依据：[report_model_artifacts.csv](results/report_model_artifacts.csv)；筛选：family=TabM，各feature_set和seed；指标列：`learning_rate,best_epoch,epochs_run,epoch_cap_reached,threshold_validation`。

| 模型 | 学习率 | 最佳epoch | 运行epoch | 触及预算上限 | 验证门限 |
| --- | --- | --- | --- | --- | --- |
| TabM_A_s0 | 0.002 | 34.0 | 50.0 | False | 0.74 |
| TabM_A_s1 | 0.002 | 12.0 | 28.0 | False | 0.9 |
| TabM_A_s2 | 0.002 | 11.0 | 27.0 | False | 0.79 |
| TabM_B_s0 | 0.002 | 5.0 | 21.0 | False | 0.87 |
| TabM_B_s1 | 0.002 | 3.0 | 19.0 | False | 0.86 |
| TabM_B_s2 | 0.002 | 2.0 | 18.0 | False | 0.8 |
| TabM_C_s0 | 0.0005 | 5.0 | 21.0 | False | 0.86 |
| TabM_C_s1 | 0.0005 | 5.0 | 21.0 | False | 0.8 |
| TabM_C_s2 | 0.0005 | 2.0 | 18.0 | False | 0.89 |

风险分层在最终模型/测试协议前冻结，哈希核验通过。原训练 WSI 用于训练，内部 WSI 用于早停、学习率与门限开发；外部 PhenoBIC 在本轮模型及门限冻结后评价。

## 4. 固定 0.5 主结果

数值依据：[report_performance_fixed_0.5.csv](results/report_performance_fixed_0.5.csv)；筛选：group_type=split；内部开发验证与外部测试分别如下；GMM始终fixed_0.5；指标列：`precision/recall/f1/average_precision/positive_fraction_absolute_error_mean及_std`。

### 内部开发验证

| 模型 | Precision | Recall | F1 | AP | 阳性比例绝对误差 |
| --- | --- | --- | --- | --- | --- |
| GMM（固定0.5） | 0.7513 | 0.9382 | 0.8344 | 0.9481 | 0.0088 |
| HGB_A | 0.8856 ± 0.0000 | 0.9543 ± 0.0000 | 0.9186 ± 0.0000 | 0.9787 ± 0.0000 | 0.0027 ± 0.0000 |
| HGB_B | 0.8859 ± 0.0000 | 0.9548 ± 0.0000 | 0.9191 ± 0.0000 | 0.9786 ± 0.0000 | 0.0027 ± 0.0000 |
| HGB_C | 0.8891 ± 0.0000 | 0.9627 ± 0.0000 | 0.9244 ± 0.0000 | 0.9826 ± 0.0000 | 0.0029 ± 0.0000 |
| TabM_A | 0.8886 ± 0.0221 | 0.9524 ± 0.0094 | 0.9192 ± 0.0076 | 0.9808 ± 0.0000 | 0.0025 ± 0.0013 |
| TabM_B | 0.8423 ± 0.0108 | 0.9666 ± 0.0019 | 0.9002 ± 0.0055 | 0.9801 ± 0.0004 | 0.0052 ± 0.0006 |
| TabM_C | 0.8753 ± 0.0284 | 0.9659 ± 0.0085 | 0.9181 ± 0.0120 | 0.9836 ± 0.0002 | 0.0037 ± 0.0016 |

### 外部测试

| 模型 | Precision | Recall | F1 | AP | 阳性比例绝对误差 |
| --- | --- | --- | --- | --- | --- |
| GMM（固定0.5） | 0.6547 | 0.9086 | 0.7610 | 0.8302 | 0.0198 |
| HGB_A | 0.9514 ± 0.0000 | 0.7617 ± 0.0000 | 0.8461 ± 0.0000 | 0.9446 ± 0.0000 | 0.0102 ± 0.0000 |
| HGB_B | 0.9555 ± 0.0000 | 0.7516 ± 0.0000 | 0.8414 ± 0.0000 | 0.9477 ± 0.0000 | 0.0109 ± 0.0000 |
| HGB_C | 0.9627 ± 0.0000 | 0.7591 ± 0.0000 | 0.8488 ± 0.0000 | 0.9571 ± 0.0000 | 0.0108 ± 0.0000 |
| TabM_A | 0.9544 ± 0.0039 | 0.7491 ± 0.0136 | 0.8393 ± 0.0070 | 0.9482 ± 0.0003 | 0.0110 ± 0.0009 |
| TabM_B | 0.9491 ± 0.0012 | 0.7725 ± 0.0022 | 0.8518 ± 0.0009 | 0.9451 ± 0.0031 | 0.0095 ± 0.0002 |
| TabM_C | 0.9637 ± 0.0018 | 0.7454 ± 0.0088 | 0.8406 ± 0.0049 | 0.9576 ± 0.0003 | 0.0116 ± 0.0005 |

总体指标先在每个种子的完整有标签集合上计算，再汇总种子波动；它们是按细胞汇总结果。训练重代入成绩、AUC、specificity、TP/FP/FN/TN、真实/预测阳性比例与覆盖率都保存在对应 CSV。此次没有新 OOF；训练成绩不是 OOF，内部验证是明确的开发集。

### 外部 TMA 分组

数值依据：[report_performance_fixed_0.5.csv](results/report_performance_fixed_0.5.csv)；筛选：split=external_val_TMA_cores,group_type=context,context=TMA1_CD8/TMA2_CD8/TMA3_CD8；指标列：`f1/precision/recall/roc_auc/average_precision/fp/fn_mean及_std`。

| TMA | 模型 | F1 | Precision | Recall | AP | AUC | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TMA1_CD8 | GMM（固定0.5） | 0.8534 | 0.8975 | 0.8135 | 0.9399 | 0.9968 | 676.0000 | 1357.0000 |
| TMA2_CD8 | GMM（固定0.5） | 0.8166 | 0.7416 | 0.9084 | 0.8935 | 0.9951 | 2173.0000 | 629.0000 |
| TMA3_CD8 | GMM（固定0.5） | 0.6728 | 0.5087 | 0.9933 | 0.9232 | 0.9946 | 7854.0000 | 55.0000 |
| TMA1_CD8 | HGB_A | 0.8937 ± 0.0000 | 0.9343 ± 0.0000 | 0.8565 ± 0.0000 | 0.9566 ± 0.0000 | 0.9978 ± 0.0000 | 438.0000 ± 0.0000 | 1044.0000 ± 0.0000 |
| TMA2_CD8 | HGB_A | 0.7139 ± 0.0000 | 0.9700 ± 0.0000 | 0.5647 ± 0.0000 | 0.9285 ± 0.0000 | 0.9942 ± 0.0000 | 120.0000 ± 0.0000 | 2989.0000 ± 0.0000 |
| TMA3_CD8 | HGB_A | 0.8961 ± 0.0000 | 0.9570 ± 0.0000 | 0.8426 ± 0.0000 | 0.9616 ± 0.0000 | 0.9974 ± 0.0000 | 310.0000 ± 0.0000 | 1289.0000 ± 0.0000 |
| TMA1_CD8 | HGB_B | 0.8943 ± 0.0000 | 0.9400 ± 0.0000 | 0.8528 ± 0.0000 | 0.9608 ± 0.0000 | 0.9978 ± 0.0000 | 396.0000 ± 0.0000 | 1071.0000 ± 0.0000 |
| TMA2_CD8 | HGB_B | 0.7035 ± 0.0000 | 0.9715 ± 0.0000 | 0.5513 ± 0.0000 | 0.9290 ± 0.0000 | 0.9936 ± 0.0000 | 111.0000 ± 0.0000 | 3081.0000 ± 0.0000 |
| TMA3_CD8 | HGB_B | 0.8905 ± 0.0000 | 0.9611 ± 0.0000 | 0.8296 ± 0.0000 | 0.9678 ± 0.0000 | 0.9975 ± 0.0000 | 275.0000 ± 0.0000 | 1395.0000 ± 0.0000 |
| TMA1_CD8 | HGB_C | 0.8978 ± 0.0000 | 0.9505 ± 0.0000 | 0.8506 ± 0.0000 | 0.9701 ± 0.0000 | 0.9983 ± 0.0000 | 322.0000 ± 0.0000 | 1087.0000 ± 0.0000 |
| TMA2_CD8 | HGB_C | 0.7157 ± 0.0000 | 0.9715 ± 0.0000 | 0.5665 ± 0.0000 | 0.9378 ± 0.0000 | 0.9958 ± 0.0000 | 114.0000 ± 0.0000 | 2977.0000 ± 0.0000 |
| TMA3_CD8 | HGB_C | 0.8994 ± 0.0000 | 0.9688 ± 0.0000 | 0.8393 ± 0.0000 | 0.9739 ± 0.0000 | 0.9979 ± 0.0000 | 221.0000 ± 0.0000 | 1316.0000 ± 0.0000 |
| TMA1_CD8 | TabM_A | 0.8932 ± 0.0027 | 0.9386 ± 0.0046 | 0.8520 ± 0.0087 | 0.9614 ± 0.0011 | 0.9980 ± 0.0000 | 405.6667 ± 36.9369 | 1077.0000 ± 63.3798 |
| TMA2_CD8 | TabM_A | 0.6994 ± 0.0115 | 0.9705 ± 0.0009 | 0.5468 ± 0.0145 | 0.9352 ± 0.0016 | 0.9968 ± 0.0001 | 114.3333 ± 6.6583 | 3112.0000 ± 99.2421 |
| TMA3_CD8 | TabM_A | 0.8888 ± 0.0078 | 0.9604 ± 0.0050 | 0.8274 ± 0.0174 | 0.9652 ± 0.0001 | 0.9979 ± 0.0001 | 280.0000 ± 43.3474 | 1413.3333 ± 142.1173 |
| TMA1_CD8 | TabM_B | 0.8973 ± 0.0004 | 0.9327 ± 0.0008 | 0.8645 ± 0.0010 | 0.9619 ± 0.0016 | 0.9977 ± 0.0002 | 453.6667 ± 5.8595 | 985.6667 ± 7.0946 |
| TMA2_CD8 | TabM_B | 0.7219 ± 0.0026 | 0.9692 ± 0.0008 | 0.5751 ± 0.0036 | 0.9229 ± 0.0095 | 0.9940 ± 0.0027 | 125.3333 ± 4.0415 | 2917.6667 ± 24.9065 |
| TMA3_CD8 | TabM_B | 0.9020 ± 0.0010 | 0.9529 ± 0.0020 | 0.8564 ± 0.0031 | 0.9654 ± 0.0019 | 0.9975 ± 0.0002 | 347.0000 ± 16.7033 | 1176.0000 ± 25.5343 |
| TMA1_CD8 | TabM_C | 0.8888 ± 0.0025 | 0.9529 ± 0.0020 | 0.8329 ± 0.0060 | 0.9703 ± 0.0008 | 0.9984 ± 0.0000 | 299.6667 ± 15.3080 | 1216.0000 ± 43.3128 |
| TMA2_CD8 | TabM_C | 0.7076 ± 0.0067 | 0.9730 ± 0.0008 | 0.5559 ± 0.0085 | 0.9426 ± 0.0021 | 0.9972 ± 0.0002 | 106.0000 ± 4.5826 | 3049.3333 ± 58.3467 |
| TMA3_CD8 | TabM_C | 0.8919 ± 0.0059 | 0.9684 ± 0.0025 | 0.8266 ± 0.0119 | 0.9720 ± 0.0004 | 0.9980 ± 0.0001 | 221.0000 ± 21.0713 | 1419.6667 ± 97.2231 |

## 5. 验证门限补充结果

数值依据：[report_performance_validation_selected.csv](results/report_performance_validation_selected.csv)；筛选：group_type=split；内部开发验证与外部测试分别如下；GMM始终fixed_0.5；指标列：`precision/recall/f1/average_precision/positive_fraction_absolute_error_mean及_std`。

### 内部开发验证

| 模型 | Precision | Recall | F1 | AP | 阳性比例绝对误差 |
| --- | --- | --- | --- | --- | --- |
| GMM（固定0.5） | 0.7513 | 0.9382 | 0.8344 | 0.9481 | 0.0088 |
| HGB_A | 0.9462 ± 0.0000 | 0.9127 ± 0.0000 | 0.9292 ± 0.0000 | 0.9787 ± 0.0000 | 0.0012 ± 0.0000 |
| HGB_B | 0.9467 ± 0.0000 | 0.9129 ± 0.0000 | 0.9295 ± 0.0000 | 0.9786 ± 0.0000 | 0.0013 ± 0.0000 |
| HGB_C | 0.9519 ± 0.0000 | 0.9180 ± 0.0000 | 0.9346 ± 0.0000 | 0.9826 ± 0.0000 | 0.0013 ± 0.0000 |
| TabM_A | 0.9533 ± 0.0063 | 0.9025 ± 0.0103 | 0.9272 ± 0.0026 | 0.9808 ± 0.0000 | 0.0019 ± 0.0006 |
| TabM_B | 0.9491 ± 0.0042 | 0.9078 ± 0.0036 | 0.9280 ± 0.0005 | 0.9801 ± 0.0004 | 0.0015 ± 0.0003 |
| TabM_C | 0.9516 ± 0.0019 | 0.9178 ± 0.0015 | 0.9344 ± 0.0003 | 0.9836 ± 0.0002 | 0.0012 ± 0.0001 |

### 外部测试

| 模型 | Precision | Recall | F1 | AP | 阳性比例绝对误差 |
| --- | --- | --- | --- | --- | --- |
| GMM（固定0.5） | 0.6547 | 0.9086 | 0.7610 | 0.8302 | 0.0198 |
| HGB_A | 0.9617 ± 0.0000 | 0.7222 ± 0.0000 | 0.8249 ± 0.0000 | 0.9446 ± 0.0000 | 0.0127 ± 0.0000 |
| HGB_B | 0.9691 ± 0.0000 | 0.6948 ± 0.0000 | 0.8094 ± 0.0000 | 0.9477 ± 0.0000 | 0.0144 ± 0.0000 |
| HGB_C | 0.9767 ± 0.0000 | 0.6846 ± 0.0000 | 0.8050 ± 0.0000 | 0.9571 ± 0.0000 | 0.0153 ± 0.0000 |
| TabM_A | 0.9677 ± 0.0028 | 0.6870 ± 0.0151 | 0.8035 ± 0.0095 | 0.9482 ± 0.0003 | 0.0148 ± 0.0009 |
| TabM_B | 0.9700 ± 0.0020 | 0.6862 ± 0.0117 | 0.8038 ± 0.0073 | 0.9451 ± 0.0031 | 0.0149 ± 0.0007 |
| TabM_C | 0.9788 ± 0.0007 | 0.6545 ± 0.0198 | 0.7843 ± 0.0141 | 0.9576 ± 0.0003 | 0.0169 ± 0.0011 |

总体指标先在每个种子的完整有标签集合上计算，再汇总种子波动；它们是按细胞汇总结果。训练重代入成绩、AUC、specificity、TP/FP/FN/TN、真实/预测阳性比例与覆盖率都保存在对应 CSV。此次没有新 OOF；训练成绩不是 OOF，内部验证是明确的开发集。

### 外部 TMA 分组

数值依据：[report_performance_validation_selected.csv](results/report_performance_validation_selected.csv)；筛选：split=external_val_TMA_cores,group_type=context,context=TMA1_CD8/TMA2_CD8/TMA3_CD8；指标列：`f1/precision/recall/roc_auc/average_precision/fp/fn_mean及_std`。

| TMA | 模型 | F1 | Precision | Recall | AP | AUC | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TMA1_CD8 | GMM（固定0.5） | 0.8534 | 0.8975 | 0.8135 | 0.9399 | 0.9968 | 676.0000 | 1357.0000 |
| TMA2_CD8 | GMM（固定0.5） | 0.8166 | 0.7416 | 0.9084 | 0.8935 | 0.9951 | 2173.0000 | 629.0000 |
| TMA3_CD8 | GMM（固定0.5） | 0.6728 | 0.5087 | 0.9933 | 0.9232 | 0.9946 | 7854.0000 | 55.0000 |
| TMA1_CD8 | HGB_A | 0.8831 ± 0.0000 | 0.9464 ± 0.0000 | 0.8278 ± 0.0000 | 0.9566 ± 0.0000 | 0.9978 ± 0.0000 | 341.0000 ± 0.0000 | 1253.0000 ± 0.0000 |
| TMA2_CD8 | HGB_A | 0.6856 ± 0.0000 | 0.9717 ± 0.0000 | 0.5296 ± 0.0000 | 0.9285 ± 0.0000 | 0.9942 ± 0.0000 | 106.0000 ± 0.0000 | 3230.0000 ± 0.0000 |
| TMA3_CD8 | HGB_A | 0.8710 ± 0.0000 | 0.9707 ± 0.0000 | 0.7898 ± 0.0000 | 0.9616 ± 0.0000 | 0.9974 ± 0.0000 | 195.0000 ± 0.0000 | 1721.0000 ± 0.0000 |
| TMA1_CD8 | HGB_B | 0.8784 ± 0.0000 | 0.9581 ± 0.0000 | 0.8109 ± 0.0000 | 0.9608 ± 0.0000 | 0.9978 ± 0.0000 | 258.0000 ± 0.0000 | 1376.0000 ± 0.0000 |
| TMA2_CD8 | HGB_B | 0.6566 ± 0.0000 | 0.9748 ± 0.0000 | 0.4950 ± 0.0000 | 0.9290 ± 0.0000 | 0.9936 ± 0.0000 | 88.0000 ± 0.0000 | 3468.0000 ± 0.0000 |
| TMA3_CD8 | HGB_B | 0.8544 ± 0.0000 | 0.9767 ± 0.0000 | 0.7593 ± 0.0000 | 0.9678 ± 0.0000 | 0.9975 ± 0.0000 | 148.0000 ± 0.0000 | 1971.0000 ± 0.0000 |
| TMA1_CD8 | HGB_C | 0.8701 ± 0.0000 | 0.9682 ± 0.0000 | 0.7900 ± 0.0000 | 0.9701 ± 0.0000 | 0.9983 ± 0.0000 | 189.0000 ± 0.0000 | 1528.0000 ± 0.0000 |
| TMA2_CD8 | HGB_C | 0.6602 ± 0.0000 | 0.9777 ± 0.0000 | 0.4983 ± 0.0000 | 0.9378 ± 0.0000 | 0.9958 ± 0.0000 | 78.0000 ± 0.0000 | 3445.0000 ± 0.0000 |
| TMA3_CD8 | HGB_C | 0.8495 ± 0.0000 | 0.9844 ± 0.0000 | 0.7471 ± 0.0000 | 0.9739 ± 0.0000 | 0.9979 ± 0.0000 | 97.0000 ± 0.0000 | 2071.0000 ± 0.0000 |
| TMA1_CD8 | TabM_A | 0.8651 ± 0.0089 | 0.9553 ± 0.0042 | 0.7906 ± 0.0174 | 0.9614 ± 0.0011 | 0.9980 ± 0.0000 | 269.3333 ± 31.9427 | 1523.6667 ± 126.8083 |
| TMA2_CD8 | TabM_A | 0.6590 ± 0.0092 | 0.9745 ± 0.0008 | 0.4979 ± 0.0106 | 0.9352 ± 0.0016 | 0.9968 ± 0.0001 | 89.6667 ± 4.5092 | 3447.6667 ± 73.0365 |
| TMA3_CD8 | TabM_A | 0.8503 ± 0.0099 | 0.9758 ± 0.0025 | 0.7536 ± 0.0169 | 0.9652 ± 0.0001 | 0.9979 ± 0.0001 | 153.3333 ± 19.7569 | 2017.6667 ± 138.7131 |
| TMA1_CD8 | TabM_B | 0.8719 ± 0.0049 | 0.9595 ± 0.0032 | 0.7990 ± 0.0102 | 0.9619 ± 0.0016 | 0.9977 ± 0.0002 | 245.3333 ± 23.2451 | 1462.6667 ± 74.2316 |
| TMA2_CD8 | TabM_B | 0.6503 ± 0.0121 | 0.9754 ± 0.0009 | 0.4878 ± 0.0137 | 0.9229 ± 0.0095 | 0.9940 ± 0.0027 | 84.6667 ± 5.5076 | 3517.0000 ± 94.3027 |
| TMA3_CD8 | TabM_B | 0.8502 ± 0.0067 | 0.9772 ± 0.0017 | 0.7524 ± 0.0114 | 0.9654 ± 0.0019 | 0.9975 ± 0.0002 | 143.6667 ± 12.8582 | 2027.3333 ± 93.3506 |
| TMA1_CD8 | TabM_C | 0.8498 ± 0.0162 | 0.9724 ± 0.0013 | 0.7550 ± 0.0261 | 0.9703 ± 0.0008 | 0.9984 ± 0.0000 | 156.3333 ± 12.6623 | 1782.6667 ± 190.0772 |
| TMA2_CD8 | TabM_C | 0.6365 ± 0.0129 | 0.9802 ± 0.0006 | 0.4713 ± 0.0142 | 0.9426 ± 0.0021 | 0.9972 ± 0.0002 | 65.3333 ± 3.7859 | 3630.3333 ± 97.1717 |
| TMA3_CD8 | TabM_C | 0.8307 ± 0.0127 | 0.9842 ± 0.0001 | 0.7188 ± 0.0190 | 0.9720 ± 0.0004 | 0.9980 ± 0.0001 | 94.3333 ± 3.0551 | 2302.6667 ± 155.4295 |

## 6. 配对消融、架构与逐组织芯差异

数值依据：[report_paired_deltas.csv](results/report_paired_deltas.csv)；筛选：split=external_val_TMA_cores,group_type=split；两套threshold_mode分别标记；指标列：`delta_f1/average_precision/tp/fp/fn_mean及_std`。

| 比较 | 门限 | ΔF1 | ΔTP | ΔFP | ΔFN |
| --- | --- | --- | --- | --- | --- |
| HGB_B_minus_A | fixed_0.5 | -0.0047 ± 0.0000 | -225.0 ± 0.0 | -86.0 ± 0.0 | 225.0 ± 0.0 |
| HGB_B_minus_A | validation_selected | -0.0155 ± 0.0000 | -611.0 ± 0.0 | -148.0 ± 0.0 | 611.0 ± 0.0 |
| HGB_C_minus_A | fixed_0.5 | 0.0028 ± 0.0000 | -58.0 ± 0.0 | -211.0 ± 0.0 | 58.0 ± 0.0 |
| HGB_C_minus_A | validation_selected | -0.0200 ± 0.0000 | -840.0 ± 0.0 | -278.0 ± 0.0 | 840.0 ± 0.0 |
| HGB_C_minus_B | fixed_0.5 | 0.0075 ± 0.0000 | 167.0 ± 0.0 | -125.0 ± 0.0 | -167.0 ± 0.0 |
| HGB_C_minus_B | validation_selected | -0.0044 ± 0.0000 | -229.0 ± 0.0 | -130.0 ± 0.0 | 229.0 ± 0.0 |
| TabM_A_minus_GMM | fixed_0.5 | 0.0783 ± 0.0070 | -3561.3 ± 304.4 | -9903.0 ± 86.7 | 3561.3 ± 304.4 |
| TabM_A_minus_GMM | validation_selected | 0.0425 ± 0.0095 | -4948.0 ± 336.7 | -10190.7 ± 55.9 | 4948.0 ± 336.7 |
| TabM_B_minus_A | fixed_0.5 | 0.0124 ± 0.0065 | 523.0 ± 268.2 | 126.0 ± 63.7 | -523.0 ± 268.2 |
| TabM_B_minus_A | validation_selected | 0.0003 ± 0.0021 | -18.0 ± 76.4 | -38.7 ± 16.6 | 18.0 ± 76.4 |
| TabM_B_minus_GMM | fixed_0.5 | 0.0907 ± 0.0009 | -3038.3 ± 50.0 | -9777.0 ± 25.6 | 3038.3 ± 50.0 |
| TabM_B_minus_GMM | validation_selected | 0.0428 ± 0.0073 | -4966.0 ± 260.2 | -10229.3 ± 40.7 | 4966.0 ± 260.2 |
| TabM_C_minus_A | fixed_0.5 | 0.0013 ± 0.0117 | -82.7 ± 491.3 | -173.3 ± 125.3 | 82.7 ± 491.3 |
| TabM_C_minus_A | validation_selected | -0.0192 ± 0.0206 | -726.7 ± 672.7 | -196.3 ± 61.8 | 726.7 ± 672.7 |
| TabM_C_minus_B | fixed_0.5 | -0.0111 ± 0.0058 | -605.7 ± 245.3 | -299.3 ± 65.0 | 605.7 ± 245.3 |
| TabM_C_minus_B | validation_selected | -0.0194 ± 0.0189 | -708.7 ± 612.5 | -157.7 ± 49.8 | 708.7 ± 612.5 |
| TabM_C_minus_GMM | fixed_0.5 | 0.0796 ± 0.0049 | -3644.0 ± 197.0 | -10076.3 ± 39.5 | 3644.0 ± 197.0 |
| TabM_C_minus_GMM | validation_selected | 0.0233 ± 0.0141 | -5674.7 ± 441.5 | -10387.0 ± 19.2 | 5674.7 ± 441.5 |
| TabM_minus_HGB_A | fixed_0.5 | -0.0067 ± 0.0070 | -280.3 ± 304.4 | -68.0 ± 86.7 | 280.3 ± 304.4 |
| TabM_minus_HGB_A | validation_selected | -0.0214 ± 0.0095 | -785.0 ± 336.7 | -129.7 ± 55.9 | 785.0 ± 336.7 |
| TabM_minus_HGB_B | fixed_0.5 | 0.0104 ± 0.0009 | 467.7 ± 50.0 | 144.0 ± 25.6 | -467.7 ± 50.0 |
| TabM_minus_HGB_B | validation_selected | -0.0056 ± 0.0073 | -192.0 ± 260.2 | -20.3 ± 40.7 | 192.0 ± 260.2 |
| TabM_minus_HGB_C | fixed_0.5 | -0.0082 ± 0.0049 | -305.0 ± 197.0 | -30.3 ± 39.5 | 305.0 ± 197.0 |
| TabM_minus_HGB_C | validation_selected | -0.0206 ± 0.0141 | -671.7 ± 441.5 | -48.0 ± 19.2 | 671.7 ± 441.5 |

同特征 TabM−HGB 辨别学习器差异；B−A 衡量形态组合，C−B 衡量空间组合。不能把 F1 改善直接归因于某一特征或污染消除。GMM 为无监督参考，新模型使用训练标签，监督条件不同。

数值依据：[report_unit_equal.csv](results/report_unit_equal.csv)；筛选：split=external_val_TMA_cores,metric=f1；先逐芯等权再种子汇总；指标列：`unit_equal_mean_mean/std,n_finite_units_mean,n_units_no_positive_predictions_mean`。

| 模型 | 门限 | 逐芯等权F1 | 有效芯数 | 无阳性预测芯数均值 |
| --- | --- | --- | --- | --- |
| GMM（固定0.5） | fixed_0.5 | 0.7667 | 42.0 | 0.0 |
| HGB_A | fixed_0.5 | 0.8713 ± 0.0000 | 42.0 | 0.0 |
| HGB_A | validation_selected | 0.8578 ± 0.0000 | 42.0 | 0.0 |
| HGB_B | fixed_0.5 | 0.8679 ± 0.0000 | 42.0 | 0.0 |
| HGB_B | validation_selected | 0.8480 ± 0.0000 | 42.0 | 0.0 |
| HGB_C | fixed_0.5 | 0.8748 ± 0.0000 | 42.0 | 0.0 |
| HGB_C | validation_selected | 0.8483 ± 0.0000 | 42.0 | 0.0 |
| TabM_A | fixed_0.5 | 0.8655 ± 0.0045 | 42.0 | 0.0 |
| TabM_A | validation_selected | 0.8450 ± 0.0058 | 42.0 | 0.0 |
| TabM_B | fixed_0.5 | 0.8741 ± 0.0003 | 42.0 | 0.0 |
| TabM_B | validation_selected | 0.8440 ± 0.0054 | 42.0 | 0.0 |
| TabM_C | fixed_0.5 | 0.8703 ± 0.0031 | 42.0 | 0.0 |
| TabM_C | validation_selected | 0.8347 ± 0.0099 | 42.0 | 0.0 |

数值依据：[report_lowest_cores.csv](results/report_lowest_cores.csv)；筛选：external, fixed_0.5；每个组合按三种子平均F1最低三个组织芯；指标列：`group_id,context,f1/precision/recall_mean及_std`。

| 模型 | TMA | 组织芯 | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| TabM_A | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_7_3 | 0.5799 ± 0.0168 | 0.9820 ± 0.0005 | 0.4116 ± 0.0171 |
| TabM_C | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_7_3 | 0.5828 ± 0.0078 | 0.9844 ± 0.0020 | 0.4140 ± 0.0079 |
| HGB_B | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_7_3 | 0.5848 ± 0.0000 | 0.9828 ± 0.0000 | 0.4163 ± 0.0000 |
| HGB_C | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_7_3 | 0.5980 ± 0.0000 | 0.9882 ± 0.0000 | 0.4287 ± 0.0000 |
| HGB_A | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_7_3 | 0.6017 ± 0.0000 | 0.9816 ± 0.0000 | 0.4338 ± 0.0000 |
| TabM_B | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_7_3 | 0.6140 ± 0.0051 | 0.9812 ± 0.0007 | 0.4468 ± 0.0056 |
| HGB_B | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_4_1 | 0.6351 ± 0.0000 | 0.9947 ± 0.0000 | 0.4665 ± 0.0000 |
| TabM_C | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_4_1 | 0.6382 ± 0.0081 | 0.9947 ± 0.0001 | 0.4698 ± 0.0087 |
| TabM_A | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_4_1 | 0.6397 ± 0.0121 | 0.9948 ± 0.0001 | 0.4715 ± 0.0131 |
| HGB_C | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_4_1 | 0.6500 ± 0.0000 | 0.9898 ± 0.0000 | 0.4839 ± 0.0000 |
| HGB_A | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_4_1 | 0.6533 ± 0.0000 | 0.9949 ± 0.0000 | 0.4864 ± 0.0000 |
| TabM_A | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_1_1 | 0.6585 ± 0.0105 | 0.9233 ± 0.0004 | 0.5118 ± 0.0128 |
| TabM_B | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_4_1 | 0.6600 ± 0.0059 | 0.9950 ± 0.0001 | 0.4938 ± 0.0066 |
| HGB_B | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_1_1 | 0.6656 ± 0.0000 | 0.9248 ± 0.0000 | 0.5198 ± 0.0000 |
| HGB_A | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_1_1 | 0.6786 ± 0.0000 | 0.9216 ± 0.0000 | 0.5371 ± 0.0000 |
| TabM_C | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_1_1 | 0.6789 ± 0.0107 | 0.9287 ± 0.0017 | 0.5351 ± 0.0137 |
| TabM_B | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_1_1 | 0.6857 ± 0.0031 | 0.9168 ± 0.0017 | 0.5477 ± 0.0043 |
| HGB_C | TMA2_CD8 | external_val_TMA_cores/TMA2_CD8/core_1_1 | 0.6872 ± 0.0000 | 0.9230 ± 0.0000 | 0.5474 ± 0.0000 |

完整分布见 metrics_all.csv、unit_performance_ranked.csv、paired_comparisons.csv。无真实阳性时 recall/AP=NaN；无阳性预测时 precision=0；真值与预测都无阳性时 F1=NaN。宏平均只使用相应指标有限的单位并记录数量，不把 NaN 填零。未做细胞级 bootstrap 或将细胞作为独立样本的显著性检验。

## 7. 弱阳性、接触侧亮信号代理与案例

数值依据：[report_risk_strata.csv](results/report_risk_strata.csv)；筛选：group_type=split,stratum=weak_positive/contact_bright_proxy_negative；按来源和门限分别比较；指标列：`n_cells_mean,recall_mean/std,false_positive_rate_mean/std,fp_mean,fn_mean`。

| 来源 | 门限 | 模型 | 分层 | 细胞数 | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| external_val_TMA_cores | fixed_0.5 | TabM_A | contact_bright_proxy_negative | 9233.0 | NA | 0.0183 ± 0.0033 |
| external_val_TMA_cores | fixed_0.5 | TabM_A | weak_positive | 13991.0 | 0.6959 ± 0.0116 | NA |
| internal_val_WSIs | fixed_0.5 | TabM_A | contact_bright_proxy_negative | 10899.0 | NA | 0.0697 ± 0.0119 |
| internal_val_WSIs | fixed_0.5 | TabM_A | weak_positive | 3153.0 | 0.7227 ± 0.0539 | NA |
| external_val_TMA_cores | validation_selected | TabM_A | contact_bright_proxy_negative | 9233.0 | NA | 0.0102 ± 0.0010 |
| external_val_TMA_cores | validation_selected | TabM_A | weak_positive | 13991.0 | 0.6400 ± 0.0143 | NA |
| internal_val_WSIs | validation_selected | TabM_A | contact_bright_proxy_negative | 10899.0 | NA | 0.0310 ± 0.0047 |
| internal_val_WSIs | validation_selected | TabM_A | weak_positive | 3153.0 | 0.4845 ± 0.0363 | NA |
| external_val_TMA_cores | fixed_0.5 | TabM_B | contact_bright_proxy_negative | 9233.0 | NA | 0.0234 ± 0.0012 |
| external_val_TMA_cores | fixed_0.5 | TabM_B | weak_positive | 13991.0 | 0.7164 ± 0.0018 | NA |
| internal_val_WSIs | fixed_0.5 | TabM_B | contact_bright_proxy_negative | 10899.0 | NA | 0.1019 ± 0.0121 |
| internal_val_WSIs | fixed_0.5 | TabM_B | weak_positive | 3153.0 | 0.8023 ± 0.0101 | NA |
| external_val_TMA_cores | validation_selected | TabM_B | contact_bright_proxy_negative | 9233.0 | NA | 0.0100 ± 0.0008 |
| external_val_TMA_cores | validation_selected | TabM_B | weak_positive | 13991.0 | 0.6394 ± 0.0119 | NA |
| internal_val_WSIs | validation_selected | TabM_B | contact_bright_proxy_negative | 10899.0 | NA | 0.0335 ± 0.0024 |
| internal_val_WSIs | validation_selected | TabM_B | weak_positive | 3153.0 | 0.5043 ± 0.0182 | NA |
| external_val_TMA_cores | fixed_0.5 | TabM_C | contact_bright_proxy_negative | 9233.0 | NA | 0.0134 ± 0.0011 |
| external_val_TMA_cores | fixed_0.5 | TabM_C | weak_positive | 13991.0 | 0.6905 ± 0.0071 | NA |
| internal_val_WSIs | fixed_0.5 | TabM_C | contact_bright_proxy_negative | 10899.0 | NA | 0.0691 ± 0.0115 |
| internal_val_WSIs | fixed_0.5 | TabM_C | weak_positive | 3153.0 | 0.7980 ± 0.0505 | NA |
| external_val_TMA_cores | validation_selected | TabM_C | contact_bright_proxy_negative | 9233.0 | NA | 0.0066 ± 0.0004 |
| external_val_TMA_cores | validation_selected | TabM_C | weak_positive | 13991.0 | 0.6085 ± 0.0205 | NA |
| internal_val_WSIs | validation_selected | TabM_C | contact_bright_proxy_negative | 10899.0 | NA | 0.0277 ± 0.0003 |
| internal_val_WSIs | validation_selected | TabM_C | weak_positive | 3153.0 | 0.5487 ± 0.0124 | NA |

- 外部 fixed_0.5 的 C−B：接触侧亮信号阴性代理组 ΔFPR=-0.0100；弱阳性 ΔRecall=-0.0259。该代理组平均假阳性率下降。同时出现弱阳性平均召回下降，应保留这一代价。

- 外部 validation_selected 的 C−B：接触侧亮信号阴性代理组 ΔFPR=-0.0034；弱阳性 ΔRecall=-0.0310。该代理组平均假阳性率下降。同时出现弱阳性平均召回下降，应保留这一代价。

弱阳性与接触侧亮信号沿用旧实验原始强度列和原训练分位数，显式 audit_raw_* 映射保证新旧风险群组相同；这些审计列不进入模型。FPR=FP/真实阴性数。弱阳性依据 raw whole_mean <= 训练正类 q25，跨图像亮度尺度会影响落入该组的比例，不能等同统一生物学弱表达。这些是图像属性代理，不证明真实邻居污染，因此不能据 FPR 下降声称已确认减少“邻近亮细胞导致”的假阳性。

实际导出 9 个开发案例，数量与选择原因见 development_review/case_index.csv、case_availability.csv，邻居记录见 neighbor_context.csv。只使用训练/内部验证图像、种子0、固定门限，优先展示 B→C 纠正与新增错误，全部案例保留原始像素 NPZ。候选类别不等于污染机制；说明性抽样不能估算总体错误率，原人工标签未经过新增病理专家复核。

## 8. 缺失、数值分辨率与成员分歧

数值依据：[report_missingness.csv](results/report_missingness.csv)；筛选：每个split全部实例，含无标签；下表各split缺失率最高五项；指标列：`n,n_missing,missing_rate`。

| 来源 | 特征 | 全部实例 | 缺失数 | 缺失率 |
| --- | --- | --- | --- | --- |
| external_val_TMA_cores | edge_sector_07_high_fraction | 438236 | 433975 | 0.99028 |
| external_val_TMA_cores | edge_sector_01_high_fraction | 438236 | 433971 | 0.99027 |
| external_val_TMA_cores | edge_sector_05_high_fraction | 438236 | 433969 | 0.99026 |
| external_val_TMA_cores | edge_sector_08_high_fraction | 438236 | 433967 | 0.99026 |
| external_val_TMA_cores | edge_sector_02_high_fraction | 438236 | 433962 | 0.99025 |
| internal_val_WSIs | neighbor_edge_mean_diff | 667125 | 239562 | 0.3591 |
| training_WSIs | neighbor_edge_mean_diff | 712422 | 243748 | 0.34214 |
| internal_val_WSIs | neighbor_edge_mean | 667125 | 221963 | 0.33272 |
| training_WSIs | neighbor_edge_mean | 712422 | 210749 | 0.29582 |
| training_WSIs | edge_sector_04_high_fraction | 712422 | 72594 | 0.1019 |
| training_WSIs | edge_sector_10_high_fraction | 712422 | 70936 | 0.09957 |
| training_WSIs | edge_sector_05_high_fraction | 712422 | 69563 | 0.09764 |
| internal_val_WSIs | edge_sector_04_high_fraction | 667125 | 30754 | 0.0461 |
| internal_val_WSIs | edge_sector_10_high_fraction | 667125 | 29117 | 0.04365 |
| internal_val_WSIs | edge_sector_05_high_fraction | 667125 | 28984 | 0.04345 |

上述分母不是有标签训练细胞数。逐单位/真值缺失与 QC 见 feature_missingness_by_unit_class.csv、qc_by_split_class.csv。插补后能够评分不表示原观测完整，未依据缺失直接赋阴性。

数值依据：[report_numeric_resolution.csv](results/report_numeric_resolution.csv)；筛选：is_hu_moment=True，仅有标签训练细胞；指标列：`raw_nunique_train,transformed_float32_nunique_train,unique_retention_ratio,resolution_note`。

| Hu特征 | 原始不同值数 | 转换后不同值数 | 保留比例 | 诊断 |
| --- | --- | --- | --- | --- |
| HuMoments1 | 100911 | 100759 | 0.99849 | no_large_unique_loss_by_audit_rule |
| HuMoments2 | 100995 | 100888 | 0.99894 | no_large_unique_loss_by_audit_rule |
| HuMoments3 | 100956 | 100835 | 0.9988 | no_large_unique_loss_by_audit_rule |
| HuMoments4 | 100963 | 100604 | 0.99644 | no_large_unique_loss_by_audit_rule |
| HuMoments5 | 100964 | 82244 | 0.81459 | no_large_unique_loss_by_audit_rule |
| HuMoments6 | 100963 | 100029 | 0.99075 | no_large_unique_loss_by_audit_rule |
| HuMoments7 | 100939 | 70378 | 0.69723 | no_large_unique_loss_by_audit_rule |

官方 QT 固定绝对尺度拟合噪声可能压缩高阶 Hu moments 的分辨率；compute_bins 成功只说明仍有不同值。本轮仅记录 float32 诊断，没有因验证表现改归一化、伪造分箱或重选特征。

数值依据：[report_disagreement.csv](results/report_disagreement.csv)；筛选：split=external_val_TMA_cores,group_type=split,true_class=all；指标列：`std_mean_mean/std,std_p95_mean/std`。

| 模型 | 逐细胞成员SD均值 | 逐细胞成员SD p95 |
| --- | --- | --- |
| TabM_A | 0.0008 ± 0.0003 | 0.0015 ± 0.0008 |
| TabM_B | 0.0013 ± 0.0003 | 0.0019 ± 0.0006 |
| TabM_C | 0.0018 ± 0.0003 | 0.0021 ± 0.0004 |

每个 TabM 的全部成员阳性概率随逐细胞表保留。分歧不是已校准不确定性，不据此自动拒判或删除细胞。

## 9. 资源、交付与复现

数值依据：[report_resources.csv](results/report_resources.csv)；筛选：逐canonical模型；下表为各组合三种子均值；指标列：`training_optimization_seconds,validation_seconds,inference_seconds,training_peak_phase_rss_GiB,training_rss_increase_GiB,training_peak_gpu_allocated_GiB`。

| 模型 | 组合 | 优化秒 | 验证秒 | 全量推理秒 | 阶段RSS峰GiB | RSS增量GiB | GPU分配峰GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGB | A | 0.412 | 0.795 | 3.174 | 3.044 | 0.032 | 0.0 |
| HGB | B | 0.662 | 0.821 | 3.171 | 3.204 | 0.108 | 0.0 |
| HGB | C | 1.699 | 0.961 | 3.864 | 3.617 | 0.362 | 0.0 |
| TabM | A | 101.376 | 157.814 | 12.223 | 3.132 | 0.717 | 0.216 |
| TabM | B | 69.103 | 109.798 | 16.368 | 3.217 | 0.129 | 0.229 |
| TabM | C | 117.823 | 200.22 | 31.879 | 3.384 | 0.128 | 0.32 |

硬件记录：`Quadro RTX 4000`。TabM 用 GPU，HGB 用同机 CPU，因此是实际配置的资源对照，不是等算力速度排名。TabM 验证时间包含每 epoch 监测，HGB 为验证推理。阶段 RSS 以 psutil 采样，含开始时驻留数据，故另报基线和增量；不以进程历史 ru_maxrss 充当当前模型峰值。GPU 指标是分配内存，不是显卡总占用。

服务器运行根目录：`/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_tabm_robust_20261030`。权重绝对路径、SHA256 与门限见 report_model_artifacts.csv；预处理位于 models/preprocessor.joblib，完整逐细胞文件位于 predictions/，含统一键、原标签、分组、评分、两套预测和成员输出。版本见 RUN_CONFIG.json；上游 commit 见 report_provenance.csv；本项目源码以 results/code_sha256.csv 和报告 manifest 标识，不捏造未记录的本地 Git commit。

复现已有冻结流程（入口校验数据/代码/配置；新科学重复运行应使用新目录保留旧输出）：

```bash
R="/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_tabm_robust_20261030"
PY="/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_nimbus_20261024/.venv/bin/python"
flock -n "$R/../.cd8_robust_gpu.lock" "$PY" "$R/scripts/run_pipeline.py" --run-dir "$R" --stage all --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
"$PY" "$R/scripts/audit_numeric_resolution.py" --run-dir "$R"
"$PY" "$R/scripts/run_pipeline.py" --run-dir "$R" --stage validate
"$PY" "$R/scripts/write_tabm_report.py" --run-dir "$R"
```
数值诊断已经存在时跳过对应命令，该脚本拒绝覆盖历史诊断。

新表推理调用示例（C 只是接口示例，不表示按测试结果推荐它）：

```bash
flock -n "$R/../.cd8_robust_gpu.lock" "$PY" "$R/scripts/predict_table.py" --run-dir "$R" --model-id TabM_C_s0 --input /path/to/keyed_features.parquet --output /path/to/new_predictions.parquet
```
输入必须具有显式键、normalization_method、calibration_valid，以及同一 E1_outer_P1_P99 协议下从归一化像素重新提取、尚未经过训练 QT 变换的特征；插补和 QT 由已保存的训练预处理器完成。已存在输出拒绝覆盖。

PNG 和文字可编辑 SVG 位于 figures/zh、figures/en；源数值位于 figures/source_data；各split及TMA分组的完整 ROC/PR 和 threshold 数组位于 curves/。曲线采用 sklearn 的 >= 门限约定，正式分类表使用预定严格 >。

## 10. 限制

内部 WSI 用于开发，外部 TMA 在模型和门限冻结后评价。WSI 数量有限，种子波动不是患者级不确定性。扩张核掩码、参照缺失和像素尺度限制膜定位与背景机制解释；弱阳性/接触侧代理仍需独立人工复核。C 的 63 项空间特征来自本轮归一化像素重提，19 项形态沿用原分割与原几何计算；空间特征含固定方位的十二扇区值，B/C 形态还含 Orientation。此次没有新增旋转稳定性检验，不能继承 E1–E4 的旋转不变性结论。本轮只评价 PhenoBIC CD8，不扩展 marker，也不声称封存 Pan-M 的泛化表现。

官方来源：[TabM](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4)；[训练示例](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb)；[论文实验说明](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/README.md)。
