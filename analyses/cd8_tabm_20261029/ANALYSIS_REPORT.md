# PhenoBIC CD8：TabM A/B/C 与匹配 HGB 的完整实验报告

本报告仅依据已冻结的完整实验。所有数值表的来源、筛选条件、聚合方式及指标列见 [report_tables.csv](results/report_tables.csv)。均值 ± 标准差是种子间描述性波动，不是患者置信区间。

## 1. 主要结论与完成边界

数值依据：[report_performance_fixed_0.5.csv](results/report_performance_fixed_0.5.csv)；筛选：split=external_val_TMA_cores,group_type=split,model_family=TabM；指标列：`f1_mean,f1_std`。

固定 0.5 门限的完整外部细胞集合上，TabM A/B/C 中平均 F1 最高为 **C**，F1 为 **0.5423 ± 0.0052**。该排序是冻结后的描述，不用于重选模型、特征或门限。

- TabM_B_minus_A：总体 F1 提高，ΔF1=0.0003 ± 0.0048；平均 ΔTP=-1.3、ΔFP=-21.3、ΔFN=1.3。

- TabM_C_minus_B：总体 F1 提高，ΔF1=0.0096 ± 0.0087；平均 ΔTP=97.3、ΔFP=-313.7、ΔFN=-97.3。

- TabM_C_minus_A：总体 F1 提高，ΔF1=0.0098 ± 0.0040；平均 ΔTP=96.0、ΔFP=-335.0、ΔFN=-96.0。

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

数值依据：[report_feature_sets.csv](results/report_feature_sets.csv)；筛选：feature_set=A/B/C；指标列：`n_requested,n_effective,excluded`。

| 组合 | 请求维数 | 有效维数 | 排除原因 |
| --- | --- | --- | --- |
| A | 8 | 8 | {} |
| B | 27 | 27 | {} |
| C | 90 | 90 | {} |

A 为原始 CD8 基础强度；B 加入指定形态特征；C 再加入现有边缘/内部、局部参照、角向与邻居特征。完整名称和顺序见 FEATURE_SETS.json、results/feature_manifest.csv。标识、患者/批次、真值和 GMM 不进入模型。当前 mask 为 expanded_nucleus；其边缘是扩张核近似边缘，不是经过验证的真实膜。长度和面积保持像素尺度。

中位数插补、带拟合噪声的正态分位数变换、PLE 分箱均仅训练拟合。HGB 使用相同有效特征和预处理，便于区分特征组合与学习器收益；它与历史未经该预处理的 HGB 是不同实验条件。

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
| TabM_A_s0 | 0.0005 | 24.0 | 40.0 | False | 0.76 |
| TabM_A_s1 | 0.0005 | 12.0 | 28.0 | False | 0.74 |
| TabM_A_s2 | 0.0005 | 21.0 | 37.0 | False | 0.7 |
| TabM_B_s0 | 0.002 | 4.0 | 20.0 | False | 0.78 |
| TabM_B_s1 | 0.002 | 12.0 | 28.0 | False | 0.78 |
| TabM_B_s2 | 0.002 | 6.0 | 22.0 | False | 0.68 |
| TabM_C_s0 | 0.002 | 4.0 | 20.0 | False | 0.84 |
| TabM_C_s1 | 0.002 | 5.0 | 21.0 | False | 0.78 |
| TabM_C_s2 | 0.002 | 1.0 | 17.0 | False | 0.87 |

风险分层在最终模型/测试协议前冻结，哈希核验通过。外部 PhenoBIC 在本轮方案冻结后评价；该数据此前已经用于本项目其他实验，不能称为整个研究过程从未接触过的全新测试集。

## 4. 固定 0.5 主结果

数值依据：[report_performance_fixed_0.5.csv](results/report_performance_fixed_0.5.csv)；筛选：group_type=split；内部开发验证与外部测试分别如下；GMM始终fixed_0.5；指标列：`precision/recall/f1/average_precision/positive_fraction_absolute_error_mean及_std`。

### 内部开发验证

| 模型 | Precision | Recall | F1 | AP | 阳性比例绝对误差 |
| --- | --- | --- | --- | --- | --- |
| GMM（固定0.5） | 0.7513 | 0.9382 | 0.8344 | 0.9481 | 0.0088 |
| HGB_A | 0.9264 ± 0.0000 | 0.9245 ± 0.0000 | 0.9255 ± 0.0000 | 0.9783 ± 0.0000 | 0.0001 ± 0.0000 |
| HGB_B | 0.9276 ± 0.0000 | 0.9235 ± 0.0000 | 0.9255 ± 0.0000 | 0.9783 ± 0.0000 | 0.0002 ± 0.0000 |
| HGB_C | 0.9273 ± 0.0000 | 0.9391 ± 0.0000 | 0.9332 ± 0.0000 | 0.9826 ± 0.0000 | 0.0004 ± 0.0000 |
| TabM_A | 0.9228 ± 0.0053 | 0.9300 ± 0.0033 | 0.9264 ± 0.0010 | 0.9796 ± 0.0001 | 0.0003 ± 0.0002 |
| TabM_B | 0.9225 ± 0.0102 | 0.9287 ± 0.0104 | 0.9255 ± 0.0008 | 0.9792 ± 0.0004 | 0.0007 ± 0.0001 |
| TabM_C | 0.9040 ± 0.0218 | 0.9527 ± 0.0089 | 0.9275 ± 0.0074 | 0.9835 ± 0.0005 | 0.0019 ± 0.0012 |

### 外部测试

| 模型 | Precision | Recall | F1 | AP | 阳性比例绝对误差 |
| --- | --- | --- | --- | --- | --- |
| GMM（固定0.5） | 0.6547 | 0.9086 | 0.7610 | 0.8302 | 0.0198 |
| HGB_A | 0.8324 ± 0.0000 | 0.3923 ± 0.0000 | 0.5333 ± 0.0000 | 0.5476 ± 0.0000 | 0.0270 ± 0.0000 |
| HGB_B | 0.8501 ± 0.0000 | 0.3869 ± 0.0000 | 0.5318 ± 0.0000 | 0.5786 ± 0.0000 | 0.0278 ± 0.0000 |
| HGB_C | 0.8730 ± 0.0000 | 0.3892 ± 0.0000 | 0.5384 ± 0.0000 | 0.6156 ± 0.0000 | 0.0283 ± 0.0000 |
| TabM_A | 0.8402 ± 0.0020 | 0.3898 ± 0.0022 | 0.5325 ± 0.0018 | 0.7590 ± 0.0045 | 0.0274 ± 0.0002 |
| TabM_B | 0.8419 ± 0.0022 | 0.3897 ± 0.0049 | 0.5328 ± 0.0043 | 0.6877 ± 0.0468 | 0.0274 ± 0.0004 |
| TabM_C | 0.8695 ± 0.0060 | 0.3941 ± 0.0043 | 0.5423 ± 0.0052 | 0.6706 ± 0.0684 | 0.0279 ± 0.0001 |

总体指标先在每个种子的完整有标签集合上计算，再汇总种子波动；它们是按细胞汇总结果。训练重代入成绩、AUC、specificity、TP/FP/FN/TN、真实/预测阳性比例与覆盖率都保存在对应 CSV。此次没有新 OOF；训练成绩不是 OOF，内部验证是明确的开发集。

### 外部 TMA 分组

数值依据：[report_performance_fixed_0.5.csv](results/report_performance_fixed_0.5.csv)；筛选：split=external_val_TMA_cores,group_type=context,context=TMA1_CD8/TMA2_CD8/TMA3_CD8；指标列：`f1/precision/recall/roc_auc/average_precision/fp/fn_mean及_std`。

| TMA | 模型 | F1 | Precision | Recall | AP | AUC | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TMA1_CD8 | GMM（固定0.5） | 0.8534 | 0.8975 | 0.8135 | 0.9399 | 0.9968 | 676.0000 | 1357.0000 |
| TMA2_CD8 | GMM（固定0.5） | 0.8166 | 0.7416 | 0.9084 | 0.8935 | 0.9951 | 2173.0000 | 629.0000 |
| TMA3_CD8 | GMM（固定0.5） | 0.6728 | 0.5087 | 0.9933 | 0.9232 | 0.9946 | 7854.0000 | 55.0000 |
| TMA1_CD8 | HGB_A | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1034 ± 0.0000 | 0.5291 ± 0.0000 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | HGB_A | 0.2446 ± 0.0000 | 0.9917 ± 0.0000 | 0.1395 ± 0.0000 | 0.8348 ± 0.0000 | 0.9344 ± 0.0000 | 8.0000 ± 0.0000 | 5909.0000 ± 0.0000 |
| TMA3_CD8 | HGB_A | 0.8793 ± 0.0000 | 0.8163 ± 0.0000 | 0.9529 ± 0.0000 | 0.9294 ± 0.0000 | 0.9964 ± 0.0000 | 1756.0000 ± 0.0000 | 386.0000 ± 0.0000 |
| TMA1_CD8 | HGB_B | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1094 ± 0.0000 | 0.5486 ± 0.0000 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | HGB_B | 0.2295 ± 0.0000 | 0.9933 ± 0.0000 | 0.1298 ± 0.0000 | 0.8430 ± 0.0000 | 0.9368 ± 0.0000 | 6.0000 ± 0.0000 | 5976.0000 ± 0.0000 |
| TMA3_CD8 | HGB_B | 0.8879 ± 0.0000 | 0.8362 ± 0.0000 | 0.9464 ± 0.0000 | 0.9401 ± 0.0000 | 0.9968 ± 0.0000 | 1518.0000 ± 0.0000 | 439.0000 ± 0.0000 |
| TMA1_CD8 | HGB_C | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.3028 ± 0.0000 | 0.5603 ± 0.0000 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | HGB_C | 0.2428 ± 0.0000 | 0.9927 ± 0.0000 | 0.1383 ± 0.0000 | 0.8702 ± 0.0000 | 0.9501 ± 0.0000 | 7.0000 ± 0.0000 | 5917.0000 ± 0.0000 |
| TMA3_CD8 | HGB_C | 0.9009 ± 0.0000 | 0.8602 ± 0.0000 | 0.9455 ± 0.0000 | 0.9613 ± 0.0000 | 0.9976 ± 0.0000 | 1258.0000 ± 0.0000 | 446.0000 ± 0.0000 |
| TMA1_CD8 | TabM_A | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9312 ± 0.0059 | 0.9957 ± 0.0026 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | TabM_A | 0.2334 ± 0.0072 | 0.9916 ± 0.0005 | 0.1323 ± 0.0047 | 0.9384 ± 0.0010 | 0.9966 ± 0.0006 | 7.6667 ± 0.5774 | 5958.6667 ± 31.9739 |
| TMA3_CD8 | TabM_A | 0.8843 ± 0.0005 | 0.8255 ± 0.0025 | 0.9521 ± 0.0024 | 0.9591 ± 0.0004 | 0.9974 ± 0.0000 | 1647.6667 ± 32.1921 | 392.3333 ± 19.5021 |
| TMA1_CD8 | TabM_B | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6950 ± 0.1310 | 0.9766 ± 0.0145 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | TabM_B | 0.2335 ± 0.0242 | 0.9924 ± 0.0007 | 0.1325 ± 0.0155 | 0.9237 ± 0.0086 | 0.9952 ± 0.0007 | 7.0000 ± 1.0000 | 5957.3333 ± 106.2560 |
| TMA3_CD8 | TabM_B | 0.8852 ± 0.0020 | 0.8273 ± 0.0038 | 0.9518 ± 0.0005 | 0.9605 ± 0.0060 | 0.9974 ± 0.0002 | 1627.0000 ± 44.3959 | 395.0000 ± 4.3589 |
| TMA1_CD8 | TabM_C | 0.0001 ± 0.0002 | 0.3333 ± 0.5774 | 0.0000 ± 0.0001 | 0.7109 ± 0.3296 | 0.9682 ± 0.0403 | 0.0000 ± 0.0000 | 7275.6667 ± 0.5774 |
| TMA2_CD8 | TabM_C | 0.2497 ± 0.0235 | 0.9922 ± 0.0003 | 0.1430 ± 0.0153 | 0.8812 ± 0.0659 | 0.9868 ± 0.0106 | 7.6667 ± 0.5774 | 5885.3333 ± 104.9778 |
| TMA3_CD8 | TabM_C | 0.9028 ± 0.0023 | 0.8563 ± 0.0051 | 0.9548 ± 0.0012 | 0.9720 ± 0.0020 | 0.9978 ± 0.0002 | 1312.6667 ± 55.7704 | 370.0000 ± 9.5394 |

## 5. 验证门限补充结果

数值依据：[report_performance_validation_selected.csv](results/report_performance_validation_selected.csv)；筛选：group_type=split；内部开发验证与外部测试分别如下；GMM始终fixed_0.5；指标列：`precision/recall/f1/average_precision/positive_fraction_absolute_error_mean及_std`。

### 内部开发验证

| 模型 | Precision | Recall | F1 | AP | 阳性比例绝对误差 |
| --- | --- | --- | --- | --- | --- |
| GMM（固定0.5） | 0.7513 | 0.9382 | 0.8344 | 0.9481 | 0.0088 |
| HGB_A | 0.9698 ± 0.0000 | 0.8741 ± 0.0000 | 0.9195 ± 0.0000 | 0.9783 ± 0.0000 | 0.0035 ± 0.0000 |
| HGB_B | 0.9666 ± 0.0000 | 0.8802 ± 0.0000 | 0.9214 ± 0.0000 | 0.9783 ± 0.0000 | 0.0031 ± 0.0000 |
| HGB_C | 0.9641 ± 0.0000 | 0.9009 ± 0.0000 | 0.9315 ± 0.0000 | 0.9826 ± 0.0000 | 0.0023 ± 0.0000 |
| TabM_A | 0.9658 ± 0.0012 | 0.8825 ± 0.0015 | 0.9223 ± 0.0005 | 0.9796 ± 0.0001 | 0.0030 ± 0.0001 |
| TabM_B | 0.9676 ± 0.0013 | 0.8763 ± 0.0011 | 0.9197 ± 0.0006 | 0.9792 ± 0.0004 | 0.0033 ± 0.0001 |
| TabM_C | 0.9710 ± 0.0021 | 0.8903 ± 0.0080 | 0.9289 ± 0.0034 | 0.9835 ± 0.0005 | 0.0029 ± 0.0004 |

### 外部测试

| 模型 | Precision | Recall | F1 | AP | 阳性比例绝对误差 |
| --- | --- | --- | --- | --- | --- |
| GMM（固定0.5） | 0.6547 | 0.9086 | 0.7610 | 0.8302 | 0.0198 |
| HGB_A | 0.8456 ± 0.0000 | 0.3750 ± 0.0000 | 0.5196 ± 0.0000 | 0.5476 ± 0.0000 | 0.0284 ± 0.0000 |
| HGB_B | 0.8598 ± 0.0000 | 0.3704 ± 0.0000 | 0.5178 ± 0.0000 | 0.5786 ± 0.0000 | 0.0291 ± 0.0000 |
| HGB_C | 0.8887 ± 0.0000 | 0.3717 ± 0.0000 | 0.5242 ± 0.0000 | 0.6156 ± 0.0000 | 0.0297 ± 0.0000 |
| TabM_A | 0.8562 ± 0.0010 | 0.3699 ± 0.0016 | 0.5166 ± 0.0017 | 0.7590 ± 0.0045 | 0.0290 ± 0.0001 |
| TabM_B | 0.8590 ± 0.0038 | 0.3681 ± 0.0017 | 0.5153 ± 0.0011 | 0.6877 ± 0.0468 | 0.0292 ± 0.0002 |
| TabM_C | 0.9104 ± 0.0154 | 0.3605 ± 0.0039 | 0.5165 ± 0.0016 | 0.6706 ± 0.0684 | 0.0308 ± 0.0006 |

总体指标先在每个种子的完整有标签集合上计算，再汇总种子波动；它们是按细胞汇总结果。训练重代入成绩、AUC、specificity、TP/FP/FN/TN、真实/预测阳性比例与覆盖率都保存在对应 CSV。此次没有新 OOF；训练成绩不是 OOF，内部验证是明确的开发集。

### 外部 TMA 分组

数值依据：[report_performance_validation_selected.csv](results/report_performance_validation_selected.csv)；筛选：split=external_val_TMA_cores,group_type=context,context=TMA1_CD8/TMA2_CD8/TMA3_CD8；指标列：`f1/precision/recall/roc_auc/average_precision/fp/fn_mean及_std`。

| TMA | 模型 | F1 | Precision | Recall | AP | AUC | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TMA1_CD8 | GMM（固定0.5） | 0.8534 | 0.8975 | 0.8135 | 0.9399 | 0.9968 | 676.0000 | 1357.0000 |
| TMA2_CD8 | GMM（固定0.5） | 0.8166 | 0.7416 | 0.9084 | 0.8935 | 0.9951 | 2173.0000 | 629.0000 |
| TMA3_CD8 | GMM（固定0.5） | 0.6728 | 0.5087 | 0.9933 | 0.9232 | 0.9946 | 7854.0000 | 55.0000 |
| TMA1_CD8 | HGB_A | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1034 ± 0.0000 | 0.5291 ± 0.0000 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | HGB_A | 0.1611 ± 0.0000 | 0.9918 ± 0.0000 | 0.0877 ± 0.0000 | 0.8348 ± 0.0000 | 0.9344 ± 0.0000 | 5.0000 ± 0.0000 | 6265.0000 ± 0.0000 |
| TMA3_CD8 | HGB_A | 0.8891 ± 0.0000 | 0.8361 ± 0.0000 | 0.9493 ± 0.0000 | 0.9294 ± 0.0000 | 0.9964 ± 0.0000 | 1524.0000 ± 0.0000 | 415.0000 ± 0.0000 |
| TMA1_CD8 | HGB_B | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1094 ± 0.0000 | 0.5486 ± 0.0000 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | HGB_B | 0.1514 ± 0.0000 | 0.9912 ± 0.0000 | 0.0820 ± 0.0000 | 0.8430 ± 0.0000 | 0.9368 ± 0.0000 | 5.0000 ± 0.0000 | 6304.0000 ± 0.0000 |
| TMA3_CD8 | HGB_B | 0.8943 ± 0.0000 | 0.8515 ± 0.0000 | 0.9415 ± 0.0000 | 0.9401 ± 0.0000 | 0.9968 ± 0.0000 | 1344.0000 ± 0.0000 | 479.0000 ± 0.0000 |
| TMA1_CD8 | HGB_C | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.3028 ± 0.0000 | 0.5603 ± 0.0000 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | HGB_C | 0.1631 ± 0.0000 | 0.9919 ± 0.0000 | 0.0888 ± 0.0000 | 0.8702 ± 0.0000 | 0.9501 ± 0.0000 | 5.0000 ± 0.0000 | 6257.0000 ± 0.0000 |
| TMA3_CD8 | HGB_C | 0.9094 ± 0.0000 | 0.8814 ± 0.0000 | 0.9393 ± 0.0000 | 0.9613 ± 0.0000 | 0.9976 ± 0.0000 | 1035.0000 ± 0.0000 | 497.0000 ± 0.0000 |
| TMA1_CD8 | TabM_A | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9312 ± 0.0059 | 0.9957 ± 0.0026 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | TabM_A | 0.1423 ± 0.0097 | 0.9906 ± 0.0007 | 0.0766 ± 0.0057 | 0.9384 ± 0.0010 | 0.9966 ± 0.0006 | 5.0000 ± 0.0000 | 6340.6667 ± 38.8887 |
| TMA3_CD8 | TabM_A | 0.8939 ± 0.0002 | 0.8484 ± 0.0008 | 0.9445 ± 0.0006 | 0.9591 ± 0.0004 | 0.9974 ± 0.0000 | 1382.0000 ± 9.5394 | 454.6667 ± 4.9329 |
| TMA1_CD8 | TabM_B | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.6950 ± 0.1310 | 0.9766 ± 0.0145 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | TabM_B | 0.1356 ± 0.0073 | 0.9907 ± 0.0011 | 0.0728 ± 0.0042 | 0.9237 ± 0.0086 | 0.9952 ± 0.0007 | 4.6667 ± 0.5774 | 6367.0000 ± 29.1033 |
| TMA3_CD8 | TabM_B | 0.8949 ± 0.0019 | 0.8517 ± 0.0044 | 0.9427 ± 0.0010 | 0.9605 ± 0.0060 | 0.9974 ± 0.0002 | 1344.6667 ± 47.6480 | 469.0000 ± 8.5440 |
| TMA1_CD8 | TabM_C | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7109 ± 0.3296 | 0.9682 ± 0.0403 | 0.0000 ± 0.0000 | 7276.0000 ± 0.0000 |
| TMA2_CD8 | TabM_C | 0.1247 ± 0.0085 | 0.9921 ± 0.0007 | 0.0666 ± 0.0048 | 0.8812 ± 0.0659 | 0.9868 ± 0.0106 | 3.6667 ± 0.5774 | 6410.0000 ± 33.1512 |
| TMA3_CD8 | TabM_C | 0.9165 ± 0.0046 | 0.9059 ± 0.0163 | 0.9275 ± 0.0080 | 0.9720 ± 0.0020 | 0.9978 ± 0.0002 | 791.6667 ± 159.6444 | 593.6667 ± 65.1562 |

## 6. 配对消融、架构与逐组织芯差异

数值依据：[report_paired_deltas.csv](results/report_paired_deltas.csv)；筛选：split=external_val_TMA_cores,group_type=split；两套threshold_mode分别标记；指标列：`delta_f1/average_precision/tp/fp/fn_mean及_std`。

| 比较 | 门限 | ΔF1 | ΔTP | ΔFP | ΔFN |
| --- | --- | --- | --- | --- | --- |
| HGB_B_minus_A | fixed_0.5 | -0.0015 ± 0.0000 | -120.0 ± 0.0 | -240.0 ± 0.0 | 120.0 ± 0.0 |
| HGB_B_minus_A | validation_selected | -0.0018 ± 0.0000 | -103.0 ± 0.0 | -180.0 ± 0.0 | 103.0 ± 0.0 |
| HGB_C_minus_A | fixed_0.5 | 0.0052 ± 0.0000 | -68.0 ± 0.0 | -499.0 ± 0.0 | 68.0 ± 0.0 |
| HGB_C_minus_A | validation_selected | 0.0046 ± 0.0000 | -74.0 ± 0.0 | -489.0 ± 0.0 | 74.0 ± 0.0 |
| HGB_C_minus_B | fixed_0.5 | 0.0066 ± 0.0000 | 52.0 ± 0.0 | -259.0 ± 0.0 | -52.0 ± 0.0 |
| HGB_C_minus_B | validation_selected | 0.0064 ± 0.0000 | 29.0 ± 0.0 | -309.0 ± 0.0 | -29.0 ± 0.0 |
| TabM_A_minus_GMM | fixed_0.5 | -0.2285 ± 0.0018 | -11586.0 ± 50.1 | -9047.7 ± 32.3 | 11586.0 ± 50.1 |
| TabM_A_minus_GMM | validation_selected | -0.2444 ± 0.0017 | -12030.3 ± 36.7 | -9316.0 ± 9.5 | 12030.3 ± 36.7 |
| TabM_B_minus_A | fixed_0.5 | 0.0003 ± 0.0048 | -1.3 ± 116.0 | -21.3 ± 27.3 | 1.3 ± 116.0 |
| TabM_B_minus_A | validation_selected | -0.0013 ± 0.0006 | -40.7 ± 10.1 | -37.7 ± 41.8 | 40.7 ± 10.1 |
| TabM_B_minus_GMM | fixed_0.5 | -0.2282 ± 0.0043 | -11587.3 ± 110.3 | -9069.0 ± 45.3 | 11587.3 ± 110.3 |
| TabM_B_minus_GMM | validation_selected | -0.2457 ± 0.0011 | -12071.0 ± 37.0 | -9353.7 ± 47.5 | 12071.0 ± 37.0 |
| TabM_C_minus_A | fixed_0.5 | 0.0098 ± 0.0040 | 96.0 ± 69.8 | -335.0 ± 71.0 | -96.0 ± 69.8 |
| TabM_C_minus_A | validation_selected | -0.0001 ± 0.0032 | -208.3 ± 114.8 | -591.7 ± 167.8 | 208.3 ± 114.8 |
| TabM_C_minus_B | fixed_0.5 | 0.0096 ± 0.0087 | 97.3 ± 185.7 | -313.7 ± 57.5 | -97.3 ± 185.7 |
| TabM_C_minus_B | validation_selected | 0.0012 ± 0.0027 | -167.7 ± 119.8 | -554.0 ± 206.4 | 167.7 ± 119.8 |
| TabM_C_minus_GMM | fixed_0.5 | -0.2187 ± 0.0052 | -11490.0 ± 95.7 | -9382.7 ± 55.2 | 11490.0 ± 95.7 |
| TabM_C_minus_GMM | validation_selected | -0.2445 ± 0.0016 | -12238.7 ± 86.6 | -9907.7 ± 159.9 | 12238.7 ± 86.6 |
| TabM_minus_HGB_A | fixed_0.5 | -0.0007 ± 0.0018 | -56.0 ± 50.1 | -108.7 ± 32.3 | 56.0 ± 50.1 |
| TabM_minus_HGB_A | validation_selected | -0.0030 ± 0.0017 | -115.3 ± 36.7 | -142.0 ± 9.5 | 115.3 ± 36.7 |
| TabM_minus_HGB_B | fixed_0.5 | 0.0010 ± 0.0043 | 62.7 ± 110.3 | 110.0 ± 45.3 | -62.7 ± 110.3 |
| TabM_minus_HGB_B | validation_selected | -0.0025 ± 0.0011 | -53.0 ± 37.0 | 0.3 ± 47.5 | 53.0 ± 37.0 |
| TabM_minus_HGB_C | fixed_0.5 | 0.0039 ± 0.0052 | 108.0 ± 95.7 | 55.3 ± 55.2 | -108.0 ± 95.7 |
| TabM_minus_HGB_C | validation_selected | -0.0077 ± 0.0016 | -249.7 ± 86.6 | -244.7 ± 159.9 | 249.7 ± 86.6 |

同特征 TabM−HGB 辨别学习器差异；B−A 衡量形态组合，C−B 衡量空间组合。不能把 F1 改善直接归因于某一特征或污染消除。GMM 为无监督参考，新模型使用训练标签，监督条件不同。

数值依据：[report_unit_equal.csv](results/report_unit_equal.csv)；筛选：split=external_val_TMA_cores,metric=f1；先逐芯等权再种子汇总；指标列：`unit_equal_mean_mean/std,n_finite_units_mean,n_units_no_positive_predictions_mean`。

| 模型 | 门限 | 逐芯等权F1 | 有效芯数 | 无阳性预测芯数均值 |
| --- | --- | --- | --- | --- |
| GMM（固定0.5） | fixed_0.5 | 0.7667 | 42.0 | 0.0 |
| HGB_A | fixed_0.5 | 0.4854 ± 0.0000 | 42.0 | 12.0 |
| HGB_A | validation_selected | 0.4606 ± 0.0000 | 42.0 | 12.0 |
| HGB_B | fixed_0.5 | 0.4827 ± 0.0000 | 42.0 | 12.0 |
| HGB_B | validation_selected | 0.4590 ± 0.0000 | 42.0 | 12.0 |
| HGB_C | fixed_0.5 | 0.4926 ± 0.0000 | 42.0 | 12.0 |
| HGB_C | validation_selected | 0.4690 ± 0.0000 | 42.0 | 12.0 |
| TabM_A | fixed_0.5 | 0.4824 ± 0.0019 | 42.0 | 12.0 |
| TabM_A | validation_selected | 0.4559 ± 0.0034 | 42.0 | 12.0 |
| TabM_B | fixed_0.5 | 0.4827 ± 0.0068 | 42.0 | 12.0 |
| TabM_B | validation_selected | 0.4539 ± 0.0022 | 42.0 | 12.0 |
| TabM_C | fixed_0.5 | 0.4959 ± 0.0081 | 42.0 | 11.667 |
| TabM_C | validation_selected | 0.4572 ± 0.0031 | 42.0 | 12.0 |

数值依据：[report_lowest_cores.csv](results/report_lowest_cores.csv)；筛选：external, fixed_0.5；每个组合按三种子平均F1最低三个组织芯；指标列：`group_id,context,f1/precision/recall_mean及_std`。

| 模型 | TMA | 组织芯 | F1 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| HGB_C | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_9_8 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| HGB_C | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_6_2 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| HGB_C | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_5_7 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_C | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_8_3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_C | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_2_1 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| HGB_B | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_9_8 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| HGB_B | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_8_3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| HGB_B | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_6_4 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| HGB_A | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_2_1 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| HGB_A | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_8_3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| HGB_A | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_6_4 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_C | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_6_4 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_B | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_5_7 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_A | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_5_7 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_A | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_5_5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_B | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_2_5 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_B | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_2_3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_A | TMA1_CD8 | external_val_TMA_cores/TMA1_CD8/core_6_2 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |

完整分布见 metrics_all.csv、unit_performance_ranked.csv、paired_comparisons.csv。无真实阳性时 recall/AP=NaN；无阳性预测时 precision=0；真值与预测都无阳性时 F1=NaN。宏平均只使用相应指标有限的单位并记录数量，不把 NaN 填零。未做细胞级 bootstrap 或将细胞作为独立样本的显著性检验。

## 7. 弱阳性、接触侧亮信号代理与案例

数值依据：[report_risk_strata.csv](results/report_risk_strata.csv)；筛选：group_type=split,stratum=weak_positive/contact_bright_proxy_negative；按来源和门限分别比较；指标列：`n_cells_mean,recall_mean/std,false_positive_rate_mean/std,fp_mean,fn_mean`。

| 来源 | 门限 | 模型 | 分层 | 细胞数 | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| external_val_TMA_cores | fixed_0.5 | TabM_A | contact_bright_proxy_negative | 9233.0 | NA | 0.0990 ± 0.0025 |
| external_val_TMA_cores | fixed_0.5 | TabM_A | weak_positive | 13991.0 | 0.0499 ± 0.0023 | NA |
| internal_val_WSIs | fixed_0.5 | TabM_A | contact_bright_proxy_negative | 10899.0 | NA | 0.0491 ± 0.0033 |
| internal_val_WSIs | fixed_0.5 | TabM_A | weak_positive | 3153.0 | 0.6094 ± 0.0143 | NA |
| external_val_TMA_cores | validation_selected | TabM_A | contact_bright_proxy_negative | 9233.0 | NA | 0.0830 ± 0.0007 |
| external_val_TMA_cores | validation_selected | TabM_A | weak_positive | 13991.0 | 0.0223 ± 0.0028 | NA |
| internal_val_WSIs | validation_selected | TabM_A | contact_bright_proxy_negative | 10899.0 | NA | 0.0244 ± 0.0004 |
| internal_val_WSIs | validation_selected | TabM_A | weak_positive | 3153.0 | 0.3765 ± 0.0114 | NA |
| external_val_TMA_cores | fixed_0.5 | TabM_B | contact_bright_proxy_negative | 9233.0 | NA | 0.0976 ± 0.0027 |
| external_val_TMA_cores | fixed_0.5 | TabM_B | weak_positive | 13991.0 | 0.0500 ± 0.0076 | NA |
| internal_val_WSIs | fixed_0.5 | TabM_B | contact_bright_proxy_negative | 10899.0 | NA | 0.0497 ± 0.0063 |
| internal_val_WSIs | fixed_0.5 | TabM_B | weak_positive | 3153.0 | 0.5978 ± 0.0566 | NA |
| external_val_TMA_cores | validation_selected | TabM_B | contact_bright_proxy_negative | 9233.0 | NA | 0.0804 ± 0.0028 |
| external_val_TMA_cores | validation_selected | TabM_B | weak_positive | 13991.0 | 0.0205 ± 0.0020 | NA |
| internal_val_WSIs | validation_selected | TabM_B | contact_bright_proxy_negative | 10899.0 | NA | 0.0226 ± 0.0009 |
| internal_val_WSIs | validation_selected | TabM_B | weak_positive | 3153.0 | 0.3510 ± 0.0056 | NA |
| external_val_TMA_cores | fixed_0.5 | TabM_C | contact_bright_proxy_negative | 9233.0 | NA | 0.0708 ± 0.0024 |
| external_val_TMA_cores | fixed_0.5 | TabM_C | weak_positive | 13991.0 | 0.0554 ± 0.0076 | NA |
| internal_val_WSIs | fixed_0.5 | TabM_C | contact_bright_proxy_negative | 10899.0 | NA | 0.0513 ± 0.0068 |
| internal_val_WSIs | fixed_0.5 | TabM_C | weak_positive | 3153.0 | 0.7243 ± 0.0517 | NA |
| external_val_TMA_cores | validation_selected | TabM_C | contact_bright_proxy_negative | 9233.0 | NA | 0.0421 ± 0.0094 |
| external_val_TMA_cores | validation_selected | TabM_C | weak_positive | 13991.0 | 0.0177 ± 0.0022 | NA |
| internal_val_WSIs | validation_selected | TabM_C | contact_bright_proxy_negative | 10899.0 | NA | 0.0170 ± 0.0023 |
| internal_val_WSIs | validation_selected | TabM_C | weak_positive | 3153.0 | 0.4154 ± 0.0366 | NA |

- 外部 fixed_0.5 的 C−B：接触侧亮信号阴性代理组 ΔFPR=-0.0269；弱阳性 ΔRecall=0.0054。该代理组平均假阳性率下降。还需结合逐种子、逐切片表判断稳定性。

- 外部 validation_selected 的 C−B：接触侧亮信号阴性代理组 ΔFPR=-0.0383；弱阳性 ΔRecall=-0.0028。该代理组平均假阳性率下降。同时出现弱阳性平均召回下降，应保留这一代价。

弱阳性与接触侧亮信号依据预先冻结的训练分位数定义；FPR=FP/真实阴性数。弱阳性依据 raw whole_mean <= 训练正类 q25，跨图像亮度尺度会影响落入该组的比例，不能等同统一生物学弱表达。这些是图像属性代理，不证明真实邻居污染，因此不能据 FPR 下降声称已确认减少“邻近亮细胞导致”的假阳性。

实际导出 9 个开发案例，数量与选择原因见 development_review/case_index.csv、case_availability.csv，邻居记录见 neighbor_context.csv。只使用训练/内部验证图像、种子0、固定门限，优先展示 B→C 纠正与新增错误，全部案例保留原始像素 NPZ。候选类别不等于污染机制；说明性抽样不能估算总体错误率，原人工标签未经过新增病理专家复核。

## 8. 缺失、数值分辨率与成员分歧

数值依据：[report_missingness.csv](results/report_missingness.csv)；筛选：每个split全部实例，含无标签；下表各split缺失率最高五项；指标列：`n,n_missing,missing_rate`。

| 来源 | 特征 | 全部实例 | 缺失数 | 缺失率 |
| --- | --- | --- | --- | --- |
| external_val_TMA_cores | edge_sector_07_high_fraction | 438236 | 433971 | 0.99027 |
| external_val_TMA_cores | edge_sector_01_high_fraction | 438236 | 433967 | 0.99026 |
| external_val_TMA_cores | edge_sector_05_high_fraction | 438236 | 433965 | 0.99025 |
| external_val_TMA_cores | edge_sector_08_high_fraction | 438236 | 433963 | 0.99025 |
| external_val_TMA_cores | edge_sector_02_high_fraction | 438236 | 433958 | 0.99024 |
| internal_val_WSIs | neighbor_edge_mean_diff | 667125 | 239562 | 0.3591 |
| training_WSIs | neighbor_edge_mean_diff | 712422 | 243748 | 0.34214 |
| internal_val_WSIs | neighbor_edge_mean | 667125 | 221963 | 0.33272 |
| training_WSIs | neighbor_edge_mean | 712422 | 210749 | 0.29582 |
| training_WSIs | edge_sector_04_high_fraction | 712422 | 72587 | 0.10189 |
| training_WSIs | edge_sector_10_high_fraction | 712422 | 70929 | 0.09956 |
| training_WSIs | edge_sector_05_high_fraction | 712422 | 69556 | 0.09763 |
| internal_val_WSIs | edge_sector_04_high_fraction | 667125 | 30744 | 0.04608 |
| internal_val_WSIs | edge_sector_10_high_fraction | 667125 | 29107 | 0.04363 |
| internal_val_WSIs | edge_sector_05_high_fraction | 667125 | 28974 | 0.04343 |

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
| TabM_A | 0.0006 ± 0.0000 | 0.0000 ± 0.0000 |
| TabM_B | 0.0009 ± 0.0001 | 0.0006 ± 0.0003 |
| TabM_C | 0.0014 ± 0.0000 | 0.0011 ± 0.0002 |

每个 TabM 的全部成员阳性概率随逐细胞表保留。分歧不是已校准不确定性，不据此自动拒判或删除细胞。

## 9. 资源、交付与复现

数值依据：[report_resources.csv](results/report_resources.csv)；筛选：逐canonical模型；下表为各组合三种子均值；指标列：`training_optimization_seconds,validation_seconds,inference_seconds,training_peak_phase_rss_GiB,training_rss_increase_GiB,training_peak_gpu_allocated_GiB`。

| 模型 | 组合 | 优化秒 | 验证秒 | 全量推理秒 | 阶段RSS峰GiB | RSS增量GiB | GPU分配峰GiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HGB | A | 0.417 | 0.772 | 3.122 | 2.988 | 0.032 | 0.0 |
| HGB | B | 0.667 | 0.803 | 3.163 | 3.143 | 0.109 | 0.0 |
| HGB | C | 1.727 | 0.968 | 3.716 | 3.557 | 0.362 | 0.0 |
| TabM | A | 97.499 | 152.234 | 12.03 | 3.083 | 0.13 | 0.216 |
| TabM | B | 80.998 | 128.431 | 16.044 | 3.147 | 0.138 | 0.229 |
| TabM | C | 111.413 | 188.339 | 31.472 | 3.324 | 0.129 | 0.32 |

硬件记录：`Quadro RTX 4000`。TabM 用 GPU，HGB 用同机 CPU，因此是实际配置的资源对照，不是等算力速度排名。TabM 验证时间包含每 epoch 监测，HGB 为验证推理。阶段 RSS 以 psutil 采样，含开始时驻留数据，故另报基线和增量；不以进程历史 ru_maxrss 充当当前模型峰值。GPU 指标是分配内存，不是显卡总占用。

服务器运行根目录：`/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_tabm_20261029`。权重绝对路径、SHA256 与门限见 report_model_artifacts.csv；预处理位于 models/preprocessor.joblib，完整逐细胞文件位于 predictions/，含统一键、原标签、分组、评分、两套预测和成员输出。版本见 RUN_CONFIG.json；上游 commit 见 report_provenance.csv；本项目源码以 results/code_sha256.csv 和报告 manifest 标识，不捏造未记录的本地 Git commit。

复现已有冻结流程（入口校验数据/代码/配置；新科学重复运行应使用新目录保留旧输出）：

```bash
R="/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_tabm_20261029"
PY="/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_nimbus_20261024/.venv/bin/python"
"$PY" "$R/scripts/run_pipeline.py" --run-dir "$R" --stage all
"$PY" "$R/scripts/audit_numeric_resolution.py" --run-dir "$R"
"$PY" "$R/scripts/validate_delivery.py" --run-dir "$R"
"$PY" "$R/scripts/write_tabm_report.py" --run-dir "$R"
```
数值诊断已经存在时跳过对应命令，该脚本拒绝覆盖历史诊断。

新表推理调用示例（C 只是接口示例，不表示按测试结果推荐它）：

```bash
"$PY" "$R/scripts/predict_table.py" --run-dir "$R" --model-id TabM_C_s0 --input /path/to/keyed_features.parquet --output /path/to/new_predictions.parquet
```
输入必须具有显式键及该模型有效原始特征；已存在输出拒绝覆盖。

PNG 和文字可编辑 SVG 位于 figures/zh、figures/en；源数值位于 figures/source_data；各split及TMA分组的完整 ROC/PR 和 threshold 数组位于 curves/。曲线采用 sklearn 的 >= 门限约定，正式分类表使用预定严格 >。

## 10. 限制

内部验证用于开发；外部仅对本轮冻结保持隔离，既往项目经验不能消除。WSI 数量有限，种子波动不是患者级不确定性。扩张核掩码、参照缺失和像素尺度限制膜定位与背景机制解释；弱阳性/接触侧代理仍需独立人工复核。C 使用旧原始完整特征，其中含固定方位的十二扇区值；B/C 形态还含 Orientation。此次没有新增旋转稳定性检验，不能继承 E1–E4 的旋转不变性结论。本轮只评价 PhenoBIC CD8，不扩展 marker，也不声称封存 Pan-M 的泛化表现。

官方来源：[TabM](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4)；[训练示例](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb)；[论文实验说明](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/README.md)。
