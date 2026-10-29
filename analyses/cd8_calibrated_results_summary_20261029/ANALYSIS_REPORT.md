# PhenoBIC CD8：既有校准实验的性能排序

**固定门限的外部主榜为 E2 > E4 > E1 > E3 > GMM。** 当前证据最支持“共同强度校准＋精简定位特征”的 E2；接入 GMM 分数没有带来一致收益。补充门限下 E4 略高于 E2，但 AP、AUC 和分来源表现并不支持 E4 全面最优。

本报告于 2026-10-29 汇总已完成结果，不重新训练或访问 Pan-M。主榜限定为 E1–E4 与原 GMM：E1 使用共同校准后的基础强度；E2 增加精简定位特征；E3 增加原 GMM 连续分数；E4 同时加入两者。E1/E2/E3/E4 分别为 8/17/9/18 项，均使用同一配置的 HGB，随机种子为 0；特征数见排名表的 `feature_count`。方法与监督条件见 [EXPERIMENT_INVENTORY.csv](EXPERIMENT_INVENTORY.csv) 的 `calibration,independence_status`。

## 外部主榜与指标分歧

| 模型 | F1 | Precision | Recall | AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2 | 0.857575 | 0.960776 | 0.774394 | 0.958357 | 0.995912 |
| E4 | 0.853606 | 0.964302 | 0.765707 | 0.949276 | 0.989751 |
| E1 | 0.849036 | 0.950508 | 0.767140 | 0.948728 | 0.996251 |
| E3 | 0.837747 | 0.958841 | 0.743809 | 0.939976 | 0.992691 |
| GMM | 0.761008 | 0.654664 | 0.908602 | 0.830201 | 0.990281 |

来源：[CALIBRATED_RANKINGS.csv](CALIBRATED_RANKINGS.csv)，筛选 `split=external_val_TMA_cores, threshold_panel=fixed_0.5`，列见表头（Precision/Recall 对应 `precision/recall`）。所有方法都覆盖相同的 437,480 个有标签细胞，覆盖率均为 1；对应 `n_total,prediction_coverage`，未通过删去难例改善成绩。

E2 的 F1 和 AP 最高，E1 的 AUC 最高；E4 的 precision 最高，GMM 的 recall 最高。E4 的 AUC 甚至略低于 GMM，表明较优的固定门限 F1 不等于全分数范围排序更好。AP 与 AUC 评价排序，F1 还依赖门限和查准／查全权衡，不能混为同一排名。

按组织芯等权，E2、E4、E1、E3、GMM 的 F1 均值依次为 0.877899、0.874258、0.869069、0.860453、0.766725，主排序不变。来源：[CALIBRATED_MACRO_METRICS.csv](CALIBRATED_MACRO_METRICS.csv)，`split=external_val_TMA_cores, threshold_mode=fixed_0.5`，列 `F1_mean`。这是芯级描述，不能当作独立患者估计。

## 收益来自哪些错误变化

E2 相对 GMM 的 F1 增加 0.096567，AP 增加 0.128156，同时 FP 减少 9,997、FN 增加 2,997。E4 相对 GMM 的 FP 减少 10,070、FN 增加 3,191。来源：[PAIRED_VS_GMM.csv](PAIRED_VS_GMM.csv)，`split=external_val_TMA_cores, threshold_panel=fixed_0.5, model=E2/E4`，列 `delta_F1,delta_AP,delta_FP,delta_FN`。两者以较低召回率换取大量误报下降，不能称全面优于 GMM，也不能把误报下降直接解释为邻居污染减少。

在同一外部主榜中，E2−E1 为 TP +162、FP −186、FN −162；E3−E1 为 TP −521、FP −179、FN +521；E4−E3 为 TP +489、FP −80、FN −489；E4−E2 为 TP −194、FP −73、FN +194。上述差值直接由主榜相应模型行的 `TP,FP,FN` 相减。定位特征在两条比较线上均有描述性收益；GMM 分数的额外作用则依赖组合，不能预设越复杂越好。

共同校准的收益还应与旧原始强度模型分开看：旧 intensity_8 的外部 F1/AP 为 0.533252/0.547629，E1 为 0.849036/0.948728。来源：[EXPERIMENT_INVENTORY.csv](EXPERIMENT_INVENTORY.csv)，筛选 `model=intensity_8/E1`，列 `F1_fixed_0_5,AP`。这支持固定图像校准流程在本数据上的价值，但不能把全部差异归因于定位特征；定位的增量应看 E2 对 E1、E4 对 E3，而非直接拿最复杂模型对原始强度模型。

共同校准使用各原始 WSI 或各组织芯自身的无标签像素，并保留图像内部细胞之间的相对强度。它没有将每个细胞单独拉伸到相同范围，也没有仅对已有均值做线性变换代替像素重新提取。当前比较将几何条件、分类器参数和训练来源固定，因此对定位特征增量的解释比跨架构比较更直接。相关实现边界来自原冻结协议及配置，具体输入组合见实验清单。

## 来源差异不能被总体排名掩盖

- **TMA1：** E2 的 F1/AP/AUC 为 0.900589/0.972558/0.998469，GMM 为 0.853435/0.939933/0.996835，支持该来源上的判别改善。
- **TMA2：** GMM recall 为 0.908403；E1–E4 分别为 0.556284、0.571720、0.562837、0.580457。新模型仍存在明显漏检，外部总体提升不能覆盖这一短板。
- **TMA3：** GMM 的 FP/FN 为 7,854/55；E2 为 320/1,013，E4 为 337/953。校准模型大幅降低误报，同时增加漏检；E4 在这里的召回权衡优于 E2，并不代表所有来源都如此。

以上均来自 [CALIBRATED_CONTEXT_METRICS.csv](CALIBRATED_CONTEXT_METRICS.csv)，筛选 `split=external_val_TMA_cores, threshold_mode=fixed_0.5, context=TMA1_CD8/TMA2_CD8/TMA3_CD8` 及相应 `model`，列为各句列明的 `F1,AP,ROC_AUC,recall,FP,FN`。

还需区分总体排序和图内排序：总体 AP/AUC 把不同图像的细胞评分放在同一数轴上，图内指标先在各单位内计算。分数跨来源的可比性与图内阴阳可分性可能不同，不能以总体 AUC 很高就断言所有组织芯都容易识别。逐芯结果应回到 [CALIBRATED_UNIT_METRICS.csv](CALIBRATED_UNIT_METRICS.csv) 的对应 `image_id,model,threshold_mode` 行查看，而不只依赖总体平均。这里没有把组织芯差异解释为患者差异，也没有用总体排名掩盖低召回来源。

## OOF 门限补充与内部成绩

| 模型 | 冻结门限 | 外部 F1 | 内部 F1 |
| --- | ---: | ---: | ---: |
| E4 | 0.30 | 0.862605 | 0.917230 |
| E2 | 0.39 | 0.861657 | 0.925111 |
| E1 | 0.42 | 0.851428 | 0.916629 |
| E3 | 0.45 | 0.841699 | 0.915947 |
| GMM | 0.50 | 0.761008 | 0.834400 |

来源：[CALIBRATED_RANKINGS.csv](CALIBRATED_RANKINGS.csv)，`threshold_panel=training_OOF_E1_E4_GMM_original`，按 `split=external_val_TMA_cores/internal_val_WSIs` 与模型取 `threshold,F1`。E1–E4 的门限只由训练 WSI 留一 OOF 确定；GMM 保持原门限，不能称为 OOF 优化后的 GMM。

补充榜中 E4 与 E2 的差距很小，且阈值不改变 AP/AUC。这里的外部 F1 是应用冻结 OOF 门限后的外部成绩；在同一 OOF 上选门限并报告的成绩另属门限开发拟合值，不能称无偏估计。补充榜不替代预先规定的主榜。

固定主门限的内部 F1 排序为 E2 0.933381、E1 0.926798、E4 0.923374、E3 0.918537、GMM 0.834400。来源：主榜 CSV，`split=internal_val_WSIs, threshold_panel=fixed_0.5`，列 `F1`。内部 WSI 等权均值的最高者却是 E3 0.924659，其次 E4 0.924518；来源：宏平均 CSV，同 split、`threshold_mode=fixed_0.5`，列 `F1_mean`。总体细胞汇总与切片等权回答不同问题，须并列解释。

## 校准、训练与评价流程

共同 L/U 与 GMM 参数只使用本单位图像和 mask，不使用人工阴阳标签。E1–E4 的监督训练与补充门限选择仅使用原三张训练 WSI；模型与门限冻结后评价内部／外部集，随后汇总排名。评价后查看指标和排序不参与已完成模型的拟合。

代码与冻结记录核查见 [LEAKAGE_AUDIT.json](LEAKAGE_AUDIT.json)，数值复算见 [AUDIT_CHECKS.csv](AUDIT_CHECKS.csv)，源表哈希见 [SOURCE_MANIFEST.csv](SOURCE_MANIFEST.csv)。本次核查未发现校准、训练或门限选择使用内部／外部标签。

分组保留官方切片划分，患者映射未提供；组织芯统计按组织芯解释。E1–E4 使用单随机种子，本报告给出描述性排名。空间特征来自扩张核掩码，其边缘为近似细胞边缘。GMM 为无监督参考，E1–E4 使用训练标签。

TabM 的训练集插补、分位数变换与嵌入分箱，以及 Nimbus 的训练来源归一化系数，与本榜共同 L/U 校准不同，因此单列在 [EXPERIMENT_INVENTORY.csv](EXPERIMENT_INVENTORY.csv)。Nimbus 的预训练来源独立性核查状态沿用原报告。

当前以 E2 作为主要参考，保留 E4 的 OOF 门限结果；TMA2 的召回率仍是后续改进重点。
