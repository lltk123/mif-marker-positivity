# PhenoBIC 逐细胞形态与 CD8 空间特征报告

## 结论与表格索引

本报告采用 Markdown 正文 + CSV 数据表。全部编号表位于同目录的 `report_tables/`。CSV 使用单行表头；比例字段为 0–1。每条结论给出文件、行筛选条件和指标列；Markdown 中的展示数字可以四舍五入，原始 CSV 不人为截断精度。

[全部表格目录](../../../cd8_morphospatial_20261023/report_tables/TABLE_INDEX.csv) · [结论—表格对应表](../../../cd8_morphospatial_20261023/report_tables/CONCLUSION_INDEX.csv)

**C01 — 完整模型内部验证优于 GMM，外部 TMA 低于 GMM，不能全面替代。** 内部 0.8344→0.9340；外部 0.7610→0.5453。

表格：[T01_model_comparison](../../../cd8_morphospatial_20261023/report_tables/T01_model_comparison.csv)、[T03_model_differences](../../../cd8_morphospatial_20261023/report_tables/T03_model_differences.csv)。筛选：`split∈{internal_val_WSIs,external_val_TMA_cores}; model∈{GMM,morphospatial_90}; T03.reference_model=GMM; metric=F1`。指标列：`F1; difference`。解释边界：监督分类器与逐图像无监督 GMM 的处理条件不同。

**C02 — 90 项完整特征相对 8 项强度特征有额外收益，但幅度较小。** 内部 F1 +0.0085；外部 +0.0121。

表格：[T03_model_differences](../../../cd8_morphospatial_20261023/report_tables/T03_model_differences.csv)。筛选：`reference_model=intensity_8; metric=F1`。指标列：`model_value; reference_value; difference`。解释边界：无法从整体对照单独归因于某个形态或空间指标。

**C03 — 外部失败集中在 TMA1/TMA2，不能用 TMA3 的改善概括全部 TMA。** TMA1：TP=0、FN=7276；TMA2 recall=0.1615；TMA3 F1=0.9000。

表格：[T02_TMA_context_metrics](../../../cd8_morphospatial_20261023/report_tables/T02_TMA_context_metrics.csv)。筛选：`model=morphospatial_90; context∈{TMA1_CD8,TMA2_CD8,TMA3_CD8}`。指标列：`F1; recall; TP; FP; FN`。解释边界：TMA1 无阳性预测时 precision 按约定记 0。

**C04 — 不同组原始强度分布明显不同，提示固定分类器迁移风险。** 训练 WSI p95≈4.06/6.03/6.15；TMA1/2/3≈0.079/0.293/15.95。

表格：[T06_feature_distributions](../../../cd8_morphospatial_20261023/report_tables/T06_feature_distributions.csv)。筛选：`feature=whole_mean; statistic=p95; split∈{training_WSIs,external_val_TMA_cores}`。指标列：`value`。解释边界：统计包含所有有效特征细胞；只能描述分布差异，不能确认唯一失败原因。

**C05 — 留一训练 WSI 的样本外结果支持有限的完整特征收益。** OOF F1：0.8969→0.9022。

表格：[T04_training_oof_summary](../../../cd8_morphospatial_20261023/report_tables/T04_training_oof_summary.csv)、[T15_training_oof_per_WSI](../../../cd8_morphospatial_20261023/report_tables/T15_training_oof_per_WSI.csv)。筛选：`model∈{intensity_8,morphospatial_90}`。指标列：`ROC_AUC; AP; F1`。解释边界：仅 3 折、未用于调参；不等同于外部 TMA 泛化。

**C06 — 局部参照和定位特征存在可解释的缺失，全部细胞保留。** 参照不足 69662（3.83%）；MAD=0 为 444878（24.47%）；形态/整体强度各 99 个 NaN。

表格：[T07_feature_missingness](../../../cd8_morphospatial_20261023/report_tables/T07_feature_missingness.csv)、[T08_QC_counts](../../../cd8_morphospatial_20261023/report_tables/T08_QC_counts.csv)。筛选：`T08.flag∈{qc_reference_insufficient,qc_reference_mad_zero,qc_truncated,qc_disconnected}; T07.feature∈{Orientation,whole_mean,edge_signal_enrichment,high_signal_edge_coverage}`。指标列：`missing; count; fraction`。解释边界：QC 可重叠；no_signal 不是细胞阴性标签。

**C07 — 受控测试、真实试验和全量 ID/强度核验通过。** 均匀差值=0、富集=1；环形富集≈2.98；单侧偏移≈0.81；全量 1817783 行映射一致。

表格：[T09_controlled_features](../../../cd8_morphospatial_20261023/report_tables/T09_controlled_features.csv)、[T10_validation_audit](../../../cd8_morphospatial_20261023/report_tables/T10_validation_audit.csv)。筛选：`T09.image_id∈{uniform,ring,one_sided,bright_neighbor_without_spill,bright_neighbor_with_spill}; T10.group∈{HistomicsTK19,API,pilot,model_and_mapping,final_audit}`。指标列：`edge_inner_mean_diff; edge_signal_enrichment; sector_signal_cv; signal_centroid_offset_normalized; check; value`。解释边界：三个规则形状对照通过不代表所有退化形状与 HistomicsTK 完全等价。

**C08 — 掩码是核扩张近似，当前输出不能解释为真实膜定位。** expanded_nucleus；无组织有效 mask 和已确认像素尺寸。

表格：[T11_configuration](../../../cd8_morphospatial_20261023/report_tables/T11_configuration.csv)、[T13_feature_dictionary](../../../cd8_morphospatial_20261023/report_tables/T13_feature_dictionary.csv)。筛选：`T11.parameter∈{RUN_CONFIG.config.mask_type,RUN_CONFIG.mask_type_evidence,RUN_CONFIG.tissue_mask_provided,RUN_CONFIG.pixel_spacing_confirmed}`。指标列：`T11.value; T13.name; T13.definition; T13.unit; T13.applicable_when; T13.nan_reason`。解释边界：这是数据说明约束，非模型性能推断；原论文链接见参数表和正文。

## 完整报告

2026-10-23 · PhenoBIC 开发分析 · Pan-M Gold 保持封存



**已完成：**1,817,783 个细胞、163 个字段，其中包含指定的 19 项形态指标和总计 90 项数值模型特征。保留全部原始实例，按 `dataset_id + image_id + cell_id` 唯一关联。两个固定参数分类器已训练，未访问 Pan-M Gold。

数据覆盖原始 3 张训练 WSI、131 个内部验证 FOV（归属 3 张 WSI）和 42 个外部 CD8 TMA 芯，共 176 个源图像记录。提取前先验证 5 个真实图像中的 1,000 个细胞，再完成全量提取。全量提取耗时约 15.8 分钟（8 个工作进程）。

**主要结果：**90 项完整特征模型的内部验证 F1 从 GMM 的 0.8344 升至 0.9340；外部 TMA F1 则从 0.7610 降至 0.5453。内部收益成立，但固定模型跨 TMA 的适用性不足，不能全面替代前次 GMM。

## 模型性能与评估边界

监督训练只使用原始训练集 101,057 个有标签细胞（3 张 WSI），模型为 `HistGradientBoostingClassifier`。固定 150 次迭代、学习率 0.06、15 个叶节点、叶节点最少 80 个样本、L2=2、random_state=0，关闭早停；阳性规则为概率 >0.5。没有超参数搜索、验证集阈值优化或 QC 行过滤。缺失值由模型原生处理。

8 项模型只输入整体强度统计；90 项模型输入完整特征。特征计算不读取标签/GMM；提取完成后才关联标签和预测。图像、患者、批次、坐标、ID、QC 标记与 GMM 分数均不进入分类器。内部和外部原始验证划分保持不变。

| 评估集 | 方法 | 细胞数 | AUC | AP | F1 | 查准率 | 查全率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 内部验证 WSI | GMM（前次基线） | 539132 | 0.9974 | 0.9481 | 0.8344 | 0.7513 | 0.9382 |
| 内部验证 WSI | 8 项整体强度 | 539132 | 0.9987 | 0.9783 | 0.9255 | 0.9264 | 0.9245 |
| 内部验证 WSI | 90 项完整特征 | 539132 | 0.9990 | 0.9827 | 0.9340 | 0.9284 | 0.9397 |
| 外部验证 TMA | GMM（前次基线） | 437480 | 0.9903 | 0.8302 | 0.7610 | 0.6547 | 0.9086 |
| 外部验证 TMA | 8 项整体强度 | 437480 | 0.8137 | 0.5476 | 0.5333 | 0.8324 | 0.3923 |
| 外部验证 TMA | 90 项完整特征 | 437480 | 0.8145 | 0.6134 | 0.5453 | 0.8734 | 0.3964 |

**该表的完整 CSV：** [T01_model_comparison.csv](../../../cd8_morphospatial_20261023/report_tables/T01_model_comparison.csv)。正文数值四舍五入，CSV 保留计算精度。

完整模型相对 8 项强度模型：内部 F1 增加 **0.0085**，外部增加 **0.0121**。依据：[T03_model_differences.csv](../../../cd8_morphospatial_20261023/report_tables/T03_model_differences.csv)，筛选 `reference_model=intensity_8; metric=F1`。这是整体附加特征的比较，不能据此将收益单独归因于形态或某一空间指标。原 GMM 采用逐图像无监督校准；分类器使用训练标签，因此两者监督条件不同。

可选图件：[内部和外部验证的 ROC 与 PR 曲线](../../../cd8_morphospatial_20261023/figures/model_curves_zh.png)。结论依据以对应 CSV 为准。

图例数字为 AUC/AP。绘图对曲线抽稀并保留较大跳变；所有指标使用完整有标签细胞计算，完整曲线数组同时保存。

### 外部 TMA 的失败与成功必须分别看

| 组 | 方法 | AUC | AP | F1 | 查准率 | 查全率 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TMA1_CD8 | GMM（前次基线） | 0.9968 | 0.9399 | 0.8534 | 0.8975 | 0.8135 | 5919 | 676 | 1357 |
| TMA1_CD8 | 90 项完整特征 | 0.5206 | 0.2817 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 7276 |
| TMA2_CD8 | GMM（前次基线） | 0.9951 | 0.8935 | 0.8166 | 0.7416 | 0.9084 | 6238 | 2173 | 629 |
| TMA2_CD8 | 90 项完整特征 | 0.9338 | 0.8640 | 0.2778 | 0.9937 | 0.1615 | 1109 | 7 | 5758 |
| TMA3_CD8 | GMM（前次基线） | 0.9946 | 0.9232 | 0.6728 | 0.5087 | 0.9933 | 8133 | 7854 | 55 |
| TMA3_CD8 | 90 项完整特征 | 0.9976 | 0.9626 | 0.9000 | 0.8585 | 0.9457 | 7743 | 1276 | 445 |

**该表的完整 CSV：** [T02_TMA_context_metrics.csv](../../../cd8_morphospatial_20261023/report_tables/T02_TMA_context_metrics.csv)。正文数值四舍五入，CSV 保留计算精度。

完整模型在 TMA1 没有预测阳性：7,276 个阳性全部漏检（查准率因无阳性预测而按约定记 0）。TMA2 查全率为 16.15%；TMA3 F1 为 0.9000。不能用 TMA3 的改善掩盖 TMA1/TMA2 的失败。

原始强度分布存在明显差异（依据：[T06_feature_distributions.csv](../../../cd8_morphospatial_20261023/report_tables/T06_feature_distributions.csv)，筛选 `feature=whole_mean; statistic=p95`）：全体有效细胞的 whole_mean 第 95 百分位在训练 WSI1/2/3 分别约为 4.06/6.03/6.15，而 TMA1/2/3 约为 0.079/0.293/15.95。WSI、TMA1/2 为 float32，TMA3 为 uint8。本次保留原始强度，没有按细胞归一化；分布差异与固定分类器的迁移困难相符，但没有通过干预实验确定唯一原因。逐细胞原始均值与旧 GMM 数据全部一致，未发现读图或 ID 对应错误。

可选图件：[内部与三个 TMA 组的 F1、查准率、查全率](../../../cd8_morphospatial_20261023/figures/context_comparison_zh.png)。结论依据以对应 CSV 为准。

### 按 WSI/组织芯分组的补充统计

训练集 3 折留一 WSI 的样本外预测（依据：[T04_training_oof_summary.csv](../../../cd8_morphospatial_20261023/report_tables/T04_training_oof_summary.csv)；逐折见 [T15_training_oof_per_WSI.csv](../../../cd8_morphospatial_20261023/report_tables/T15_training_oof_per_WSI.csv)）：8 项模型 AUC/AP/F1 = 0.9914/0.9505/0.8969；完整模型为 0.9924/0.9582/0.9022。该检查不用于调参。训练集拟合内成绩另存，不能当成泛化成绩。

| 评估集 | 方法 | 图像单位数 | 图像等权 F1 均值 | 图像间 SD | 最低 F1 | 最高 F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 内部验证 WSI | GMM（前次基线） | 3 | 0.8170 | 0.0644 | 0.7555 | 0.8839 |
| 内部验证 WSI | 8 项整体强度 | 3 | 0.9050 | 0.0449 | 0.8537 | 0.9370 |
| 内部验证 WSI | 90 项完整特征 | 3 | 0.9184 | 0.0304 | 0.8869 | 0.9475 |
| 外部验证 TMA | GMM（前次基线） | 42 | 0.7667 | 0.1029 | 0.5055 | 0.8827 |
| 外部验证 TMA | 8 项整体强度 | 42 | 0.4854 | 0.3999 | 0.0000 | 0.9551 |
| 外部验证 TMA | 90 项完整特征 | 42 | 0.5027 | 0.4039 | 0.0000 | 0.9552 |

**该表的完整 CSV：** [T05_image_equal_weight_metrics.csv](../../../cd8_morphospatial_20261023/report_tables/T05_image_equal_weight_metrics.csv)。正文数值四舍五入，CSV 保留计算精度。

上表为图像单位等权统计，SD 是图像间差异，不是置信区间。细胞存在同图像内相关性；仅有 3 张内部 WSI，不能将数十万个细胞视为同等数量的独立样本，也不额外假定不同组织芯来自独立患者。人工标签沿用发布版本；原论文描述训练/内部标签包含算法辅助人工复核步骤。这里所有评估均属 PhenoBIC 开发分析，未触及封存的 Pan-M Gold。

## 特征定义与 mask 语义

依据 [PhenoBIC 原论文的 Cell segmentation 与 Figure 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC13308180/)，此批自建 WSI/TMA 实例来自核分割后扩张的区域，设置为 `expanded_nucleus`。边缘是近似细胞边界，不能认定为真实膜；inner 是掩码内部，不等同于细胞核。原始核掩码和已确认的像素尺寸未提供，因此默认只输出像素尺度。已有可靠实例 mask 即可运行，不依赖 DAPI。

- **19 项形态：**Orientation、Area、ConvexHullArea、MajorAxisLength、MinorAxisLength、Perimeter、Circularity、Eccentricity、EquivalentDiameter、Extent、MinorMajorAxisRatio、Solidity、HuMoments1–7。

- **8 项整体强度：**mean、median、std、p10、p90、p95、IQR、integrated_intensity。

- **8 项边缘/内部：**各自 mean/median/p90、均值差、归一化差。

- **7 项局部参照/校正：**中位数、MAD、三种均值减参照、正信号的 log2 边缘/内部比、面积校正边缘富集。

- **36 项扇区与 5 项汇总：**12 个扇区各自原始均值、校正信号均值、高信号覆盖率；原始/校正信号 CV 和最大值减中位数，以及整体边缘高信号覆盖率。

- **2 项偏移与 5 项邻居特征：**信号质心像素/等效半径归一化偏移；邻居侧/非邻居侧均值、差值及各自边缘面积比例。

其余 73 个字段保存键、元数据、区域像素数、实际参数、物理尺度可选输出与 QC。完整公式、单位、NaN 条件及模型输入标记见 [特征字典 CSV](../../../cd8_morphospatial_20261023/report_tables/T13_feature_dictionary.csv)。

```text
w = max(1, 0.2 × max(EDT(M)))
edge = M ∩ (d ≤ w); inner = M ∩ (d > w)
reference = (2 < distance_to_M ≤ 5) ∩ background_label_0 ∩ valid_support
S = max(CD8 − reference_median, 0)
enrichment = [sum(S_edge)/sum(S_whole)] / [area(edge)/area(whole)]
T = reference_median + 3 × 1.4826 × MAD; epsilon = 1e−8（原始强度单位）
邻居距离 ≤2 px；whole/edge/inner 最少 5 像素；参照最少 20 像素；
扇区最少 2 像素，扇区汇总至少 6/12 个有效扇区。
```

距离变换对实例补零，不用裁剪框代替边界；读取局部 ROI 时保留所有相邻实例，并包含足够 halo。内部 FOV 的重叠区域按已验证的非零实例 ID 合并上下文，不重复计数。参照排除所有已分割实例和无效像素；如果提供组织有效 mask，则进一步排除组织外区域。本次没有组织 mask，所有细胞明确标记 `reference_tissue_unknown`，不从 CD8 猜测组织，也不将该区域认定为真实背景。

k=3 仅为高信号描述的工程起点，不是细胞阳性门限；MAD=0 或参照不足时覆盖率为 NaN。环形不完整、信号偏移或邻居接触侧较亮均不直接决定阴阳性，也不能单独证明污染。原始强度特征与参照扣除结果并存。

HistomicsTK 1.4.0 要求 Python≥3.9，与服务器现有 Python 3.8 不兼容，因此采用 scikit-image 0.21.0。使用该版本 HistomicsTK 的实际函数源码，在圆、椭圆、矩形上比较指定 19 项，最大绝对差均为 0；这不代表所有退化情形完全等价。退化、截断或过小区域按本任务规则输出 NaN。Orientation 按实际 regionprops 实现记录为相对行轴的弧度；离散周长可能使 Circularity>1。未纳入 FractalDimension 或 WeightedHuMoments。

## 验证与缺失情况

9 个受控图像情形全部通过；补充 8 项 API 检查涵盖 ROI 裁剪不变性、NaN 像素、组织排除、物理单位、断裂实例、错误输入、标签元数据拒绝、空表 Parquet 固定结构。单图 CLI 已实际验证 uint16 强度 1234、实例 ID=73、163 列 Parquet 和 CSV 输出。

均匀信号的 edge−inner=0、富集=1；环形信号差值=10、富集≈2.98；单侧亮点扇区 CV≈2.25、归一化偏移≈0.81。邻居独自变亮不会被计入目标细胞；在接触侧加入信号后接触差值升至 12。全暗、极小、无参照和截断情况稳定返回 NaN/QC。

可选图件：[九个受控图像验证](../../../cd8_morphospatial_20261023/figures/controlled_validation_zh.png)。结论依据以对应 CSV 为准。

真实试验 1,000 个细胞的面积与原始均值均与旧逻辑一致；保存 15 个固定 ID 示例，人工视觉检查其中 5 个代表案例。全量 1,817,783 个细胞再次核验键、面积和原始均值，无重复键、无无穷值。全空 mask 返回固定 163 列空表。

可选图件：[真实图像、实例边界、edge、inner、局部参照和扇区值](../../../cd8_morphospatial_20261023/figures/real_examples_zh.png)。结论依据以对应 CSV 为准。

固定索引选取示例，不依据标签或模型表现挑选。灰度拉伸仅用于显示，计算始终使用原始像素；示例的比例尺单位为像素。

| 特征 | NaN 细胞数 | 缺失比例 |
| --- | --- | --- |
| whole_mean | 99 | 0.01% |
| inner_mean | 126 | 0.01% |
| edge_inner_normalized_diff | 196832 | 10.83% |
| local_reference_median | 69662 | 3.83% |
| edge_signal_enrichment | 256189 | 14.09% |
| sector_signal_cv | 329132 | 18.11% |
| high_signal_edge_coverage | 514610 | 28.31% |
| signal_centroid_offset_normalized | 256170 | 14.09% |
| neighbor_edge_mean_diff | 567185 | 31.20% |

**该表的完整 CSV：** [T07_feature_missingness.csv](../../../cd8_morphospatial_20261023/report_tables/T07_feature_missingness.csv)。正文数值四舍五入，CSV 保留计算精度。

19 项形态及整体强度各有 99 个 NaN（82 个边界截断、17 个断裂实例）；inner 过小涉及 28 个细胞，部分与上述情况重叠。参照不足 69,662 个（3.83%），参照 MAD=0 有 444,878 个（24.47%）。无信号导致某些比值/定位指标缺失，属于数值适用条件，不是将其自动标为阴性。缺少像素尺寸时物理尺度列全部为 NaN。未删除任何细胞或改写原标签。

**完整 QC 计数（可重叠，不能相加作为细胞总数）**

| flag | count | 比例 |
| --- | --- | --- |
| qc_truncated | 82 | 0.005% |
| qc_small_inner | 28 | 0.002% |
| qc_reference_insufficient | 69662 | 3.832% |
| qc_reference_mad_zero | 444878 | 24.474% |
| qc_no_signal | 274832 | 15.119% |
| qc_sector_insufficient | 1669 | 0.092% |
| qc_sector_partial | 136013 | 7.482% |
| qc_no_neighbors | 213465 | 11.743% |
| qc_no_contact_edge | 486414 | 26.759% |
| qc_no_noncontact_edge | 80672 | 4.438% |
| qc_reference_tissue_unknown | 1817783 | 100.000% |
| qc_context_clipped | 112 | 0.006% |
| qc_disconnected | 17 | 0.001% |

**该表的完整 CSV：** [T08_QC_counts.csv](../../../cd8_morphospatial_20261023/report_tables/T08_QC_counts.csv)。正文数值四舍五入，CSV 保留计算精度。

[全部 90 项缺失率](../../../cd8_morphospatial_20261023/report_tables/T07_feature_missingness.csv) · [全部 QC 计数](../../../cd8_morphospatial_20261023/report_tables/T08_QC_counts.csv) · [按截断 QC 分层性能](../../../cd8_morphospatial_20261023/report_tables/T16_metrics_by_truncation_QC.csv)

## 保存位置与复用

```text
/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_morphospatial_20261023/
  PhenoBIC_CD8_features.parquet                 1,096,366,557 bytes
  PhenoBIC_CD8_model_predictions.parquet           25,733,928 bytes
  models/intensity_8.joblib
  models/morphospatial_90.joblib
  scripts/mif_features.py
  scripts/extract_single.py
  scripts/batch_features.py
  scripts/train_compare_v2.py
  RUN_CONFIG.json / TRAINING_PROTOCOL.json
  FEATURE_DICTIONARY.csv / FEATURE_DICTIONARY.json
```

原数据路径：`/mnt/disk3/mIHC/mif_marker_positivity_20261022/datasets/PhenoBIC-Data`。版本 `aeffd288d922ee5df4cbb414f3c1f77507ba0001`。原始图像、位深、实例 ID、标签及数据划分均未修改。大 Parquet 留在服务器；本地审阅包包含报告、脚本、字典、模型、指标、验证样例与中英文可编辑图。

[运行说明与 API 示例](USAGE.md) · [核心特征模块](scripts/mif_features.py) · [通用单图入口](../../../cd8_morphospatial_20261023/scripts/extract_single.py) · [配置及软件版本](../../../cd8_morphospatial_20261023/report_tables/T11_configuration.csv) · [训练协议](../../../cd8_morphospatial_20261023/report_tables/T11_configuration.csv) · [完整模型指标](../../../cd8_morphospatial_20261023/report_tables/T01_model_comparison.csv) · [逐图像指标](../../../cd8_morphospatial_20261023/report_tables/T14_metrics_per_image.csv) · [训练 WSI 样本外指标](../../../cd8_morphospatial_20261023/report_tables/T04_training_oof_summary.csv) · [产物大小与 SHA256](../../../cd8_morphospatial_20261023/report_tables/T12_artifact_integrity.csv)

Python 3.8.20；numpy 1.24.4；pandas 1.1.4；scipy 1.10.1；scikit-image 0.21.0；scikit-learn 1.1.3；tifffile 2023.7.10；pyarrow 15.0.2。独立 venv 未修改旧 GMM 环境。CSV 以 round_trip 读取后旧 GMM 指标完全复现；只重新核算基线，没有重新拟合或调整分类器。最初浮点解析版本保留以供审计。

### 图表与来源

每幅图均有中文与英文 PNG/PDF/SVG：[性能曲线 EN PDF](../../../cd8_morphospatial_20261023/figures/model_curves_en.pdf)、[性能曲线 ZH SVG](../../../cd8_morphospatial_20261023/figures/model_curves_zh.svg)、[受控验证 EN PDF](../../../cd8_morphospatial_20261023/figures/controlled_validation_en.pdf)、[真实案例 EN PDF](../../../cd8_morphospatial_20261023/figures/real_examples_en.pdf)。图表源数值位于 results/ 和 tests/；示例原图与区域数组位于 examples/。

[PhenoBIC 官方数据](https://huggingface.co/datasets/mittal-research-lab/PhenoBIC-Data) · [PhenoBIC 论文与分割说明](https://pmc.ncbi.nlm.nih.gov/articles/PMC13308180/) · [SCIMAP rescale](https://scimap.xyz/Functions/pp/rescale/) · [HistomicsTK features](https://digitalslidearchive.github.io/HistomicsTK/histomicstk.features.html) · [scikit-image 0.21 measure](https://scikit-image.org/docs/0.21.x/api/skimage.measure.html)

## 全部 CSV 文件位置

| 表号 | 文件 | 内容 | 行键 |
| --- | --- | --- | --- |
| T01 | [T01_model_comparison.csv](../../../cd8_morphospatial_20261023/report_tables/T01_model_comparison.csv) | 原始划分的全部模型指标；training_resubstitution=True 为训练拟合内结果 | split + model |
| T02 | [T02_TMA_context_metrics.csv](../../../cd8_morphospatial_20261023/report_tables/T02_TMA_context_metrics.csv) | TMA1/2/3 分组模型性能和混淆计数 | context + model |
| T03 | [T03_model_differences.csv](../../../cd8_morphospatial_20261023/report_tables/T03_model_differences.csv) | 完整模型相对基线的绝对差；difference=model_value-reference_value | split + reference_model + metric |
| T04 | [T04_training_oof_summary.csv](../../../cd8_morphospatial_20261023/report_tables/T04_training_oof_summary.csv) | 原训练集留一 WSI 样本外指标；不用于调参 | model |
| T05 | [T05_image_equal_weight_metrics.csv](../../../cd8_morphospatial_20261023/report_tables/T05_image_equal_weight_metrics.csv) | 图像等权汇总；std 为图像间标准差，非置信区间 | model + split + metric |
| T06 | [T06_feature_distributions.csv](../../../cd8_morphospatial_20261023/report_tables/T06_feature_distributions.csv) | 按原始组统计全部有效特征细胞的强度与空间分布，含未标注细胞 | split + context + feature + statistic |
| T07 | [T07_feature_missingness.csv](../../../cd8_morphospatial_20261023/report_tables/T07_feature_missingness.csv) | 90 项模型特征的 NaN 数与比例；fraction 为 0–1 | feature |
| T08 | [T08_QC_counts.csv](../../../cd8_morphospatial_20261023/report_tables/T08_QC_counts.csv) | QC 可重叠；不能相加为总细胞数；fraction 为 0–1 | flag |
| T09 | [T09_controlled_features.csv](../../../cd8_morphospatial_20261023/report_tables/T09_controlled_features.csv) | 9 个受控图像的原始数值、QC 和区域计数 | image_id + cell_id |
| T10 | [T10_validation_audit.csv](../../../cd8_morphospatial_20261023/report_tables/T10_validation_audit.csv) | 受控测试、实现对照、API、真实试验及全量核验结果 | group + check |
| T11 | [T11_configuration.csv](../../../cd8_morphospatial_20261023/report_tables/T11_configuration.csv) | 特征参数、版本、模型参数、训练限制及来源 | parameter |
| T12 | [T12_artifact_integrity.csv](../../../cd8_morphospatial_20261023/report_tables/T12_artifact_integrity.csv) | 服务器完整特征表、预测表和模型的大小与 SHA256 | file |
| T13 | [T13_feature_dictionary.csv](../../../cd8_morphospatial_20261023/report_tables/T13_feature_dictionary.csv) | 全部 163 字段的定义、单位、NaN 条件和模型输入角色 | name |
| T14 | [T14_metrics_per_image.csv](../../../cd8_morphospatial_20261023/report_tables/T14_metrics_per_image.csv) | 逐 WSI/组织芯完整模型性能 | model + image_id |
| T15 | [T15_training_oof_per_WSI.csv](../../../cd8_morphospatial_20261023/report_tables/T15_training_oof_per_WSI.csv) | 留一训练 WSI 的逐折性能 | model + heldout_image_id |
| T16 | [T16_metrics_by_truncation_QC.csv](../../../cd8_morphospatial_20261023/report_tables/T16_metrics_by_truncation_QC.csv) | 按边界截断 QC 分层性能 | model + split + qc_truncated |

统计、图表和模型结果均沿用本次已经核验的运行；本次只重排报告与表格，不重新训练或更改结果。原 HTML 保留归档，后续以本 Markdown 和对应 CSV 为主要交付。
