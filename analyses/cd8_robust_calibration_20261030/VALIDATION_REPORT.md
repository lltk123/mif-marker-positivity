# 共同归一化与特征模块验证记录

本记录只报告已完成的输入、校准和提取接口验证，不将样例验证称为全量模型实验。全量特征是否完成，以 `extraction_manifest.json` 的 `status` 和文件哈希为准；分类性能由 TabM/Nimbus 独立报告提供。

## 输入与方法

仅使用 PhenoBIC CD8；Pan-M 未读取。保持原图像、mask、实例 ID 和官方划分。采用原扩张核掩码（`expanded_nucleus`）；其边缘是近似掩码边缘，不是经过验证的真实膜。未获得独立组织 mask 或像素尺寸，保留相关 QC 与 px 单位。

每个原实例至少含 5 个有限像素时，以其原像素 P10/P90 参与校准；不因几何 QC 或标签缺失排除。每个单位以实例等权计算 `L=P1(cell_P10)`、`U=P99(cell_P90)`，分位数为 linear。训练按原 WSI、内部按去重原 WSI、外部按单组织芯。像素共同转换为 float64 `clip((I-L)/(U-L),0,1)`，无效支持保持 NaN，不用旧统计量线性变换代替像素重提。

`calibration/unit_bounds.csv` 全表汇总：1,817,783 个实例、48 个单位，全部实例满足本轮边界估计资格，校准失败单位为 0（列 `n_instances,n_eligible,calibration_invalid`）。按 `split` 分组求和，训练 712,422、内部 667,125、外部 438,236 个实例；对应单位数为 3、3、42。以上是全部实例数，含无标签细胞，不能当作监督训练/评价样本数。

`tests/source_index_integrity/source_counts.csv` 共 176 个源索引，其中 `n_instances==0` 为 33 个。这仅表示旧 bbox/source 索引无归属待提取实例，内部 FOV 去重后可发生；未据此认定原图或 mask 无细胞。全部源索引的四键合并与冻结集合相同，证据见同目录 `PASSED.json` 的 `all_keys_sha256`。原图/mask 内容哈希全部核验；引用源清单见 `calibration/input_manifest.csv`。

## 已通过的数值验证

- 本地受控与真实裁剪检查共 95 项通过；其中 15 个真实裁剪覆盖 5 个来源。见 `tests/local_validation/checks.csv` 的 `passed` 和 `real_crop_validation.csv` 的 `image_id`。受控检查含等权外层分位数、行序无关、重复键拒绝、clip 必须像素重提、无效支持、暗信号、极小/边缘实例、空 mask、参照缺失及增益不变性。
- 本地裁剪使用诊断 P5/P95，仅验证映射与重提，不被用于估计真实完整单位 L/U，不供训练。全部 90 项缓存与完整重算精确一致，几何保持一致；见 `real_crop_validation.csv` 对应布尔列。
- 服务器真实 pilot 使用完整单位 L/U，训练、内部、TMA1/2/3 各 20 个实例，共 100 个。见 `pilot/validation.csv` 按 `image_id` 计数。原像素 P10/P90 与既有恢复表精确一致；19 项原几何逐值一致；缓存路径与重算路径全部 90 项逐值一致。对应列为 `raw_percentiles_exact,original_morphology_exact,all_90_cache_reference_equal`，均为真。`raw_mean_exact_support` 记录实例支持精确一致及原均值在 1e-12 容差内一致。
- 另有 15 个服务器真实实例的快速数值检查，见 `tests/quick_numeric_check.csv` 的 `all90_exact`。它不替代上述完整 source-hash 审计后的正式 pilot。

## 提取实现与限制

重提原 8 项强度和 63 项空间指标，保留原 19 项几何，共 90 项。原几何通过完整三键显式关联并绑定原特征文件 SHA；可选 `morphology_cache` 只替换独立 globals 副本中的纯几何函数，原几何 QC、强度与空间代码保持不变。慢参考路径为 `--recompute-morphology`。

沿用旧 `virtual_roi`，halo 7 px、邻居 ID、有效支持和 owner-FOV 截断规则保持一致。原配置实际 epsilon 为 1e-8，本轮仍为 1e-8。`observed_mean_CD8` 表示归一化尺度；`raw_observed_mean_CD8` 是原尺度审计值，不能互换。新字典 `FEATURE_DICTIONARY.csv` 保留原单位并明确归一化后的单位。

目标图自身的无标签校准属于预定 transductive 设置。上述实现验证不证明分类性能改善，也不把空间特征代理解释为已确认的邻居污染。

## 预训练前实现修订

1. CSV 边界读取统一为 `float_precision="round_trip"`，保证共享提取与 Nimbus 使用相同 L/U。服务器旧 pandas 默认解析有 10 个边界值出现末位差，最大绝对差约 1.42e-14；见 `tests/implementation_fix_001/parser_differences.csv`，列 `difference`。原边界 CSV 字节、分位数公式、实例键未改变。
2. 正式 pilot 首个排序源索引可能没有归属实例，选例改为每组首个非空源索引。空源仍在审计中，完整实例集合不变。`tests/implementation_fix_002/COMPATIBILITY_PROOF.json` 与 `pilot_only.diff` 证明，除 pilot 函数外整个模块 AST 相同；归一化和特征模块哈希、bounds 哈希和键哈希均未改变。

全部旧协议/脚本/manifest 已保留。最终冻结协议以根目录 `PROTOCOL.json` 为准。上述修订发生在全量特征生成和新模型训练之前，未依据分类结果调整方案。
