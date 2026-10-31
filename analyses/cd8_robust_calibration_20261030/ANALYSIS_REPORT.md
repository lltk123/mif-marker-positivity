# E1_outer_P1_P99 共同校准与 90 项特征：完成报告

已完成 PhenoBIC CD8 全量准备：**1,817,783 个细胞、48 个校准单位、90 项特征**，全部原实例保留。四键无重复或缺失，与冻结原集合一致；19 项原几何及 NaN 位置逐值一致；无无穷特征值。依据 `FINAL_INTEGRITY_CHECKS.csv` 的对应 `check` 行与 `passed` 列；完成清单为 `extraction_manifest.json`。

完整表留在服务器：

`/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_robust_calibration_20261030/PhenoBIC_CD8_features.parquet`

文件 SHA-256：`e3893a90b72e5cd7a91e14c612d9f220b24dbe93e9d2f67dfe42bffcc80cd36b`。本报告评价数据与特征准备；TabM/Nimbus 的分类结果由各模型目录独立提供。

## 校准与像素裁剪

每个原实例至少含 5 个有限像素，先取原像素 P10/P90；按同一单位的实例等权取 P1/P99，得到 L/U，再对像素共同执行 float64 `clip((I-L)/(U-L),0,1)`。训练/内部按原 WSI，外部按单组织芯。所有实例都参与无监督边界估计资格检查，不读取标签。不是全图像素 P1/P99，也不是逐细胞 min–max。

48 个单位均有效，校准失败为 0。统计基数为 **606,591,471 个去重、有效的实例掩码内有限像素**；严格超过 U 的比例为 **0.254771%**，严格低于 L 的比例为 0。见 `calibration/clipping_summary.csv`，筛选 `aggregation=overall`，列 `n_units,calibration_invalid_units,finite_instance_pixels,above_U_percent,below_L_percent`。

| 来源 | 有限实例像素数 | 严格超过 U | 映射到 0 | 映射到 1 |
| --- | ---: | ---: | ---: | ---: |
| 训练 WSI | 226,204,390 | 0.283732% | 9.687086% | 0.283732% |
| 内部 WSI | 238,869,744 | 0.235747% | 4.815618% | 0.235747% |
| 外部 TMA | 141,517,337 | 0.240592% | 94.854086% | 0.242581% |

表源：同一 CSV，`aggregation=split`，按 `split` 选择训练/内部/外部；列 `finite_instance_pixels,above_U_percent,at_zero_percent,at_one_percent`。逐 WSI/TMA 概览用 `aggregation=context`，逐组织芯原表在 `calibration/clipping_audit.csv`。

**外部大量零像素并非本轮新增的下界裁剪。** 所有 L 都是 0，严格低于 L 的像素数是 0（同表 `all_L_equal_zero,below_L_pixel_count`）；因此映射到 0 的这些有效原像素原本就是零。这个比例是像素比例，不能解读为细胞阴性率。

## 缺失特征：主要继承原数据与参照规则

原 8 项强度与 19 项几何的缺失模式保持原样。空间特征保留预定 NaN/QC，不以 0 或阴性标签替代。下表分母是全部实例，包括无标签实例；不是模型有标签评价分母。

| 来源 | 边缘高信号覆盖率缺失 | 新增缺失 | 边缘富集缺失 | 新增 / 恢复 |
| --- | ---: | ---: | ---: | ---: |
| 训练 WSI | 8.6280% | 7 | 3.8789% | 3 / 0 |
| 内部 WSI | 2.8954% | 10 | 2.3909% | 8 / 1 |
| TMA1 | 98.9015% | 3 | 63.2390% | 1 / 2 |
| TMA2 | 99.8703% | 0 | 50.3338% | 1 / 0 |
| TMA3 | 98.0689% | 1 | 30.3098% | 1 / 0 |

表源：`missingness_summary.csv`。前两行筛选 `aggregation=split` 及对应 `split`；后三行筛选 `aggregation=context, split=external_val_TMA_cores, context=TMA1_CD8/TMA2_CD8/TMA3_CD8`。两项特征分别为 `high_signal_edge_coverage` 和 `edge_signal_enrichment`；数值列为 `new_missing_percent,newly_missing,recovered`。旧值在 `old_missing_percent`，逐来源完整 90 项对照在 `missingness_vs_raw.csv`，重点列在 `feature_missingness_highlights.csv`。

外部总体的覆盖率缺失从 **98.997572% 变为 98.998485%**，仅新增 4 个实例；富集缺失从 **48.516096% 变为 48.516325%**，新增 3 个、恢复 2 个。见 `missingness_summary.csv`，筛选 `aggregation=split, split=external_val_TMA_cores` 与两项 feature，列 `old_missing_percent,new_missing_percent,newly_missing,recovered`。因此高缺失主要是既有问题，不能将其大幅归因于本轮归一化。

对 NaN 按冻结公式的适用条件作互斥归因，顺序为校准、几何/像素支持、edge/inner 支持、参照支持，再检查相应信号条件。`missingness_causes.csv` 筛选 `version=normalized,aggregation=split,split=external_val_TMA_cores`：

- 覆盖率：406,718 个缺失归到参照 MAD 恰为 0；27,124 个归到参照像素不足；另有 5 个几何/像素支持问题。筛选 `feature=high_signal_edge_coverage`，使用 `cause,cause_count`。
- 富集：185,478 个在前述支持条件满足后，校正总信号未超过 epsilon×面积；27,123 个参照不足；另有 5 个几何/像素支持问题与 10 个 inner 支持不足。筛选 `feature=edge_signal_enrichment`，同列。

覆盖率要求 MAD>epsilon；富集本身不要求 MAD>epsilon，而要求足够的校正总信号。不能用一个 `no_signal` 标记解释所有空间特征缺失，也不能把 NaN 当作“没有边缘表达”。原因表解释的是计算适用条件，不是已证实的生物学污染。重叠 QC 标记与原表对照另见 `qc_vs_raw.csv`。

## 验证、运行与限制

受控/本地验证 95 项通过，服务器真实 pilot 为 100 个实例，覆盖训练、内部、TMA1/2/3；缓存与完整重算的 90 项逐值一致。细节与实现修订记录见 `VALIDATION_REPORT.md`、`tests/local_validation/checks.csv` 及 `pilot/validation.csv`。最终全量检查见 `FINAL_INTEGRITY_CHECKS.csv`；本轮没有修改原图、mask、实例 ID、旧结果或特征公式。

`RUN_RESOURCES.csv`：`stage=extract` 用时 **44.8059 分钟**，`stage=merge` 用时 **6.0068 分钟**，均包含各阶段输入哈希核验；提取为 4 个进程。对应列为 `wall_minutes,parallel_feature_workers,includes_input_hash_audit`。提取的单进程最高观察 HWM 为 **20.5682 GiB**；进程树 RSS 和峰值 **79.7066 GiB** 重复计算共享 mmap 页，不能解释成独占物理内存消耗。见同表 `max_observed_process_hwm_GiB,sampled_peak_sum_rss_GiB,rss_sum_note`。

实例 mask 仍是 expanded_nucleus，边缘只能称近似掩码边缘。没有独立组织 mask，局部环带仍是周细胞局部参照；没有确认像素尺寸，保留 px。共同校准使用目标图自身无标签观测，属于预定 transductive 设置。高缺失、空间特征收益和分类错误变化需结合后续冻结模型结果解释，本报告不预设归一化或任何模型必然更优。Pan-M 始终封存。
