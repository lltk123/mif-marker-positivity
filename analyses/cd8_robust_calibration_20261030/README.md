# Shared robust CD8 calibration and feature extraction

本目录实现 `E1_outer_P1_P99`：每个原始实例至少含 5 个有限像素时，读取已审计并补齐的原像素 P10/P90；在每个原始 WSI（内部验证为去重后完整 WSI）或单个 TMA core 内，按实例等权计算 P1(P10) 与 P99(P90)。全部实例与原划分保留；边界估计不读取真值。所有有效像素共同映射到浮点 `[0,1]`，不使用逐细胞缩放。

归一化后的 71 项强度/空间特征必须重新从像素计算。19 项几何特征通过完整 `dataset_id,image_id,cell_id` 显式键复用，并绑定原特征文件 SHA-256；默认 `morphology_cache` 仅在独立函数 globals 副本中替换纯 `morphometry` 调用，原几何 QC、强度与空间计算字节码不变。`--recompute-morphology` 提供慢参考路径；真实 pilot 对照全部 90 项，而非只检查几何。

沿用原 `virtual_roi` 重建：halo 7 px、全部相邻实例 ID、原 owner-FOV 截断标记、支持不足为 invalid。不是更换为整张无缝画布后放宽边缘 QC。掩码为 expanded_nucleus，边缘是近似掩码边缘，不能解释为已证实真实细胞膜。缺少组织 mask 和像素尺寸的限制保留。epsilon 为旧配置实际使用的 `1e-8`，用于归一化浮点尺度的数值稳定。

## 执行

```bash
P=/mnt/disk3/mIHC/mif_marker_positivity_20261022
R=$P/analyses/cd8_robust_calibration_20261030
$P/analyses/cd8_morphospatial_20261023/.venv/bin/python "$R/scripts/run_features.py" --run-dir "$R" --stage all --workers 4
```

可用阶段：`prepare`、`pilot`、`extract`、`merge`。准备阶段实际核验所有引用的 PhenoBIC 图像/mask、实例 bbox、旧特征和恢复后百分位文件的内容哈希。后续阶段重核哈希；不兼容缓存或部分写入产物报错，不能静默覆盖。旧项目只读；不访问 Pan-M。

本地验证：

```powershell
python scripts/validate_features.py --run-dir . --local-delivery F:/Codex/Projects/学习/mif_spatial_features/delivery
```

本地 15 个裁剪覆盖训练、内部、TMA1/2/3；裁剪诊断使用 P5/P95 只是验证映射和重提代码，并不是全单位校准参数，不用于训练。`tests/local_validation` 的通过不表示全量完成。全量开始前还会用真实完整单位边界完成 100 个实例的 `pilot`。

## 共享接口与输出

- `scripts/calibration_core.py:normalize_pixels(image,L,U,valid_mask=None)`：float64 映射；不支持像素保留 NaN；校准退化时全部强度 NaN，无替代边界。Nimbus 在此之后按网络要求转 float32。
- `calibration/unit_bounds.csv` 和 `calibration_manifest.json`：48 单位的固定 L/U、资格计数、源版本和键哈希；线性分位数的两端贡献实例见 `bound_interpolation_witnesses.csv`。
- `PROTOCOL.json`：完整原几何参数、代码/数据/边界哈希及软件版本。
- `PhenoBIC_CD8_features.parquet`：原 90 列及元数据；`observed_mean_CD8` 已归一化，`raw_observed_mean_CD8` 是审计原均值；两者不可互换。新增 raw_*、L/U 和有效性字段仅作审计元数据。
- `calibration/clipping_audit.csv`：每个唯一实例有限像素只统计一次的原动态范围、严格越界裁剪比例以及落在 0/1 的比例；不包含背景和重复 FOV 视图。
- `FEATURE_DICTIONARY.csv`：保留原定义，明确归一化强度单位；`feature_missingness.csv` 是不使用标签的缺失概览。
- `extraction_manifest.json`：只有 1,817,783 行、48 单位、唯一键、19 原几何及无无穷值验证全部通过后才写 `complete`。下游应复核其所有哈希。

训练/验证标签及类别分层缺失统计由下游训练模块在特征冻结后按完整键关联。无监督目标图校准属于预定 transductive 设置，不应称为完全不接触目标图的归纳推理。
