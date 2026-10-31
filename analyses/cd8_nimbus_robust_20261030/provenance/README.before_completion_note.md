# Nimbus：E1_outer_P1_P99 输入适配

本目录是新实验，保留原 `cd8_nimbus_20261024`。完成状态以 `provenance/delivery_complete.json` 为准；没有该文件时不能称全量实验已完成。模型分别命名为 `Nimbus_outerP1P99_pretrained` 与 `Nimbus_outerP1P99_finetuned`。

归一化参数来自同级 `cd8_robust_calibration_20261030/calibration/unit_bounds.csv`：每原实例内部 P10/P90，完整原 WSI 或单个 TMA 芯的全部合格实例等权外层 P1/P99，float64 映射后转 float32。没有每细胞独立 min-max，也不再除以旧训练强度系数。这是项目适配，不称为 Nimbus 官方预处理的原样复现。

## 执行

在服务器项目 `analyses/cd8_nimbus_robust_20261030` 中，使用原 Nimbus 环境的解释器，但加载新目录的固定官方源码副本：

```bash
/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_nimbus_20261024/.venv/bin/python scripts/run_pipeline.py --stage all
```

`--stage` 支持 `bootstrap`、`prepare`、`pilot`、`pretrained`、`finetune`、`predict`、`evaluate`、`report`。共享校准须先完成。`prepare` 为CPU准备，`pilot/pretrained/finetune/predict`使用GPU；实验间由外部调度串行执行。可加 `--font-path /path/to/CJK-font.ttf`。

恢复本次运行请使用 `RUN_CONFIG.json` 中记录的绝对解释器路径。独立校准覆盖率审计为 `audit_scripts/calibration_coverage.py`；其CSV和完成manifest供总体比较使用，不从模型是否返回有限数值推断校准有效性。

bootstrap 仅从原运行复制经过SHA校验的官方Python源码与V1权重到新目录；不改原虚拟环境或旧包assets。新 Trainer 的检查点只写入新 `upstream/src/nimbus_inference/assets`。维持seed0、最多10epochs、patience5、batch8、LR3e-6及原缩放、切块、TTA和实例聚合。

离线数值验证：

```bash
python scripts/validate_offline.py --crop-dir /path/to/existing/development_crops
```

该检查使用显式测试bounds，不把crop分布当作完整WSI尺度；不会训练或生成模型性能。`predict_api.predict_cell_scores()`要求调用方提供完整单位已冻结的L/U。`normalization.normalize_pixels()`保留无效支持为NaN。整单位校准无效时保留原ID并输出不可评分，不能硬填零预测。

结果表和报告由完成的真实评估生成：`results/all_metrics.csv`、`overall_metrics.csv`、`paired_vs_previous_nimbus.csv`、`paired_vs_GMM.csv`、`runtime_summary.csv`、`ANALYSIS_REPORT.md`。完整预测、模型权重、缓存留服务器，不纳入Git。Pan-M不读取。

官方源码固定commit和文件哈希在 `UPSTREAM_SOURCE_PINS.json`；运行协议由 `freeze_protocol.py` 在核验全量键和共享校准后生成。断点续用须满足代码、协议、校准表、模型和输入哈希一致；官方Trainer不支持优化器状态续训，未完成的训练不能冒充连续恢复。
