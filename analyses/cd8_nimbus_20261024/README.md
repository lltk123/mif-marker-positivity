# Nimbus PhenoBIC CD8 基线

服务器目录：/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_nimbus_20261024

此目录保存对官方 Nimbus 的数据适配。原数据与旧模型结果保留，Pan-M 不读取。实际完成状态以 provenance/numeric_pipeline_complete.json 和最终交付核验为准。

## 执行

```bash
cd /mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_nimbus_20261024
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
.venv/bin/python scripts/run_pipeline.py
```

独立入口：

```bash
.venv/bin/python scripts/audit_sources.py
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/validate_contract.py
.venv/bin/python scripts/pilot.py
.venv/bin/python scripts/inference.py --model Nimbus_pretrained
.venv/bin/python scripts/finetune.py --prepare-only
.venv/bin/python scripts/finetune.py
.venv/bin/python scripts/inference.py --model Nimbus_finetuned
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/plot_results.py
.venv/bin/python scripts/review_errors.py
.venv/bin/python scripts/audit_predictions.py
.venv/bin/python scripts/finalize_provenance.py
.venv/bin/python scripts/write_report.py
```

推理可加 `--image-id 原始image_id` 或 `--split training_WSIs`。可用键来自 provenance/cell_keys.parquet。

完成缓存校验输入、协议、归一化、代码和权重哈希后复用。完成微调不会覆盖；重新训练需新目录。官方 Trainer 不提供完整优化器恢复，本入口不会把仅加载权重称为同一优化过程的断点恢复。

## 接口

- predict_api.predict_cell_scores()：单图数组接口，显式提供冻结归一化系数、模型与原始键。
- common.load_cells()：四键清单，禁止按行号关联。
- prepare_data.raw_unit(image_id)：原位深 CD8、uint32 mask、有效支持；内部 FOV 按全局坐标无重复合并。
- prepare_data.prepare_unit(image_id,factor)：共享 marker 归一化、全图缩放。
- inference.StreamingNimbus：继承官方 Nimbus，保持相同全局网格、padding 和模型。
- inference.official_tta_stream：保留固定官方版本实际 TTA。
- inference.aggregate：按完整原实例平均最近邻还原后的输出。
- finetune.PhenoTileDataset：安全 ID 和稀疏标签，供官方 Trainer 使用。
- evaluate.metrics：完整集合、覆盖率、无阳性单位 NA 约定。

网络仅输入 CD8 图像与实例内部二值图。原形态、空间统计、GMM 和身份编号不进入网络。旧特征仅帮助选择开发集审阅候选。

## 环境与结果

依赖版本见 provenance/requirements.lock.txt；源码和权重固定信息见 NIMBUS_PROTOCOL.json。固定本地 V1.pt，不调用 latest 或下载作者示例数据。安装脚本与加速来源保存在 scripts/install_environment.py 和环境日志。

- ANALYSIS_REPORT.md：中文报告，数字对应 CSV、筛选条件和列名。
- results/overall_metrics.csv、all_metrics.csv：总体与逐单位结果，两套门限分开。
- results/paired_vs_GMM.csv：性能及 TP/FP/FN 差值。
- results/curves：完整 ROC/PR 和 threshold 数组、压缩 CSV。
- predictions/*/all_cells.parquet 与 all_cells.csv.gz：全部原实例结果。
- development_review：开发集 FP/FN、原始 crop NPZ、案例索引。
- figures/zh、figures/en、figures/source_data：双语 PNG/SVG/PDF 与绘图 CSV。
- FULL_ARTIFACT_PATHS.csv：服务器全量路径。

内部验证参与检查点和阈值选择。原 GMM 复用保存的分数，不重跑旧拟合。预训练同源关系仍需完整作者图像清单进一步核实；Pan-M Gold 不称为已确认独立数据。

连续评分以 Parquet 为主要读取格式。若读取导出的 CSV，请使用 pandas 的 `float_precision="round_trip"`；保留严格 `score > threshold` 约定，不使用四舍五入后的展示值重算预测。

重跑训练时为新运行创建独立目录、官方源码副本和虚拟环境，使官方 Trainer 写入的包内 assets 检查点归属于该次运行；当前目录仅复用已完成且校验一致的结果。

曲线格式：NPZ 中 PR 点数比 PR threshold 数多 1；对应 PR CSV 最后一个端点的 threshold 为 NA。ROC 起点的 threshold 为正无穷，这是 scikit-learn 曲线约定，不代表细胞评分失败。
