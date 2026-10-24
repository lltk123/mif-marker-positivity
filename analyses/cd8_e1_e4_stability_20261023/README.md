# E1–E4 与稳定性复现

本轮仅 PhenoBIC CD8；Pan-M Gold 封存且未读取。所有新文件位于本目录；原始数据、mask、ID、标签、划分和旧模型保持不变。全量 Parquet 保留服务器，审阅包包含 MD、CSV、配置、源码、字典、图件与少量示例。

## 执行入口

```bash
cd /mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_e1_e4_stability_20261023
PY=/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_morphospatial_20261023/.venv/bin/python
$PY scripts/test_compact.py
$PY scripts/batch_extract.py prepare
$PY scripts/batch_extract.py full --workers 8
$PY scripts/train_new.py
$PY scripts/evaluate.py
$PY scripts/stability.py --workers 6
$PY scripts/export_examples.py
$PY scripts/make_documentation.py
```

不要重跑旧实验。审计源码仅用于追溯；原 GMM 可以用 calibration/original_gmm_states.json 加 score_fixed 核对，无需正常输入重拟合。历史正常输入校验在用户最新指示前已结束。本轮新的监督拟合严格为 12 个 OOF 折加 4 个最终模型；稳定性中 192 次 GMM 重拟合属于规定的亮度扰动检查。

缓存校验数据清单、配置、代码、键表和模型哈希，不兼容时拒绝复用。输入清单还保存源文件 SHA256、大小、修改时间。完整重做必须另建运行目录并重新生成审计键表与协议，不能把另一数据版本替换到同一路径后复用缓存。

## 主要产物

- PhenoBIC_E1_E4_features.parquet：1,817,783 个实例的 17 项新特征、QC、支持像素数和来源。
- predictions/training_OOF_*.parquet：101,057 个有标签训练细胞的四组 OOF 分数、两套门限预测。
- predictions/final_all_cells_*.parquet：全部实例的七组分数/预测、原标签、QC 和两套门限。训练最终预测是重代入。
- models/E1.joblib 至 E4.joblib：最终模型、特征顺序、门限、训练来源、配置；另留 12 个 OOF 折模型支持断点复核。
- calibration/calibration_parameters.csv：48 单位 L/U、贡献 ID、原始动态范围与裁剪比例；bound_contributors.parquet 保留全部贡献 ID。
- keys/：各划分全体/有标签键、固定抽样清单和哈希。
- results/：完整预定有标签集合的分层性能、配对变化、缺失率及 QC CSV。
- curves/：完整 ROC/PR 及 threshold 数组；PR 阈值数组比端点短 1，见 curve_index.csv。
- stability/paired_cell_results.parquet：9,600 细胞×9 扰动×7 模型，含两套门限预测、输入有效性、输出有限性及配对有效性。
- stability/stability_*.csv：每单位/划分/TMA 的完整稳定性统计；perturbed_feature_QC.parquet 保留扰动后缺失原因；gmm_states/ 保存全部扰动拟合状态。
- tests/full_unit_affine_equivalence.csv：5 个代表完整单位×4 亮度扰动逐像素重算核验。

## API 与方法边界

inference.py：calibrate_instances、extract_cell、predict_features。校准必须使用完整 WSI/组织芯；内部 FOV 先重建并去重，不能用 200 个抽样细胞代替完整单位。io_adapter.py 复用原实例索引与上下文重建。gmm_interface.py 实现原配置 fit_score 与 score_fixed。

正增益/共同偏置的 L/U 和全单位原始均值使用经代表完整单位逐像素验证的数学等价加速；各完整单位 GMM 实际重新拟合。旋转共同变换 ROI 中图像、所有 ID 和有效区域，8 px halo 覆盖目标、参照、邻居及 mask 扰动所需支持；核验均值不变后才复用 GMM 分数。mask 变化冻结原 L/U 与 GMM 映射，将新均值输入原映射，保留原 MinMaxScaler clip=False 外推，不额外截断。

主结果严格 score>0.5。补充门限仅由新 E1–E4 的三张训练 WSI OOF 等权 F1 选择。旧三组在两个面板均保留原 0.5。同一 OOF 上选门限后的成绩标为门限开发拟合值，不能称门限选择的无偏泛化成绩。没有验证集调参、额外种子或校准器。

正式指标不因特征缺失删行。稳定性中输入失效即使 HGB 返回有限分数也不可评分；S 局部缺失但必要强度有效时仍评分。翻转率以双方可评分的配对为分母，同时用原抽样总数报告翻转或不可评分比例。稳定性必须结合原始分类准确性解释。


## 最终补充与断点恢复

交付版 OOF 使用 `predictions/training_OOF_*_with_QC.parquet`，已补入 QC、必要强度输入有效性和校准有效性；原 OOF 分数未改变。原始 OOF 表仍保留。

在已建立本轮协议和审计目录的服务器中，可使用：

```bash
$PY scripts/resume_checked.py extract
$PY scripts/resume_checked.py train
$PY scripts/resume_checked.py evaluate
$PY scripts/resume_checked.py stability
```

该入口先重新核验实际源文件 SHA256 和冻结代码，再执行相应阶段的兼容缓存检查；已完成阶段核验后直接返回，不重复拟合。校验失败应保留原结果，另建运行目录，不能跳过哈希检查。

后处理顺序为 `complete_delivery.py`（首次补齐 OOF QC 与完整画布旋转检查）、`validate_run.py`、`test_inference_delivery.py`。完整画布旋转验证见 `tests/full_field_rotation_checks.csv`。

绘图和中文报告在本地 Python 环境生成，可运行 `python scripts/plot_results.py --root <审阅目录>` 及 `python scripts/make_report.py --root <审阅目录>`。绘图软件版本和中文字体记录在 `figures/PLOTTING_RUNTIME.json`；在其他系统生成中文图需配置可用中文字体，英文图不依赖该 Windows 字体。所有图件数值来源均为包内 CSV。

本轮没有根据验证成绩修改校准、特征、训练参数或门限。后处理补充仅增加 QC 元数据、接口测试、全图旋转验证及报告图表，不改变模型或已有分数。
