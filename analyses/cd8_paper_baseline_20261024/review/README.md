# CD8 论文基线准备模块

本轮完成数据、标签、分组核查和模型无关评估接口。论文尚未指定，模型未训练。

## 已可运行的命令

```bash
ROOT=/mnt/disk3/mIHC/mif_marker_positivity_20261022
RUN=/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_paper_baseline_20261024
PY=$ROOT/analyses/cd8_morphospatial_20261023/.venv/bin/python
$PY $RUN/scripts/test_contract.py
$PY $RUN/scripts/preflight.py --project "$ROOT" --output "$RUN/artifacts_run02"
$PY $RUN/scripts/verify_evaluation_cli.py --output "$RUN/tests_run02"
```

输出目录必须不存在，已存在时停止而不覆盖。重新执行将校验当前输入，不直接使用本次核查缓存。大小/修改时间与历史清单不符时拒绝继续。大图 SHA256 本轮没有全量重算，见报告。

## 模型确定后的评估命令模板

以下入口需要先完成官方模型接入和冻结预测，当前没有论文训练入口：

```bash
$PY $RUN/scripts/evaluate_predictions.py \
  --manifest "$RUN/artifacts/cells.parquet" \
  --predictions /path/to/new_model_scores.parquet \
  --model-spec /path/to/frozen_model_spec.json \
  --split internal_val_WSIs \
  --split external_val_TMA_cores \
  --output "$RUN/new_model_evaluation" \
  --export-cell-csv
```

预测表包含且仅包含 `dataset_id,image_id,cell_id,marker,score`，覆盖所选 split 的全部原实例，包括无标签实例；不可按 QC 或 NaN 丢行。cell_id 是原整数 ID。

冻结说明要求 `status=frozen`、`model_name`、`model_version`、`paper_reference`、`official_repository`、`code_revision`、`weights_sha256`、`threshold`、`threshold_operator`、`threshold_source`、`frozen_at`。门限比较符 `gt/ge` 遵循实际模型；门限来源只能为 `official_fixed`、`training_grouped_oof` 或 `grouped_development_validation`。

`MODEL_SPEC_PENDING.json` 是待明确事项，不能冒充冻结配置。预训练数据、输入通道和预处理必须在模型训练前核查。ID/批次/真值/GMM 不作为主模型输入。

详见 [ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)，其中每项数字明确对应 CSV。全量 Parquet 留在服务器。合成测试分数不是论文结果。Pan-M 持续封存。
