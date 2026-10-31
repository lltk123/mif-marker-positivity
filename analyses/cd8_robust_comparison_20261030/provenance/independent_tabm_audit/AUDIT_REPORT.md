# TabM 独立数值审计

结论：未发现阻塞性错误。未改动训练、权重、冻结配置、正式指标或报告，也未访问 Pan-M。

18 个正式监督模型覆盖 TabM/HGB × A/B/C × 种子 0/1/2。保存的训练记录对应 12 次 TabM 拟合（种子 0 的 6 个候选，以及种子 1/2 的 6 次拟合）和 9 次 HGB 拟合。canonical TabM 目录是被选候选的引用，不另计为一次训练。依据 `LOCAL_AUDIT.json` 与 `LOCAL_CHECK_SUMMARY.csv`；完整逐项检查表共 229,883 行，仅保留服务器同目录 `LOCAL_CHECKS.csv`。

服务器实际读取了 18 个完整预测 Parquet，逐模型核对 1,817,783 个显式四键、文件 SHA 和冻结协议；加上原 GMM，重算全部 1,998 行正式指标。TP/FP/FN/TN 及评价细胞数量完全相同，浮点指标最大绝对差为 2.78×10⁻¹⁷。9 个 TabM 的全部 32 成员概率均值与保存分数一致至 float32 累加精度，float64 重新求均值的最大差为 9.88×10⁻⁸。两套正式预测标签与正式分数解释均无不一致。来源：`FULL_PREDICTION_AUDIT.csv` 的 `n_all_cells/metrics_rows_checked/max_metric_error/max_member_mean_error/stored_prediction_mismatches`，全部模型；执行源码与环境见同名 JSON。

本地独立核验了 seed 汇总、单位等权汇总、配对变化和报告表；综合报告的新旧 TabM/HGB 数值逐项与各自正式来源一致。全部正式内部门限按保存的验证分数及其原 dtype 重算成功。内部验证用于早停、配置选择和门限开发，其成绩不能称为独立泛化成绩。

外部固定 0.5、三个种子的平均 F1：A=0.839332、B=0.851751、C=0.840610；平均 AP：A=0.948203、B=0.945115、C=0.957581。来源：`analyses/cd8_tabm_robust_20261030/results/report_performance_fixed_0.5.csv`，筛选 `split=external_val_TMA_cores, group_type=split, model_family=TabM`，列 `f1_mean/average_precision_mean`。B 的 F1 最高，C 的 AP 最高。C 相对 B 平均 FP 减少 299.3、FN 增加 605.7，不能把误报减少自动解释为污染已解决；来源同 run 的 `results/report_paired_deltas.csv`，相同 split/group_type、`threshold_mode=fixed_0.5, comparison=TabM_C_minus_B`，列 `delta_fp_mean/delta_fn_mean`。

## 审计 dtype 诊断

首版审计将原 float32 验证分数提升为 float64，导致 TabM_B_s1 在未被选中的门限 0.99 处出现差异：内部 WSI3 的 1 个真阳性细胞由等于 float32 门限变成高于 float64 门限，等权 macro F1 差 4.1104320232×10⁻⁵。按原 dtype 重建后全部 99 项网格完全一致；实际门限始终为 0.86。此模型全部 539,132 个内部验证分数与全量推理分数逐位相同。全部正式模型的获胜门限、主结果和已保存预测均未受影响。仅修正独立审计的重建口径，没有更改实验。

来源：`THRESHOLD_DTYPE_DIAGNOSTIC.csv` 的 `score_source_and_dtype/max_grid_error/winner`，以及 `THRESHOLD_DTYPE_CHANGED_COUNTS.csv` 的 `threshold/image_id/n_changed/n_true_positive`；完整诊断与源码哈希见 JSON。独立审计 v2 的全模型结果显示，仅上述未获胜网格点对 dtype 提升敏感。

## 复核范围与复现

`audit_local_tables.py` 是实际执行的 Windows 只读表格审计源码，项目根路径固定为当前工作区；它只向自身目录写审计材料。`audit_tabm_predictions_readonly_v2.py` 是服务器实际执行源码，使用原 Nimbus 环境的 Python、NumPy 1.26.4、pandas 2.2.3，分别令 `AUDIT_PART=0/1` 完成两组模型。它只向明确命名的 `/tmp/cd8_tabm_independent_audit_20261031_v2_part*` 写小型审计表，并拒绝覆盖已存在目录。`diagnose_threshold_grid_with_counts.py` 为 dtype 诊断源码。再次运行应使用新的独立审计输出目录并保留原记录；不要覆盖本目录已完成证据。所有最终证据现已保存到项目，而非仅保留于临时目录。

风险分层表的非空分母、F1 和跨模型细胞集合已核验；空分层 `n_cells=0` 时保留 NaN，不视为阴性观测（见 `RISK_TABLE_CHECK.json`）。未开展新的患者级独立性核验、显著性检验或训练。视觉 QA 由专门审阅者完成，见原 TabM `figures/FIGURE_QA.json`；本报告不把数值审计替代视觉或生物学机制验证。
