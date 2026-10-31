# TabM / HGB：E1_outer_P1_P99 重训练

当前状态：全量特征接入、TabM A/B/C 三种子与匹配 HGB 三种子共 18 个正式模型、逐细胞预测、评价、图件及中文报告已完成。实际结果与覆盖率见 ANALYSIS_REPORT.md；原始数据、模型权重、全量逐细胞文件和完整曲线数组保留服务器。

本轮保留旧 TabM A/B/C 实验的官方实现、训练预算、三个种子、内部验证选择方式和匹配 HGB 对照，仅将模型特征来源换为共享提取器从归一化像素重新计算的 90 项特征。官方 TabM commit 固定为 `28e47ae301c92ec37787dde1ce923a0793f405b4`；未读取 Pan-M。

共同校准每个原 WSI / 内部重建 WSI / 外部组织芯独立计算：
`li=P10(I|Mi), ui=P90(I|Mi), L=P1({li}), U=P99({ui})`，
再以共同浮点映射 `clip((I-L)/(U-L),0,1)` 处理像素。参与估计的实例至少有 5 个有限像素，包含无标签实例。该方法受文献指导，但本项目的双层公式并非文献已验证最优。

- A：8 项重新提取的归一化强度。
- B：A + 19 项原有几何形态。
- C：B + 63 项从同一归一化像素重提的空间表达特征。
- 仅原 3 张训练 WSI 的 101,057 个有标签细胞进入监督损失与预处理拟合。
- 内部 3 张 WSI 用于早停、两个学习率候选选择和门限选择；其性能是开发结果。
- 外部 42 个组织芯标签仅在所有模型与门限冻结后用于评价。此前项目已多次观察这些外部数据，本轮不能称为全新的独立确认测试。
- 无标签实例保留预测；特征缺失、校准失败都不删除细胞。正式指标同时给出数值评分覆盖率和图像校准覆盖率。
- 旧 GMM 分数仅作原基线，严格大于 0.5；不进入 TabM / HGB 输入。

原始数据、旧模型和旧结果不修改。新模型仍用 `TabM_A_s0` 等局部名称，但模型、逐细胞输出和指标携带 `normalization_method=E1_outer_P1_P99`。跨目录汇总必须同时使用运行目录和归一化方法，不能只凭局部模型名关联。

## 必需输入

项目根目录下：

1. `analyses/cd8_robust_calibration_20261030/PhenoBIC_CD8_features.parquet`
2. 同目录的 `extraction_manifest.json`、`PROTOCOL.json`、`FEATURE_DICTIONARY.csv`、`calibration/unit_bounds.csv`
3. `analyses/cd8_robust_comparison_20261030/RETRAIN_PROTOCOL.json`
4. 原形态特征表、特征定义源码、原 E1–E4 `audit/cells.parquet`
5. 原 TabM `data/risk_strata.json` 与固定上游源码/依赖

共享 manifest 的 complete 状态、方法名、计数和文件 SHA 必须全部通过。缺少全量输入时明确失败，不生成空预测或成功标记。连接使用服务器原有兼容 Python 环境；不自动安装环境。

## 执行

以下 `P` 为实际项目根目录，`PY` 指向既有 Nimbus Python。先同步本目录源码，再仅复制原 TabM 的固定官方源码和依赖：

```bash
P=/mnt/disk3/mIHC/mif_marker_positivity_20261022
R="$P/analyses/cd8_tabm_robust_20261030"
PY="$P/analyses/cd8_nimbus_20261024/.venv/bin/python"
"$PY" "$R/scripts/bootstrap_runtime.py" --source-run "$P/analyses/cd8_tabm_20261029" --run-dir "$R"
"$PY" "$R/scripts/test_contracts.py"
"$PY" "$R/scripts/test_calibrated_inputs.py"
"$PY" "$R/scripts/run_pipeline.py" --run-dir "$R" --dry-run
```

统一全量入口（必须先协调 GPU 使用；锁不应强制抢占其他任务）：

```bash
flock -n "$P/analyses/.cd8_robust_gpu.lock" "$PY" "$R/scripts/run_pipeline.py" --run-dir "$R" --stage all --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

CPU 步骤可提前独立执行 `--stage prepare`、`--stage numeric-audit`；`pilot` 也使用 GPU，需等锁与调度。 `--stage validate` 中的小样本 API 一致性检查同样使用 GPU，独立执行时会自行取得共享非阻塞锁；不要为这个独立入口重复套外层锁。全量 `--stage all` 仍按上述外层 `flock` 命令运行。断点续跑依据数据、代码、上游和配置哈希校验，不复用不兼容缓存。

## 核查与结果

逐细胞键为 `dataset_id,image_id,cell_id,marker`，没有行号匹配。原始均值核查使用 `raw_observed_mean_CD8`，归一化 `observed_mean_CD8` 不与原始值比较。19 项形态逐值核对，固定 expanded_nucleus 掩码。

`audit_raw_*` 只作为审计元数据保存：旧训练集的弱阳性和邻居亮信号代理群组/门限保持相同，避免输入校准同时更换错误分析群组。它们不会进入模型输入。插补、正态分位数变换、分箱仍只在训练标签集合拟合。

主要输出：`data/cells.parquet`、`models/`、`predictions/`、`results/metrics_all.csv`、逐来源/核心/风险/缺失/成员分歧表、完整 ROC/PR 数组、图件和中文报告。大表与权重留在服务器。最终报告脚本依赖完整 18 个监督模型、GMM 基线、预测哈希、完整曲线、开发案例和交付验证，不以代码可运行代替全量完成。
