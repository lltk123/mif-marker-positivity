# TabM CD8 实验

本轮在 PhenoBIC 的既有逐细胞特征上从头训练官方 TabM，并比较相同数值输入的 HGB。Pan-M 保持封存。本轮已完成 A/B/C 各 3 个种子的 TabM、相同特征的 9 个 HGB 对照，以及全量预测与评价。优先阅读[独立核对的关键结论](independent_review/INDEPENDENT_REVIEW.md)、[完整中文报告](ANALYSIS_REPORT.md)和[表格索引](results/report_tables.csv)；案例说明见[视觉核对](VISUAL_QA.md)。

## 输入与划分

- 四列唯一键：dataset_id、image_id、cell_id、marker；禁止用行号关联独立表。
- A：8 项原始 CD8 基础强度；B：A + 19 项形态；C：B + 63 项既有空间表达特征。精确顺序见 FEATURE_SETS.json 与 results/feature_manifest.csv。
- 原 3 张训练 WSI 的 101,057 个有标签细胞用于训练与拟合数值预处理。
- 内部 3 张 WSI 的 539,132 个有标签细胞为开发验证集，用于早停、两个预定学习率的选择及门限开发。
- 外部 42 个组织芯的 437,480 个有标签细胞为冻结后的测试集。组织芯不等同于独立患者；无患者映射，按原 WSI 划分。
- 全部 1,817,783 个实例都保留预测；无标签实例不进入监督训练和分类指标。
- mask 为 expanded_nucleus，近似边缘不能解释为已验证的真实细胞膜。

## 官方模型与任务适配

固定 yandex-research/tabm commit 28e47ae301c92ec37787dde1ce923a0793f405b4，TabM 0.0.3，rtdl_num_embeddings 0.0.12。使用 TabM + PLE：32 成员，2 个宽 512 的隐藏块，dropout 0.1；48 个分位数区间，16 维嵌入，version B，无嵌入 activation。

二分类输出每成员两个 logits。训练是每成员交叉熵的均值；推理是每成员 softmax 后阳性概率的均值。保存 32 个成员概率及其标准差；分歧未经不确定性校准，不用于自动拒判。

训练标签上的中位数插补与 normal QuantileTransformer 只拟合一次，变换的拟合噪声遵循官方示例，实际 transform 使用未加噪数值。分箱仅使用变换后的训练数据。训练全缺失/常量列需显式报告；本轮验证时所有 90 列均可用。原始特征与 NaN 保留在 data/cells.parquet，插补仅属于模型预处理。

预算：每个特征组合 seed 0 比较 lr=0.002 和 0.0005，其他参数一致；按内部 WSI 等权 AP 选择，然后用 seed 1、2 重复优胜配置。共 12 次 TabM 拟合，正式结果为每组合 3 种子。最多 100 epochs，16 个连续未改善 epoch 后停止。内部验证指标是开发拟合结果，不能称无偏测试成绩。

HGB 使用同一有效特征、相同训练拟合的插补和数值变换；参数沿用原固定 HGB 配置，分别 seed 0、1、2。它是本轮新建的相同预处理对照，与旧 raw-HGB 的输入流水线有区别。GMM 仅复用缓存结果评价，既有模型不重新训练。

## 运行

在服务器设置：

```bash
P=/mnt/disk3/mIHC/mif_marker_positivity_20261022
R="$P/analyses/cd8_tabm_20261029"
PY="$P/analyses/cd8_nimbus_20261024/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
"$PY" "$R/scripts/run_pipeline.py" --run-dir "$R" --stage all
```

已准备数据及完整模型可断点复用；被中断且没有完整元数据的单次训练会从头重跑，不冒充精确恢复优化器状态。配置、核心代码、输入数据、特征顺序和权重必须匹配哈希。不要修改已有运行目录后强制复用模型，应新建运行目录。

对独立原始特征表推理：

```bash
"$PY" "$R/scripts/predict_table.py" --run-dir "$R" --model-id TabM_C_s0 \
  --input input_features.parquet --output new_predictions.parquet
```

输入必须含完整四列键和模型要求的原始数值特征；不得将标签、GMM、患者编号或图像编号加入数值输入。标签及元数据只作为输出和审计字段。

## 输出

- data/：完整细胞表、数值矩阵、划分键清单、哈希和训练拟合的风险分层边界。
- models/：预处理、候选/最终模型入口、权重、训练日志和门限；完整权重留服务器。
- predictions/：18 个正式模型的全量 Parquet，TabM 含全部成员阳性概率。
- results/：总体、逐 WSI、逐 TMA、逐组织芯指标；两套门限、多种子差异、缺失、风险分层、资源记录。
- curves/：完整 ROC/PR 点和对应门限，末尾 PR 点无门限会明确标记。
- figures/：中英文 PNG、可编辑文字 SVG 和源数据 CSV。
- development_review/：仅开发集错误实例、mask 叠加、原始 crop 和邻居明细。

正式模型先保存 FROZEN_TEST_PROTOCOL.json，再计算外部性能；准备阶段已读取外部键和标签用于对齐及计数审计，不参与拟合或选择。固定门限使用 score > 0.5；补充门限仅由内部验证 WSI 等权 F1 从 0.01–0.99 中选定。无真阳性的单位 recall/AP 为 NaN；无预测阳性 precision=0；既无真阳性也无预测阳性 F1=NaN。所有此类单位在表中保留。

## 解释边界

弱阳性与接触侧亮信号等分层由训练集确定的图像特征定义，不是独立人工确认的机制标签。若 C 的假阳性降低，可描述为该代理分层的误报下降，不能直接认定已消除邻近细胞污染。三种子的标准差只描述训练随机性，不是患者层面的置信区间。本轮为论文方法在 CD8 任务上的适配，不是重现原论文所有数据集的指标。


## 全新重复运行

验证环境为 Linux、Python 3.11、PyTorch 2.2.2+cu118；实际依赖见 DEPENDENCIES.json。CPU 训练未在本轮验证，官方 TabM 入口使用 CUDA。既有 Nimbus 环境本轮只读复用；TabM 的额外纯 Python 包放在新运行的 dependencies 中，未更改旧环境。

如需另建环境，可按记录安装依赖（原始数据与上游分析输入须另行具备）：

```bash
python3.11 -m venv /path/to/new-tabm-env
PY=/path/to/new-tabm-env/bin/python
"$PY" -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118
"$PY" -m pip install -r "$R/requirements.txt"
```

固定官方源码可通过 `scripts/fetch_official.py --run-dir "$R"` 获取。只下载 commit 对应的代码与示例，不下载数据或预训练权重；已有不同文件会报错。本轮 TabM 没有预训练权重，全部从训练标签随机初始化训练。

在同一项目的 analyses 下建立新的运行目录，保留本轮全部结果：

```bash
NEW="$P/analyses/cd8_tabm_repeat_001"
"$PY" "$R/scripts/initialize_run.py" --source-run "$R" --run-dir "$NEW"
"$PY" "$NEW/scripts/run_pipeline.py" --run-dir "$NEW" --stage all \
  --font-path /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

输入依赖是既有 `cd8_morphospatial_20261023/PhenoBIC_CD8_features.parquet`、特征字典/提取代码，以及 `cd8_e1_e4_stability_20261023/audit/cells.parquet` 的显式标签与 GMM 映射。开发图例还复用原 FOV 上下文读取代码和 PhenoBIC 原图。克隆代码或取得审阅包本身不包含这些大型输入。

完整管线包括准备、少量样本检查、风险分层规则冻结、训练数值诊断、训练、冻结推理、嵌入分箱审计、评价、绘图、开发案例、完整性验证及中文报告。单步重跑通过 `--stage` 指定；不能绕过数据、代码和配置兼容性检查。

本轮 A 使用既有**原始强度**，没有增加 E1–E4 的图像级共同校准；因此结果回答本次 A/B/C 的消融问题，不应把变化仅归因于神经网络架构。

代码归档 commit 与执行文件逐一校验见 [SOURCE_CODE_VERSION.json](SOURCE_CODE_VERSION.json)。完整模型、逐细胞预测与曲线保留在本服务器运行目录；审阅包提供代码、配置、MD、汇总 CSV 和图件。
