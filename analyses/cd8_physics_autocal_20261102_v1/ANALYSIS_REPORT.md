# 2026-11-02：CD8 技术扰动、自校准与条件训练结果

本次只补做此前缺失计算，旧权重、干净预测、随机 20/50/100 标签校准与固定病例均复用。外部是已反复查看的探索性基准，不能称新盲测。结果中的比例均为 0–1，弱阳性指冻结定义的弱**观测**信号阳性，不是独立专家判定的生物学真值。

## 主要结果与证据

1. **Stage0 已完成 48/48。** 每组覆盖 539,132 个内部细胞，16 种变换×3 seed；干净输入复用原预测。95% 零值目标下 ENS F1 从 0.891812 升至 0.919382，但弱信号召回从 0.767309 降至 0.129620。总体 F1 不能替代弱信号检查。证据 T05：[STAGE0_FULL_METRICS.csv](results/STAGE0_FULL_METRICS.csv)，筛选 `model=ENS_A, unit=pooled, threshold_mode=dev_FPR01_each_WSI, scenario=clean/zero_0.95`，列 `F1, AP, weak_recall, FPR`。这是归一化后的 crop 级技术代理，不是曝光或染料浓度的物理复刻。

2. **零标签 Gate1 未通过。** 空间候选方法在外部全量 437,480 例中得到 F1=0.892868125、AP=0.965875528、弱信号召回=0.507091129、FPR=0.002579797。其 AP 比预设 0.966 少约 0.000124472，不能四舍五入为通过。混合法也未通过；ridge 未过开发门槛，因此未评价外部。证据 T01：[ALL_EXTERNAL_SIX_METRICS.csv](results/ALL_EXTERNAL_SIX_METRICS.csv)，`model=ENS_A, method=spatial/mixture/reused_baseline, level=pooled, threshold_mode=dev_FPR01_each_WSI`，列 `n,F1,AP,weak_recall,FPR`；门槛见 [GOALS_AND_BOUNDARIES.json](results/GOALS_AND_BOUNDARIES.json)。

3. **主动 20 标签/core 的 Gate2 仅重复均值通过。** 每次 42 core 共 840 个标签，先共同排除每 core 的 100 个预留实例（合计 4,200），评价 433,280 例。3 次均值 F1=0.903875168、弱信号召回=0.557923478；只有重复 101/102 通过，103 为 F1=0.900121774、弱信号召回=0.538368029。证据 T03：[ALL_FEWSHOT_POOLED.csv](results/ALL_FEWSHOT_POOLED.csv)，`strategy=active_boundary,budget=20`，列 `replicate,n,F1,weak_recall,FPR`；均值/SD 见 T04：[ALL_FEWSHOT_SUMMARY.csv](results/ALL_FEWSHOT_SUMMARY.csv)。这与全量零标签评价的分母不同，不能混算。

4. **技术一致性训练未过完整内部门槛。** 3 训练 WSI 留一×4 分支共 12 次开发拟合，仅 physics 通过选型；其与配对继续训练对照各 3 seed，共 6 次最终拟合。内部 ENS ΔAP=+0.003187283、ΔF1=−0.003022875，三 seed 平均 F1 亦下降。之后的 3 条件合成复验及新模型全部外部推理未执行，未换候选继续调参。证据 T06：[CONTINUATION_CV.csv](results/CONTINUATION_CV.csv)，`variant,fold,F1,AP`；T07：[CONTINUATION_INTERNAL_METRICS.csv](results/CONTINUATION_INTERNAL_METRICS.csv)，`level=pooled,threshold_mode=dev_FPR01_each_WSI,model=NEW_ENS_control/NEW_ENS_physics`，列 `F1,AP`；[停止记录](results/CONTINUATION_EXTERNAL_SKIPPED.json)。

5. **保留 core 层面的退步与误报。** 空间法在 42 core 中 27 个 F1 提高、14 个降低、1 个持平；平均 core 阳性比例误差由 0.007649 降至 0.006078。两种截距校准的逐 core AP 完全不变，池化 AP 变化来自跨 core 排序。主动 20 的重复 103 有 1 个 core FPR=1.2825%，不能用池化 FPR 掩盖。证据：[CORE_DISTRIBUTION_DIAGNOSTIC.csv](results/CORE_DISTRIBUTION_DIAGNOSTIC.csv)，`method` 分组；T09：[PAIRED_CORE_DELTAS.csv](results/PAIRED_CORE_DELTAS.csv)，`model=ENS_A,method=spatial,threshold_mode=dev_FPR01_each_WSI`；T10：[NEW_FEWSHOT_METRICS.csv](results/NEW_FEWSHOT_METRICS.csv)，`strategy=active_boundary,budget=20,replicate=103,level=core`，列 `FPR`。

## 验证与未完成边界

原运行 14+17+1 项测试通过；Stage0 136 组、最终 59 组、零标签开发 204 组独立指标复算通过，最大绝对误差约 1.33×10⁻¹⁵。18 个新权重的哈希和实际初始化/采样记录保留。神经活动计时 5,397.58 秒（1.499 小时），低于 8 小时上限；其中拟合约 409.73 秒、推理约 4,978.21 秒，余量为 smoke 与记账，不是 GPU 核函数耗时。见 [最终审计](results/FINAL_AUDIT_COMPLETE.json)、[零标签独立复算](results/ZERO_DEVELOPMENT_AUDIT_COMPLETE.json)、[预算](results/BUDGET_BREAKDOWN.json)。

55 例含 27 例内部、28 例外部，个体级图像及 640 条预测对照保留本地，不发布；全部专家生物学判读仍待完成。已完成科学图视觉检查，HTML 只完成结构/链接核验，未宣称浏览器实测。缺少真实曝光/染色对照、患者及实染色批次映射，不能推出唯一技术漂移原因，也不能将 core 当独立患者。

[中文完整报告](reports/EXECUTION_REPORT_ZH.md) · [English full report](reports/EXECUTION_REPORT_EN.md) · [表格索引](TABLE_INDEX.csv) · [结论索引](CONCLUSION_INDEX.csv) · [复现说明](README.md)。
