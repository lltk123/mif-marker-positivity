# CD8 归一化：限制极端细胞对整图尺度的影响

日期：2026-10-30。完成内容：文献核查与新方法定义；未运行新的特征提取、训练或评价。

## 推荐方法

针对用户明确提出的“进一步限制极端细胞对整图尺度的影响”，建议新增 **E1_outer_P1_P99**：保留每个实例内的 P10/P90，将跨实例的 min/max 改为 P1/P99。每张原始 WSI 或每个 TMA 组织芯独立计算一组共同参数。

这是文献指导的项目适配：PhenoBIC 提供实例内 P10/P90 的直接依据；CODEX 和 SPACEc 提供跨细胞表达值 P1/P99 截断的先例。**未查到论文直接验证这套双层分位数公式最优。** 原 E1 保留作为对照。

证据见 [LITERATURE_EVIDENCE.csv](LITERATURE_EVIDENCE.csv)，筛选 evidence_id 为 PHENOBIC_CELL_10_90、CODEX_CELL_1_99、SPACEC_CELL_1_99，查看 statistical_population、percentiles、evidence_scope 和 project_implication 列。精确定义见 [PROPOSED_NORMALIZATION.json](PROPOSED_NORMALIZATION.json)。

## 具体公式

设实例 i 的掩码为 M_i，首先保留现有实例内分位数：

\[
l_i=P_{10}(I\mid M_i), \qquad u_i=P_{90}(I\mid M_i).
\]

在同一校准单位的全部合格实例之间，等权计算：

\[
L=P_1(\{l_i\}), \qquad U=P_{99}(\{u_i\}).
\]

将共同映射应用到本单位所有有效像素：

\[
I_{\mathrm{norm}}(x)=\operatorname{clip}\left(\frac{I(x)-L}{U-L},0,1\right).
\]

百分数记法 1/99 对应 NumPy quantile 的参数 0.01/0.99；两层均使用 float64 和 linear 分位数插值。

每个实例只贡献一个 l_i 和一个 u_i，不按细胞面积加权。原 E1 的区别仅在外层：L=min(l_i)，U=max(u_i)。

**不删除最亮或最暗的细胞，不删除标签，也不将尾部细胞直接判成阴性。** 全部实例继续保留。高于 U 的像素在新的模型输入中截到 1，原始图像及已有结果保持原样。

P99 并不意味着整图恰有 1% 像素被截断，更不意味着删除 1% 细胞。实际截断比例须另算；并列值、样本量和分位数插值会影响尾部计数。

## 为什么选 P1/P99

[Hickey 等，Frontiers in Immunology，2021](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.727626/full) 的 Methods → Normalization Techniques → Min_Max Normalization，对分割后的单细胞 marker 表达矩阵使用 P1/P99 限制极端值，再做 min–max。这为跨细胞分位数限制极值提供了直接的多重成像分析先例；论文没有研究“实例内 P90 的跨实例 P99”。

[Tan 等，SPACEc，Nature Communications，2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12660697/) 的 Methods → Normalization 同样列出单细胞表达矩阵 P1/P99 min–max 选项。但论文实际分析采用 Z-score，因此不能把它描述成 P1/P99 优于其他方法的证明。

[PhenoBIC 原文，2026 预印本](https://pmc.ncbi.nlm.nih.gov/articles/PMC13308180/) 的 Methods → Preprocessing 采用实例内 P10/P90，再汇总外层 min/max，直接对应现有 E1。作者报告整图像素 P0.01/P99.99 截断会放大背景并降低精确率。这个结果支持保留实例层面的边界统计；该来源尚未同行评议。

上述事实分别对应证据表 CODEX_CELL_1_99、SPACEC_CELL_1_99、PHENOBIC_CELL_10_90、PHENOBIC_GLOBAL_ABLATION 四行。

## 不混用不同文献的统计对象

- [Cellpose](https://doi.org/10.1038/s41592-020-01018-x) 的 P1/P99 是逐图像像素归一化，用于分割；[固定代码 v3.1.1.1](https://github.com/MouseLand/cellpose/blob/v3.1.1.1/cellpose/transforms.py#L151-L179) 并没有把结果截断到 [0,1]。它不是本项目跨细胞双层公式的验证。
- [Nimbus 已发表论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13317004/) 的 Methods → Preprocessing / Model inference 使用数据集内、通道级像素 P99.99。它不是外层细胞分位数。
- [固定 Nimbus-Inference 代码](https://github.com/angelolab/Nimbus-Inference/blob/241281ef85ba34d9857cc03886cc45aedd0f0b34/src/nimbus_inference/utils.py) 默认使用各 FOV 正值像素 P99.9，再对选定 FOV 的值按 marker 求均值。代码默认与论文 P99.99 的数值和汇总描述应分开记录。

对应证据表 CELLPOSE_PIXEL_1_99、NIMBUS_PAPER_99_99、NIMBUS_CODE_99_9。仅检索方法与公开代码，未访问 Pan-M 数据。

## 计算规则

- 校准单位保持原 E1 定义：原始训练 WSI；按原坐标去重合并的内部验证 WSI；独立外部组织芯。
- 原始实例至少包含 5 个有限有效像素即可参与，不依据人工标签、模型预测或结构特征 QC 挑选。
- 每个实例的像素集合保持原 mask 和 ID；有效零值不剔除。估计边界不混入未分割背景。
- 大图先在完整校准单位估计 L/U，再分块推理；目标、邻居及局部参照上下文使用同一映射。
- 无有效实例或 U-L 在原 E1 的相对精度条件下退化时，保留全部键，记录 calibration_invalid 和 NaN；不擅自换边界。
- TabM 需从新像素重提强度和空间特征，并重新拟合训练预处理及模型。几何形态定义不变。
- Nimbus 新适配独立命名；采用该映射后不再除以 17.2643。微调与推理保持同一输入定义。

## 需要验证的权衡

极亮实例可能是真实 CD8 阳性。P99 可降低异常细胞对上界的影响，也可能在阳性很稀少时把上界降得过多，压缩真阳性之间的强度差异或放大背景。P1/P99 是有文献先例的预定起点，并非文献已经证明的本数据最优参数。

后续应记录每单位实例数、原/新 L/U、上界变化、尾部实例数量、实际像素截断率和退化情况，并检查强阳性形态是否过度饱和。比较时保留原 E1，报告 precision、recall、F1、AP 与 TP/FP/FN 的变化。

本轮只完成方法选择和可复现定义，尚无新性能结果；没有替换既有代码、数据、模型或报告。
