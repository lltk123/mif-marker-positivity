# 开发案例与图件视觉核对

已逐图查看全部 9 个开发案例的原始 CD8 与 mask 叠加 PNG，并核对图中文字与 case_index.csv。此处为工程视觉检查，不是新的病理人工复核；保留原始标签。

逐例观察、统一键和变化类型见 [visual_qa_cases.csv](visual_qa_cases.csv)。评分、原图/掩码路径、显示范围、有效像素数和 QC 来自 [case_index.csv](development_review/case_index.csv)，按 case_name 对应；其余邻居 ID 见 [neighbor_context.csv](development_review/neighbor_context.csv)。

- 原图面板与叠加面板使用同一裁剪和显示范围。显示上限为本裁剪有限像素 p99.5，gamma=1；不同案例显示上限不同，不能仅凭图片明暗跨案例比较绝对强度。原始像素 NPZ 留服务器，没有用于模型的逐细胞 min–max。
- 橙色是扩张核 mask，灰色是邻居实例边界，比例尺单位为 px。边缘不能解释成已证实真实细胞膜。
- 初版邻居 ID 文字过密，已改为最多 6 个最近且不遮挡的邻居标签，全部边界和完整邻居表仍保留。极小非零分数改为科学计数显示。
- mask 不完整类别实际只导出 1 例，其余四类各 2 例；这是选例结果，不是遗漏的第十张图。类别与候选数量见 case_availability.csv。
- 既有 corrected_FN、corrected_FP，也有 introduced_FN、introduced_FP 和 persistent_FN，不能仅选择成功示例推断总体收益。总体性能以完整集合的 CSV 为准。
- 初版逐组织芯和成员分歧图的 context 筛选不匹配实际 `_CD8` 后缀，已修正并加入拒绝空面板的检查。修复记录及保护文件哈希见 [visual_fix_audit.json](results/visual_fix_audit.json)。重绘后模型、预测、指标与 case_index 均保持原哈希。

重点限制：case_08 的 qc_disconnected=True，多个原始特征为 NaN；虽然分类器经训练中位数插补后返回有限的极低评分，这不证明该碎片实例有可靠阴性信号。case_02 的近邻亮度代理对应纠正阳性漏检，也不能把整个代理类别称为已确认污染。
