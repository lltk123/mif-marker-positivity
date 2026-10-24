# 图件审阅
所有 16 幅双语图均提供 PNG、可编辑文字 SVG 和 PDF。SVG 文字元素、PNG 尺寸、文件哈希及最终排版抽检范围见 [FIGURE_QA.csv](FIGURE_QA.csv)。

31 个开发集案例面板已逐例核对，记录见 [VISUAL_REVIEW.md](../development_review/VISUAL_REVIEW.md) 与 [visual_review_notes.csv](../development_review/visual_review_notes.csv)。这是图像属性审阅，不是病理专家重新标注；原始真值没有改变。

修正了逐组织芯图的组名遮挡，以及案例图第二行标题重叠。最终对中英文两行案例图、单行截断案例图和统计图进行排版抽检。数值、模型、门限、样本集合和显示灰度选择均未因此改变。

图中扩张核边界不能解释为已验证的细胞膜；组织芯不能作为独立患者。案例图逐幅设置显示对比度，不能跨面板比较绝对亮度。所有图例均保留这些限制。
