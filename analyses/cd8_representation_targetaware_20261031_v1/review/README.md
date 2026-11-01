# Real-cell review / 真实病例复核

31 个真实细胞已完成展示 QA，**生物学人工判读尚未完成**。公开许可未确认，本仓库不发布病例图像、原始 crop、mask、病例级 ID/分数清单或私人下载链接。

The 31 cases passed display QA only; biological adjudication remains pending. They are illustrative samples selected by predefined source/error-proxy rules, not a population error-rate estimate.

经授权持有原始数据和预测后，可在独立运行目录使用 [stage4_review.py](../scripts/stage4_review.py) 生成审阅包；填写 `user_decision`, `segmentation_issue`, `reviewer_confidence`, `reviewer_notes`。原始 96×96 像素与实例 mask 保留，显示为记录的线性范围，尺度条为 20 pixels，不代表物理长度。图像仅用于原授权环境内的判读，未经许可不要再次分发。

Display QA does not replace expert labels. Review should distinguish label disagreement, segmentation/target attribution, weak signal and bright-neighbor proxies. No synthetic images were substituted for real cases.
