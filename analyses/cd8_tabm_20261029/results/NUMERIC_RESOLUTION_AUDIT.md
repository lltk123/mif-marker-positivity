# 训练特征数值分辨率审计

本审计只统计原训练 WSI 的有标签细胞；不读取验证/测试成绩，不重拟合变换、不调整模型或特征。

- 训练细胞数：101057。
- 请求特征数：90；有效特征数：90。
- 数值表：`numeric_resolution_train.csv`，一行一个请求特征。
- 列定义：`raw_nunique_train` 为原始有限值的不同值数；`transformed_float32_nunique_train` 为现有模型输入的不同值数。
- `unique_retention_ratio` = 转换后不同值数 / 原始有限值不同值数；缺失中位数插补可能引入一个新值，因此此比例可略大于 1。`imputed_unique_retention_ratio` 使用训练中位数插补后的不同值数作为分母。
- `raw_std` 和 `transformed_float32_std` 均使用 ddof=0；NaN 单独计数。
- `noise_to_raw_std_ratio` = 1e-5 / 原始训练有限值标准差，仅描述量纲。
- 少于 16 个转换值或保留不足 10% 的标记为工程审计提示，不是性能结论、阳性判定或删除特征规则。

## Hu moments 与固定绝对噪声

官方 QT 示例在拟合输入上加入标准差 1e-5 的绝对噪声，实际 transform 使用未加噪声的值。高阶 Hu moments 的量纲可能远小于该噪声；随后转换到 float32 可能压缩可区分值数。即使 compute_bins 成功，也只证明至少存在两个不同训练值，并不证明所有原始数值分辨率均被保留。

下表是训练数据的描述诊断，不据此或任何验证表现修改本轮已冻结流程。若下一轮探索训练拟合的尺度预处理，应另立实验并与本轮明确区分。

| 特征 | 原始不同值数 | float32 不同值数 | 插补后唯一值保留率 | 噪声/原始 std | 诊断 |
| --- | ---: | ---: | ---: | ---: | --- |
| HuMoments1 | 100911 | 100759 | 0.998494 | 0.000328355 | no_large_unique_loss_by_audit_rule |
| HuMoments2 | 100995 | 100888 | 0.998941 | 0.000685281 | no_large_unique_loss_by_audit_rule |
| HuMoments3 | 100956 | 100835 | 0.998801 | 0.0208867 | no_large_unique_loss_by_audit_rule |
| HuMoments4 | 100963 | 100604 | 0.996444 | 0.0930611 | no_large_unique_loss_by_audit_rule |
| HuMoments5 | 100964 | 82244 | 0.814587 | 19.2457 | no_large_unique_loss_by_audit_rule |
| HuMoments6 | 100963 | 100029 | 0.990749 | 0.31983 | no_large_unique_loss_by_audit_rule |
| HuMoments7 | 100939 | 70378 | 0.697233 | 133.099 | no_large_unique_loss_by_audit_rule |

输入与脚本 SHA256 见 `numeric_resolution_audit_manifest.json`。
