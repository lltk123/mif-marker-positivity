# TabM 官方实现审计与 CD8 任务适配

审计日期：2026-10-29。本文审计公开代码，不代表训练或评价已完成。未读取 Pan-M 图像、标签或预测。

## 固定来源

| 项目 | 固定值 |
| --- | --- |
| 官方仓库 | https://github.com/yandex-research/tabm |
| 官方 commit | `28e47ae301c92ec37787dde1ce923a0793f405b4` |
| commit 依据 | `git ls-remote https://github.com/yandex-research/tabm.git HEAD` |
| 源码版本 | `tabm.__version__ == '0.0.3'` |
| 嵌入依赖 | `rtdl_num_embeddings==0.0.12`，与官方 pyproject.toml 的 uv 约束一致 |
| rtdl wheel SHA256 | `87fd61270118915cf40888f2164f3a1f0353edf702f15fd91e71a147d23c930d` |
| 兼容要求 | Python >= 3.9；PyTorch >= 1.12 且 < 3 |
| 许可证 | Apache-2.0 |

精确来源：

- [官方 README](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/README.md)
- [官方 example.ipynb](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/example.ipynb)
- [模型 tabm.py](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/tabm.py)
- [论文实验说明](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/README.md)
- [论文训练代码](https://github.com/yandex-research/tabm/blob/28e47ae301c92ec37787dde1ce923a0793f405b4/paper/bin/model.py)
- [嵌入包 0.0.12](https://pypi.org/project/rtdl-num-embeddings/0.0.12/)

作者建议实际应用使用当前 `tabm` 包；`paper/bin/model.py` 是论文实验旧接口。应使用官方 `tabm.TabM.make`，不自写骨干。现有 PyTorch 2.2.2 在声明依赖范围内，无需因 TabM 升级 CUDA 栈。

## 模型与数值嵌入

默认 `arch_type='tabm'`、`k=32`、`d_block=512`、`dropout=0.1`、ReLU。有数值嵌入时 `n_blocks=2`，无嵌入时默认 3。因此应记录解析后的全部参数，不能只记 default。

本任务建议遵循官方示例，使用 TabM + piecewise-linear embeddings，即 TabM dagger：

```python
bins = rtdl_num_embeddings.compute_bins(X_train_tensor, n_bins=48)
emb = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
    bins, d_embedding=16, activation=False, version='B',
)
model = tabm.TabM.make(
    n_num_features=X_train_tensor.shape[1], cat_cardinalities=[],
    d_out=2, num_embeddings=emb, arch_type='tabm', k=32,
    n_blocks=2, d_block=512, dropout=0.1,
)
```

`version='B'` 使用线性分支并将分段线性分支初始权重设零，之后再学习分段部分。不能遗漏版本而隐式退回 A。`tabm-mini` 是另一个变体，本轮不同时改变变体、k 和学习率。标识、患者、图像、批次、真值和 GMM 分数均不作为当前输入。

## 成员损失与推理

当前官方 notebook 对二分类也用 **2 logits + CrossEntropy**，不是 1 logit BCE；当前论文训练代码同样使用分类交叉熵。

```python
logits = model(x_num)                         # (B, 32, 2)
loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1),                    # (B*32, 2)
    y.long().repeat_interleave(model.k),      # (B*32,)
)
```

先分别计算每个成员的损失，再求均值。若沿用项目细胞权重，应先将 reduction='none' 结果重塑为 (B,k)，每细胞先平均 k 个成员损失，再做细胞权重归一化加权；是否使用权重属于任务适配，必须记录。

```python
model.eval()
with torch.inference_mode():
    member_positive = model(x_num).float().softmax(-1)[..., 1]
    positive_score = member_positive.mean(1)
```

禁止先平均 logits 再 softmax，禁止优化“平均预测的损失”。保存 32 个成员的阳性概率与成员编号。分歧统计只描述成员不同，不直接等于经过校准的不确定性或拒判理由。

## 训练专用预处理

官方 notebook 的实际路径为 noisy QuantileTransformer：

```python
noise = np.random.default_rng(0).normal(0.0, 1e-5, X_train.shape)
noise = noise.astype(X_train.dtype)
qt = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(X_train) // 30, 1000), 10),
    output_distribution='normal',
    subsample=10**9,
).fit(X_train + noise)
X_train_transformed = qt.transform(X_train)
X_val_transformed = qt.transform(X_val)
```

**noise 仅用于 fit 的训练输入，transform(train) 仍使用原始已插补训练值，不再次加 noise。** 约十万训练细胞时 n_quantiles=1000。噪声种子固定 0，三个模型种子可共用同一预处理。官方示例也列出 StandardScaler 可选路径，但此轮固定 QT，不靠外部表现更换。

本任务需在 QT 前增加缺失适配：只用训练有限值拟合中位数；先报告缺失率；插补、QT、bins 均只由训练拟合。对所有 split 保留相同行和显式 cell_id，不能因缺失删细胞。

0.0.12 的 compute_bins 明确拒绝 NaN/Inf 与任何仅有一个不同值的训练列。因此全训练缺失列、训练常量列需按训练信息剔除有效输入并报告原因，保留 requested 和 effective 特征清单，不能伪造分箱。QT 后再次出现常量的列也需记录。官方会去重 quantile edges，实际每列 bins 可以少于 48，保存每列 edges。

HGB 轻量对照可以使用完全相同的训练中位数、QT 和 effective 特征列，以区分特征与模型收益。需明确这不是旧原始特征 HGB 的同一预处理流程，旧结果仅保留参考。

## 推荐默认与本任务预算

| 参数 | 官方 notebook |
| --- | --- |
| optimizer | AdamW |
| lr | 0.002 |
| weight_decay | 0.0003 |
| batch size | 256 |
| clip gradient L2 norm | 1.0 |
| share_training_batches | True |
| patience | 16；示例 remaining_patience < 0 才停，即 17 个连续无改善 epoch |
| max epochs | 1,000,000,000，主要依赖早停 |
| AMP / torch.compile | 默认均关闭 |
| eval batch size | 8096，可按资源调整 |
| 分类早停示例指标 | accuracy |

共享 batch 的实际训练循环是 `torch.randperm(n_train, device=device).split(256)`，同一对象的 batch 同时送给全部 k 成员。独立 ensemble indices 属于 share_training_batches=False 的可选路径，本任务不采用。

预定有限预算建议：每个 A/B/C 用 seed=0 跑默认 lr=0.002 与唯一候选 lr=0.0005，其他参数固定；用内部 3 WSI 等权 AP 早停与选择配置，再补优胜配置 seed=1、2。seed=0 优胜模型直接复用为正式模型，无需重训。总计 12 次拟合，每组正式结果有 3 个随机种子。

最大 100 epochs、内部 WSI 等权 AP 早停是任务适配，不是论文原预算或原指标。开发验证用于早停、调参、阈值开发，不能将其成绩称为无偏最终测试。所有配置、阈值冻结后才评价外部 TMA。若达到预算上限，报告触顶，不声称已收敛。

官方教学示例每个 epoch 打印 test，并随机拆分示例数据；本项目不能照搬这两点。保留官方切片/患者分组，只监测开发验证集。无正类切片 AP 不应混入未经定义的等权平均，应明确 NaN/排除和有效单位数。

## 不改变方案的运行优化

1. 两个 lr 与三个模型 seed 共用训练中位数、QT 和 bins 缓存；绑定数据键、特征顺序、代码与配置哈希。
2. float32 张量一次转换，内存允许时留 GPU；评价按 batch 流式，及时将成员概率转 CPU 或写分块文件。
3. 使用 model.eval 与 torch.inference_mode；可调低评价 batch，不修改训练 batch。
4. 正式成员概率保留 float32。保存每细胞统一键、成员编号、聚合概率、种子、模型与特征版本。
5. 首轮保留官方关闭 AMP/compile 的配置。若后续启用，应记录精度与数值核验，不能宣称位级等价。
6. 减少训练样本、k 或改变训练 batch 不属于纯运行优化，不能冒充相同正式方案。

## 必要的小规模核验

- 显式逐成员 CE 的均值与 flatten CE 一致；如果使用细胞权重，也核查权重轴。
- 手工 logits 验证实现确实为 mean(softmax)，不是 softmax(mean)。
- 行顺序扰动后，按统一键连接预测仍指向同一细胞。
- 验证/测试极值改变后，已拟合中位数、QT 与 bins 哈希不变。
- NaN/Inf、全缺失列、训练常量列及无正类切片均有明确处理。
- 保存再加载模型及预处理后，同批成员概率、聚合概率一致。

最终性能、耗时、弱阳性和邻近强阳性相关假阳性结论应以实际新实验为准，不由此源码审计预设。
