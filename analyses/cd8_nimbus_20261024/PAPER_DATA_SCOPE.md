# 论文数据可用性与本轮范围核对

本轮仅运行 PhenoBIC CD8。作者正文的 Data availability 指向公开 Pan-M；“原研究曾使用未发表队列”不等于这些子集现在仍未公开。没有证据时，本报告不会将整个 Pan-M 或具体 Vectra 子集标成未公开。

| 数据／资料 | 已核实信息 | 本轮状态 |
|---|---|---|
| Pan-M：CODEX colon、Vectra colon、Vectra pancreas、MIBI breast、MIBI decidua | 论文说明 Pan-M 可通过公开仓库获取 | 按用户要求封存，不读取本地数据，不下载完整数据 |
| Pan-M Gold-Standard | 作者从 Pan-M 图像中生成的人工复核子集；论文原方案把这些 FOV 分到测试集 | 本轮不读取；不能据原论文 FOV 划分直接确认独立患者或独立来源 |
| 图 5 的 TNBC 展示队列 | 论文图注明确不属于原模型训练数据 | 本轮未获取、未运行；未核实其精确公开版本和逐图清单 |
| 图 5 的 DCIS／乳腺癌前病变展示队列 | 论文图注明确不属于原模型训练数据 | 本轮未获取、未运行；未核实其精确公开版本和逐图清单 |
| V1.pt 对应的完整训练图像哈希、患者映射及权重训练清单 | 固定权重仓库未提供本轮可核对的完整映射 | 缺失可审计资料，不能完全排除与 PhenoBIC 的同源关系 |

因此，能够明确声明的是“这些实验未在本轮复现”和“精确来源映射尚未核实”；没有足够证据将未运行的队列一律称为未公开。本项目的微调采用官方轻量 Trainer，不重做原始 100,000+100,000 步大规模训练。

来源：[论文作者稿](https://pmc.ncbi.nlm.nih.gov/articles/PMC13317004/)、[期刊论文](https://www.nature.com/articles/s41592-025-02826-9)、[作者归档的数据可用性说明](https://authors.library.caltech.edu/records/rjeg2-bq939)、[固定推理仓库](https://github.com/angelolab/Nimbus-Inference/tree/241281ef85ba34d9857cc03886cc45aedd0f0b34)。
