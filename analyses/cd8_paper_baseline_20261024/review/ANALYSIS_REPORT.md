# CD8 论文基线：接口实现与数据核查

**状态：基础接口与数据核查已完成；论文待用户指定，模型实验未执行。**

用户明确论文将于后续提供。本轮没有选择替代模型、修改架构、训练新分类器或生成真实细胞的新预测。原始数据、mask、GMM、E1–E4 和既有结果保留，Pan-M 持续封存。状态逐项见 [work_status.csv](work_status.csv)。

## 实际细胞集合与标签

| 官方划分 | 全部实例 | 有二分类标签 | 无标签实例 | 切片分组数 |
| --- | ---: | ---: | ---: | ---: |
| external_val_TMA_cores | 438,236 | 437,480 | 756 | 3 |
| internal_val_WSIs | 667,125 | 539,132 | 127,993 | 3 |
| training_WSIs | 712,422 | 101,057 | 611,365 | 3 |

来源：[cell_counts.csv](cell_counts.csv)，按 `split` 筛选，读取 `n_all/n_labelled/n_unlabelled/n_group_units`。全部共 **1,817,783** 个细胞，有标签 **1,077,669** 个；`n_unresolved=0`。无标签实例继续保留，不进入监督损失或分类指标。

全量键为 `dataset_id + image_id + cell_id + marker`，marker 为 CD8。所有 ID 保留为原实例整数，没有按行号推断。已核对全量实例索引与原特征表键，独立重读 9 个原始 CD8 CSV，并按 ID 和 TMA 的 `Patch` 连接；新关联标签与既有细胞表完全一致。原 CSV 的 `label` 是实例 ID，`CD8` 的 pos/neg 才是类别。

官方 README 将这些标签描述为人工注释并经独立质量复核。清单的 `label_type` 为 `manual_independently_quality_reviewed` 或 `unlabelled`，保留原字符串，不从 GMM 生成标签。9 个文件均重新计算 SHA256，见 [label_file_hashes.csv](label_file_hashes.csv)。

当前 CD8 涉及 **176** 对图像/mask，构成 **48** 个图像单位：训练与内部各 3 张 WSI，外部 42 个组织芯。来源：[source_images.csv](source_images.csv) 行数及不同 `image_id` 数。

## 分组、历史使用与封存边界

患者 ID 全部为空，见 [cell_counts.csv](cell_counts.csv) 的 `n_patient_ids=0`。因此退回原始切片分组：训练和内部按 WSI；外部同一 TMA 的所有组织芯归同一切片组。42 个组织芯不等于 42 位独立患者，不随机打散细胞。

[group_manifest.csv](group_manifest.csv) 记录图像、切片、分组和角色。[development_group_folds.csv](development_group_folds.csv) 提供留一训练 WSI 的开发折清单；尚未执行模型训练，论文具体训练策略待确定。

内部和外部此前已用于 GMM 与 E1–E4 评价，**不能称为从未接触过的新测试集**。`prior_evaluation_exposure` 保存此历史。后续基线将锁定官方划分，模型选择、预处理拟合、早停及阈值开发仅可使用指定开发数据。

Pan-M 未列目录、未读取图像、标签、分布或预测。跨 PhenoBIC 与 Pan-M 的重叠检查明确为 **未执行**，不宣称不存在交叉。预训练模型尚未指定，预训练数据接触史仍待核查。

## 原始尺度与 mask 语义

训练、内部、TMA1/2 的 CD8 为 float32，TMA3 为 uint8；实例 mask 为 uint32。文件数与分组见 [input_scale_by_context.csv](input_scale_by_context.csv) 的 `split/context/image_dtype/image_files/mask_dtype`。

读取和样例 NPZ 保留原 dtype/数值范围，没有逐细胞 min–max、JPEG 转换、改变位深或重新分割。PNG 仅作线性显示，不参与模型输入。模型归一化等待官方实现，当前没有把 E1 的共同归一化强加给论文模型。

全量元数据与原冻结 `RUN_CONFIG.json` 均为 `expanded_nucleus`，边缘只能解释为扩张核的近似边界，不能认定为真实膜。语义继承原项目保存的来源说明；本轮访问所引用 PMC 页面遇到验证页面，未独立重新核实原论文。像素尺寸和独立组织 mask 仍未确认。

## 已执行的验证及限度

[sample_alignment.csv](sample_alignment.csv) 覆盖训练、内部、TMA1/2/3，每处 3 个内部完整且有标签的实例，共 **15** 个，固定 seed=0。没有按预测正确性或分数筛选；包括 13 个 neg、2 个 pos。

`mask_id_present/label_join_verified/pixel_count_matches/raw_mean_matches` 全为 True。均值从目标实例有效原始像素重算，匹配既有无标签强度统计。5 张边界叠加图已目视检查，位于 `samples/input_alignment_00.png` 至 `04.png`。这是输入对齐证据，不是模型性能或生物学膜定位验证。

[contract_test_results.csv](contract_test_results.csv) 的 **14** 项接口测试和 [synthetic_cli_checks.csv](synthetic_cli_checks.csv) 的 **7** 项合成数据文件读写检查全部通过。后者合成分数不是真实细胞预测，更不是论文指标。

[input_file_checks.csv](input_file_checks.csv) 中 **532** 个既有文件的当前大小和修改时间匹配历史清单。`historical_sha256` 为原有哈希，本轮没有重新全读大图计算 SHA256。三份 ZIP 的大小、签名、中央目录可读取，见 [archives.csv](archives.csv)，未重复全 ZIP CRC。不能把这些检查表述成新的全量密码学/逐像素核验。

官方划分间的历史精确文件哈希未发现重复，见 [cross_split_exact_image_duplicates.csv](cross_split_exact_image_duplicates.csv)。这不排除同患者、不同裁剪或不同编码的重叠。内部 FOV 历史重叠核查另存 [within_WSI_overlap_checks_historical.csv](within_WSI_overlap_checks_historical.csv)；读取继续复用原上下文重建逻辑。

## 代码与指标约定

- `baseline_contract.py`：四键验证、严格一对一连接、键哈希、禁止元数据/真值/GMM 为主模型特征的校验、指标计算。
- `preflight.py`：全量输入/标签审计，复用既有 `io_adapter.py`，输出清单和真实样例。
- `evaluate_predictions.py`：要求先提供冻结模型说明；论文、代码、权重或阈值未明确时拒绝评估。预测输入仅允许四键和 `score`，标签来自权威清单。
- `test_contract.py`、`verify_evaluation_cli.py`：接口及合成数据端到端检查。

评估器输出每细胞 Parquet，可选 CSV，以及每个官方 split、切片和图像单位的 TP/FP/FN/TN、precision、recall、F1、AP、ROC-AUC、真实/预测阳性比例及绝对误差。

不能评分的行保留；有标签集合中有缺失分数时，全体性能置为未定义并报告覆盖率，不把成功子集冒充全体。未知/含糊标签不计入二分类指标，更不映射为阴性。

无阳性切片：precision/recall/F1 使用与既有项目一致的 `zero_division=0`，AP/ROC-AUC 为 NaN；无阴性时 ROC-AUC 也为 NaN。保存 `no_true_positives/no_positive_predictions` 便于解释。

## 环境、资源与版本

服务器检测到 Quadro RTX 4000、8192 MiB 显存；PyTorch 1.10.0+cu113、torchvision 0.11.1+cu113，CUDA 可用。完整信息见 `ENVIRONMENT.json`。本轮未安装或升级依赖，目标论文是否兼容此环境仍待检查。

项目不是 Git 仓库，commit 不能虚构；新增代码 SHA256 见 [code_hashes_final.csv](code_hashes_final.csv)，复用源码见 [code_hashes.csv](code_hashes.csv)。官方仓库 commit 和实际权重 SHA256 待模型确定后记录。

本次核查耗时 **48.90 秒**，进程峰值 RSS **3215.95 MiB**，来源 `RUNTIME.json`。这些是数据准备资源记录，**不是模型训练或推理性能**；训练、推理耗时和模型峰值显存均为空。

## 保存位置与后续依赖

服务器新目录：`/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_paper_baseline_20261024`。

全量清单：`/mnt/disk3/mIHC/mif_marker_positivity_20261022/analyses/cd8_paper_baseline_20261024/artifacts/cells.parquet`。各划分的全体/有标签键表路径及哈希见 [key_manifest.csv](key_manifest.csv)。原始样例图像、mask、valid 与坐标保存于服务器 `artifacts/samples/*.npz`。

本地审阅目录：`F:/Codex/Projects/学习/cd8_paper_baseline_20261024/review`，包含 MD、CSV、配置、源码和输入示例图。

用户提供论文后，才能核查官方代码、预训练来源、通道、预处理并接入模型。真实模型训练、逐细胞预测、性能、开发集 FP/FN 以及训练/推理资源比较均尚未执行，见 [work_status.csv](work_status.csv)。**接口可运行不等于已复现论文指标。**
