# Historical execution order / 历史执行顺序

以下命令记录原运行顺序，固定服务器路径是历史记录。仓库发布未执行这些命令。迁移时请先阅读[配置与输入边界](../README.md)，使用独立目录与 `MIF_*` 配置；不要在已完成目录重跑。公开仓库不含病例图像，见[复核范围](../review/README.md)。

# 复现与完成标志

本次运行保留在服务器独立目录，模型和逐细胞预测不包含于轻量包。原项目与数据目录固定，Pan-M不访问。源码使用既有两个独立环境；新机器缺数据/原环境时不能宣称仅克隆轻量包即可复现全部数值。

```bash
PROJECT=/mnt/disk3/mIHC/mif_marker_positivity_20261022
RUN=$PROJECT/analyses/cd8_representation_targetaware_20261031_v1
PYF=$PROJECT/analyses/cd8_morphospatial_20261023/.venv/bin/python
PYN=$PROJECT/analyses/cd8_nimbus_20261024/.venv/bin/python
cd "$RUN"
# 已完成阶段先核验完成标志与输入/源码哈希；不要覆盖原运行。
# 下列为本次真实执行顺序，不是要求重新训练旧实验。
$PYF -B scripts/stage1_prepare.py --stage pilot
$PYF -B scripts/stage1_prepare.py --stage full --workers 4
$PYF -B scripts/stage1_hgb.py
$PYN -B scripts/stage2_neural.py --smoke-only
$PYN -B scripts/stage2_neural.py
$PYF -B scripts/stage1_composition.py
$PYF -B scripts/stage3_freeze_hgb_predict.py
$PYN -B scripts/stage3_neural_predict.py
$PYF -B scripts/stage3_evaluate.py
$PYN -B scripts/stage4_review.py
$PYN -B scripts/stage4_figures.py
$PYN -B scripts/stage4_report.py
$PYN -B scripts/stage5_validate_package.py
```

对已完成目录，优先做只读哈希和保存预测核验。全量准备命令在已有分块时复用分块，但会重建汇总文件；正式新重复请复制代码到新的独立目录，统一调整各脚本RUN配置及协议路径并重新冻结，不能借用原完成标志。训练数据/原始图像/旧E2及共享校准表必须使用协议中相同来源与哈希。

完成标志：data/FEATURES_COMPLETE.json；models/HGB_COMPLETE.json；models/NEURAL_COMPLETE.json；models/MODEL_SELECTION_FROZEN.json；results/COMPOSITION_COMPLETE.json；results/EVALUATION_COMPLETE.json；review/REVIEW_COMPLETE.json；reports/DELIVERY_NUMERIC_VALIDATION.json。

轻量包只含代码、配置、模型JSON元数据、汇总数值、图与有限真实病例；完整权重、连续分数和逐细胞判定表留服务器predictions/、models/及data/。中文与英文图件为PNG/SVG/PDF；SVG文字保留可编辑。病例CSV中的人工判读栏留空，等待用户填写。
