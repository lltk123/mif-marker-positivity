"""Build the experiment dictionary, run notes and artifact indexes."""

import pandas as pd
from common import *


def main():
    rows = []
    for c, formula in zip(
        I8,
        [
            "mean(v)",
            "median(v)",
            "std(v,ddof=0)",
            "p10(v)",
            "p90(v)",
            "p95(v)",
            "p75(v)-p25(v)",
            "sum(v)",
        ],
    ):
        rows.append(
            dict(
                feature=c,
                formula=formula,
                unit="normalized intensity * px"
                if "integrated" in c
                else "normalized intensity",
                valid_condition="Calibration valid; >=5 finite target pixels; no truncation, disconnected mask or invalid target pixels",
                NaN_reason="Invalid calibration or necessary whole-region support",
            )
        )
    formulas = [
        "(mean(edge)-mean(inner))/(abs(mean(edge))+abs(mean(inner))+epsilon)",
        "(sum(S_edge)/sum(S_whole))/(area(edge)/area(whole))",
        "mean(I_norm_edge > reference_median+3*1.4826*MAD)",
        "abs(sum(S_edge*exp(1j*theta)))/sum(S_edge)",
        "abs(sum(S_edge*exp(2j*theta)))/sum(S_edge)",
        "distance(signal_centroid,geometric_centroid)/sqrt(area/pi)",
        "(mean(contact)-mean(noncontact))/(abs(mean(contact))+abs(mean(noncontact))+epsilon)",
        "area(contact_edge)/area(edge)",
        "(mean(contact_edge)-mean(neighbor_band))/(abs(mean(contact_edge))+abs(mean(neighbor_band))+epsilon)",
    ]
    conditions = [
        "Both regions >=5 valid px; signal denominator >epsilon",
        "Reference >=20 px; edge and inner >=5 px; sum(S_whole)>epsilon*area",
        "Reference >=20 px; MAD>epsilon; edge >=5 px",
        "Reference >=20 px; edge >=5 px; edge corrected signal >0",
        "Reference >=20 px; edge >=5 px; edge corrected signal >0",
        "Reference >=20 px; sum(S_whole)>epsilon*area",
        "Both sides >=5 px; signal denominator >epsilon",
        "Valid geometry and nonempty edge; zero is valid when no contact",
        "Both sides >=5 px; signal denominator >epsilon",
    ]
    for c, formula, condition in zip(S9, formulas, conditions):
        rows.append(
            dict(
                feature=c,
                formula=formula,
                unit="dimensionless",
                valid_condition="Valid whole intensity input; " + condition,
                NaN_reason="One or more required conditions absent; see cell QC and support counts",
            )
        )
    rows.append(
        dict(
            feature="GMM_score",
            formula="Original SCIMAP rescale_score: unsupervised 3-component gate and piecewise MinMaxScaler",
            unit="GMM calibrated score, not positive probability",
            valid_condition="Audited source and exact canonical-key join",
            NaN_reason="Invalid raw mean or unavailable mapping",
        )
    )
    pd.DataFrame(rows).to_csv(A / "FEATURE_DICTIONARY.csv", index=False)
    (A / "FEATURE_DICTIONARY.md").write_text(
        """# 特征定义与适用范围

逐项公式、单位、有效条件、NaN 原因见 [FEATURE_DICTIONARY.csv](FEATURE_DICTIONARY.csv)。E1 使用 8 项归一化整体强度；E2 增加固定 S9；E3 增加原 GMM 校准分数；E4 使用三者。实际特征顺序冻结于 E1_E4_PROTOCOL.json、RUN_CONFIG.json 和各模型包。

原实例至少含 5 个有限像素即参与共同边界估计。L=min(各实例原始 p10)，U=max(各实例原始 p90)。每个单位所有像素共享 float64 clip((I-L)/(U-L),0,1)，不是逐细胞归一化。所有 8 项统计从裁剪后的像素重提取。标准差 ddof=0，分位数采用 NumPy 默认线性插值。支持不足、截断、断裂及校准失效保留对应 NaN 和原 ID。

掩码类型为 expanded_nucleus。edge 使用补零后内部 EDT≤max(1,0.2*maxEDT)；inner 是其余内部区域，不能自动称为细胞核。2–5 px 局部参照排除所有实例和无效支持，至少需 20 px。无组织 mask，不从 CD8 反推组织区域，保留 reference_tissue_unknown；该参照不保证是真实背景。无像素尺寸，几何单位保持 px。

S=max(I_norm-reference_median,0)。接触侧为距离其他实例≤2 px 的目标边缘；邻居侧带为位于其他实例内且距目标≤2 px 的有效像素，不是整个邻居细胞。R1/R2 只使用角向复数模，不使用相位。几何质心由实例像素计算。

epsilon=1e-8 为归一化尺度的数值稳定量。沿用旧代码数值条件：MAD≤epsilon 视为数值零；校正总信号≤epsilon*面积时不计算富集与质心。R1/R2 的边缘校正信号必须严格大于 0。k=3 的高信号像素门限是工程起点，只描述图像属性，不是细胞阳性门限。

QC、像素数、实际宽度、参照中位数/MAD、原始观测均值、标签、坐标、来源、患者/批次 ID 不进入分类器。S 部分缺失保留 NaN，HGB 原生处理，不补零、不直接赋阴性。正式分类指标保留完整集合；稳定性单独判断 input_valid、score_finite、paired_valid。

核扩张区域边缘只是近似细胞边缘，不能证明真实细胞膜定位。接触侧亮度不能单独证明信号污染，环形不完整不能直接判阴性。
""",
        encoding="utf-8",
    )
    (A / "README.md").write_text(
        """# E1–E4 与稳定性复现

本轮仅 PhenoBIC CD8；Pan-M Gold 封存且未读取。所有新文件位于本目录；原始数据、mask、ID、标签、划分和旧模型保持不变。全量 Parquet 保留服务器，审阅包包含 MD、CSV、配置、源码、字典、图件与少量示例。

## 执行入口

```bash
cd """
        + str(A)
        + """
PY="""
        + str(OLD)
        + """/.venv/bin/python
$PY scripts/test_compact.py
$PY scripts/batch_extract.py prepare
$PY scripts/batch_extract.py full --workers 8
$PY scripts/train_new.py
$PY scripts/evaluate.py
$PY scripts/stability.py --workers 6
$PY scripts/export_examples.py
$PY scripts/make_documentation.py
```

不要重跑旧实验。审计源码仅用于追溯；原 GMM 可以用 calibration/original_gmm_states.json 加 score_fixed 核对，无需正常输入重拟合。历史正常输入校验在用户最新指示前已结束。本轮新的监督拟合严格为 12 个 OOF 折加 4 个最终模型；稳定性中 192 次 GMM 重拟合属于规定的亮度扰动检查。

缓存校验数据清单、配置、代码、键表和模型哈希，不兼容时拒绝复用。输入清单还保存源文件 SHA256、大小、修改时间。完整重做必须另建运行目录并重新生成审计键表与协议，不能把另一数据版本替换到同一路径后复用缓存。

## 主要产物

- PhenoBIC_E1_E4_features.parquet：1,817,783 个实例的 17 项新特征、QC、支持像素数和来源。
- predictions/training_OOF_*.parquet：101,057 个有标签训练细胞的四组 OOF 分数、两套门限预测。
- predictions/final_all_cells_*.parquet：全部实例的七组分数/预测、原标签、QC 和两套门限。训练最终预测是重代入。
- models/E1.joblib 至 E4.joblib：最终模型、特征顺序、门限、训练来源、配置；另留 12 个 OOF 折模型支持断点复核。
- calibration/calibration_parameters.csv：48 单位 L/U、贡献 ID、原始动态范围与裁剪比例；bound_contributors.parquet 保留全部贡献 ID。
- keys/：各划分全体/有标签键、固定抽样清单和哈希。
- results/：完整预定有标签集合的分层性能、配对变化、缺失率及 QC CSV。
- curves/：完整 ROC/PR 及 threshold 数组；PR 阈值数组比端点短 1，见 curve_index.csv。
- stability/paired_cell_results.parquet：9,600 细胞×9 扰动×7 模型，含两套门限预测、输入有效性、输出有限性及配对有效性。
- stability/stability_*.csv：每单位/划分/TMA 的完整稳定性统计；perturbed_feature_QC.parquet 保留扰动后缺失原因；gmm_states/ 保存全部扰动拟合状态。
- tests/full_unit_affine_equivalence.csv：5 个代表完整单位×4 亮度扰动逐像素重算核验。

## API 与方法边界

inference.py：calibrate_instances、extract_cell、predict_features。校准必须使用完整 WSI/组织芯；内部 FOV 先重建并去重，不能用 200 个抽样细胞代替完整单位。io_adapter.py 复用原实例索引与上下文重建。gmm_interface.py 实现原配置 fit_score 与 score_fixed。

正增益/共同偏置的 L/U 和全单位原始均值使用经代表完整单位逐像素验证的数学等价加速；各完整单位 GMM 实际重新拟合。旋转共同变换 ROI 中图像、所有 ID 和有效区域，8 px halo 覆盖目标、参照、邻居及 mask 扰动所需支持；核验均值不变后才复用 GMM 分数。mask 变化冻结原 L/U 与 GMM 映射，将新均值输入原映射，保留原 MinMaxScaler clip=False 外推，不额外截断。

主结果严格 score>0.5。补充门限仅由新 E1–E4 的三张训练 WSI OOF 等权 F1 选择。旧三组在两个面板均保留原 0.5。同一 OOF 上选门限后的成绩标为门限开发拟合值，不能称门限选择的无偏泛化成绩。没有验证集调参、额外种子或校准器。

正式指标不因特征缺失删行。稳定性中输入失效即使 HGB 返回有限分数也不可评分；S 局部缺失但必要强度有效时仍评分。翻转率以双方可评分的配对为分母，同时用原抽样总数报告翻转或不可评分比例。稳定性必须结合原始分类准确性解释。
""",
        encoding="utf-8",
    )
    print("DOCUMENTATION COMPLETE")


if __name__ == "__main__":
    main()
