# 归一化与缓存修订记录

本运行使用完整校准单位的共享 `E1_outer_P1_P99` 边界。`NIMBUS_PROTOCOL.json`、`RUN_CONFIG.json` 和各阶段完成清单是当前运行状态的依据；存在代码或样例文件并不代表全量模型已经完成。

两项实现修订均发生在本运行的全量预训练推理和微调开始之前：

- 共享校准 CSV 读取改用 `float_precision="round_trip"`，避免不同读取入口造成浮点末位差异。Nimbus 的读取入口原已采用此规则；共享边界表的内容没有变化。最初尚未完成单位的准备任务被停止，旧配置及缓存保存在 `provenance/revisions/csv_round_trip_fix_v1`，记录见 `provenance/IMPLEMENTATION_FIX_001.json`。
- 共享小样验证原先可能选中零实例图像，改为选取第一个非空来源。除样例选择函数外的模块 AST、校准边界、细胞键、归一化核心与特征适配算法均保持一致。Nimbus 的 48 个完整准备单位通过明确的兼容性核验后重新关联新共享协议，未重写模型输入数组；旧元数据和证明保存在 `provenance/revisions/shared_pilot_fix_v2`、`provenance/CACHE_COMPATIBILITY_AUDIT.json`。这不是根据模型成绩调整方法。

验证文件的范围需区分：

- `tests/local_real_crop_numeric_checks.csv`：本地数学测试及已有真实开发集 crop 的像素算术测试，使用明确给定的人工边界，仅验证局部归一化运算，不替代完整单位校准。
- `tests/offline_numeric_checks.csv`：服务器运行的归一化公式、无效支持、退化边界等受控检查。
- `tests/global_preparation_pixel_checks.csv`：完整单位准备后的像素及全局缩放核验。
- `tests/pilot_reference_comparison.csv`：训练、内部验证和 TMA1/2/3 样例与固定官方权重/推理实现的数值对照。
- `results/calibration_coverage.csv`：按完整有标签分母统计校准是否有效，不从模型是否输出有限评分反推校准有效性；其文件及源哈希见 `provenance/calibration_coverage_complete.json`。

图像缓存、逐细胞预测及模型权重保存在服务器同名分析目录。审阅包仅收录代码、配置、核验记录、汇总表与图件；具体文件及 SHA256 见包内 `REVIEW_PACKAGE_CONTENTS.csv`。
