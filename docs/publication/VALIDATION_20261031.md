# 2026-10-31 发布核验

本次只整理已完成的校准、TabM／Nimbus 重训练和比较结果，没有重训、改动模型、重选门限或访问 Pan-M。

- **历史结果保留：** 原实验目录未修改；性能总览原 12 行完全保留，追加本轮 8 行（TabM A/B/C、匹配 HGB A/B/C、Nimbus 两组）。20 行使用相同外部有标签集合及固定 0.5 门限。新增排名入口可重复执行且输出一致，见 [robust_results_validation_20261031.json](robust_results_validation_20261031.json)。
- **代码可读性：** 76 个新 Python 发布文件仅经 Black 排版，完整 AST 逐一相同；原始、LF 规范化及发布后 SHA 见 [robust_readability_20261031.csv](robust_readability_20261031.csv)。模型逻辑没有改变，冻结协议仍保留原运行时哈希。
- **交付文件一致性：** 已完成交付中的非 Python 文件未修改；发布文件及 Git LF 内容哈希见 [robust_source_manifest_20261031.csv](robust_source_manifest_20261031.csv)。服务器原运行目录与原审阅包保持不变。
- **实际测试：** 5 项发布边界单元测试、18 项比较接口测试通过；后者包含缺种子、错单位键、评分失败、校准失败、缺冻结标记及旧交付格式等情况。未把这些软件检查写成重新完成数值实验。
- **发布范围：** 29 项实际 Git 忽略检查通过，覆盖数据、Pan-M、逐细胞结果、模型、环境、密码密钥、缓存和压缩包；同时验证代码、汇总表、模型元数据及案例图可发布。见 [robust_gitignore_checks_20261031.csv](robust_gitignore_checks_20261031.csv)。
- **提交前检查：** `tools/check_publication.py` 检查拟提交树的大小、凭据模式、Python 语法、CSV 粒度及 Markdown 链接。较大的风险细分配对表保留服务器，仓库发布其汇总表；不使用强制添加绕过忽略规则。

历史数值实验的独立证据另见[独立审计报告](../../analyses/cd8_robust_comparison_20261030/provenance/independent_tabm_audit/AUDIT_REPORT.md)、各子目录的完成清单和[综合报告](../../analyses/cd8_robust_comparison_20261030/ANALYSIS_REPORT.md)。提交阶段还会核对暂存区文件与上述发布内容一致，再以普通 fast-forward 推送追加历史。
