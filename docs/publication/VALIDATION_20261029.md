# 2026-10-29 发布核验

- `tools/check_publication.py`：路径、文件大小、凭据模式、CSV 粒度与 Markdown 链接检查通过。
- `python -m unittest discover -s tests`：5 项发布工具检查通过。
- TabM 的 16 个发布脚本经格式化后精确 AST 一致；逐文件原始/LF SHA 与 AST SHA 见 [排版核验](tabm_readability_20261029.csv)。未重训模型或修改原指标。
- 校准汇总的 Python 标准库入口实际运行成功；8 张 CSV 与首次交付数值一致，见 [重建核验](../../analyses/cd8_calibrated_results_summary_20261029/PYTHON_REBUILD_VALIDATION.json)。
- 汇总的原始数值审计含 142 项混淆矩阵、门限、固定集合与逐单位加总检查。
- 新增 `export_tables.mjs` 已明确纳入源码放行规则；未使用 `git add -f`。

原始数据、逐细胞大表、模型权重和环境仍保留在服务器。Pan-M 未读取、未提交。本次公开整理保留原运行来源记录；当前发布源码与历史运行源码的哈希分别记录。
