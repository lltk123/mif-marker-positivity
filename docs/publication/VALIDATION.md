# 首次发布核验

- 88 个历史 Python 文件与服务器原导出源码逐一比较 AST：除模块和函数说明外完全一致。见 [code_readability_audit.csv](code_readability_audit.csv)。
- 在原 Python 3.8.20 特征环境执行发布副本的单图 CLI：3 个合成实例，包含大于 uint16 的原始 ID；ID 对应、原始均值、输入位深及 mask 保持、重复输出拒绝均通过。见 [feature_cli_validation.json](feature_cli_validation.json)。
- 发布工具的 5 项独立测试通过；覆盖禁止路径、合法汇总表、凭据检测的脱敏输出、正常配置和 Markdown 链接。
- 实际 Git 忽略规则覆盖原始数据、Pan-M、权重、逐细胞结果、环境及密钥。见 [gitignore_checks.csv](gitignore_checks.csv)。
- 提交前检查完整文件集合：Python 语法、文件大小、凭据模式、CSV 粒度及本地 Markdown 链接。该检查不进入研究数据目录。
- 本次没有重新训练或重新计算模型性能。旧报告的验证记录描述原数值实验，当前发布核验描述代码与材料整理。
