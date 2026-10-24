# 数据与发布范围

本仓库只发布 PhenoBIC CD8 的研究代码、冻结配置、Markdown 报告、汇总 CSV、来源记录及派生图表。图上的少量案例是报告插图，不是原始图像／mask 数据包。

## 保留在服务器的内容

- 原始 TIFF、mask、ground_truth 和数据压缩包。
- 大型逐细胞 Parquet/CSV、特征缓存、完整 ROC/PR 数组、错误细胞全量列表。
- 下载与微调权重、HGB 模型、虚拟环境、上游仓库副本和运行日志。
- 密码、SSH 密钥、访问令牌及本机配置从未纳入发布目录。
- Pan-M 全部封存；发布检查不进入其目录。本仓库中的 Pan-M 文字只说明实验范围和作者公开的训练来源。

服务器工程根目录为 `/mnt/disk3/mIHC/mif_marker_positivity_20261022`。逐文件位置见：
- [E1–E4 产物索引](../analyses/cd8_e1_e4_stability_20261023/FULL_ARTIFACT_PATHS.csv)
- [Nimbus 产物索引](../analyses/cd8_nimbus_20261024/FULL_ARTIFACT_PATHS.csv)

## Git 规则

`.gitignore` 默认忽略未知文件，仅放行约定的源码、配置、文档、汇总表和图件；随后再次排除数据、环境、权重、密钥、缓存和逐细胞文件。`tools/check_publication.py` 在提交前检查实际待提交路径、文件大小、凭据模式、表格粒度和 Markdown 本地链接。不能用 `git add -f` 绕过这些边界。

原始结果的记录、历史运行哈希与发布后源码哈希分别保留；格式化不会修改原服务器文件。发布清单见 [source_manifest.csv](publication/source_manifest.csv)，其路径和哈希用于追溯，不是数据访问凭据。
