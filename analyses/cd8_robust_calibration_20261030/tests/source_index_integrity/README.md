# Source-index audit scope

`source_counts.csv` 的 `n_instances` 是既有 `bboxes/<source_key>.npz` 中显式 `cell_id` 列表的长度；它表示该源索引归属的待提取实例。`empty` 和 `PASSED.json` 中 `empty_source_images` 仅表示这一列表为空。

176 个源索引中有 33 个没有待提取归属实例。内部 FOV 去重与实例归属处理后，某源可以没有归属实例，但原 mask 中仍可能包含细胞。本审计没有据此断言其原 mask 全 0，也没有删除原图、mask 或任何预定细胞。

将所有非空源索引按 `dataset_id,image_id,cell_id,marker` 合并后，得到 1,817,783 个唯一键，哈希与冻结原集合完全相同。空索引自然对应 0 个提取任务；pilot 只从有归属实例的源中选例。
