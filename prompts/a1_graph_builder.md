# A1 提示词约束

输出必须是 JSON，包含：`entities`、`claims`、`evidence`。不得输出 Markdown；不得把候选当作已写入事实图谱；每个 Claim 的 `evidence_ids` 至少包含一个输入文本、KG 实体、KG 关系或 RAG 片段引用。

每条 Claim 还必须尽量给出稳定 `claim_id`、`subject_type`、`object_type`、受控的 `predicate_normalized`，以及 `qualifiers.case_id` / `document_id` / `source_id` 之一。证据不足、类型不明或关系无法归一化时，不得假造字段，应显式保留空值并让 A2 输出 `HOLD`。`evidence` 数组必须包含所有被 Claim 引用的 `evidence_ids`，使 A2 能验证引用是否真实存在。
