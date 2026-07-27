"""A3：GraphRAG 诊断草稿生成的角色契约。"""

from __future__ import annotations

from ..contracts import A3DiagnosisDraft, EvidenceRef


class A3DiagnosisAgent:
    role_id = "a3"
    system_instruction = (
        "你是电力设备故障 GraphRAG 诊断专家。仅依据提供的 KG 和 RAG 证据生成现象、"
        "原因、措施和报告草稿；每个关键结论须绑定 evidence_ids；不得直接写最终 Word 报告。"
    )

    @staticmethod
    def empty_draft(run_id: str, model_profile: str, evidence: list[EvidenceRef]) -> A3DiagnosisDraft:
        return A3DiagnosisDraft(
            run_id=run_id,
            diagnosis_object={},
            report_sections=[],
            evidence=evidence,
            model_profile=model_profile,
        )
