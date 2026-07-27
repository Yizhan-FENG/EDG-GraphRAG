"""A1：只提出图谱候选，不直接写入事实图谱。"""

from __future__ import annotations

from ..contracts import A1GraphProposal


class A1GraphBuilder:
    role_id = "a1"
    system_instruction = (
        "你是电力设备知识图谱构建专家。只输出有证据支撑的实体、关系和属性候选；"
        "每个候选必须附 evidence_ids；不得写入事实图谱。"
    )

    @staticmethod
    def empty_proposal(run_id: str, query: str, model_profile: str) -> A1GraphProposal:
        return A1GraphProposal(run_id=run_id, original_query=query, model_profile=model_profile)
