from __future__ import annotations

import asyncio
import json
from typing import Any

from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag import QueryParam


def _state_text(payload: dict[str, Any]) -> str:
    state = payload.get("current_state_summary", {}) or {}
    recent = payload.get("recent_feedback", []) or []
    return "\n".join(
        [
            f"instruction: {payload.get('instruction', '')}",
            f"phase: {payload.get('phase', 'unknown')}",
            f"step: {payload.get('step', 'unknown')}",
            f"latest_perception: {state.get('latest_perception', '')}",
            f"visible_objects: {state.get('visible_objects', [])}",
            f"holding: {state.get('holding', None)}",
            f"current_status: {state.get('current_status', {})}",
            f"active_entities: {state.get('active_entities', {})}",
            f"active_relationships: {state.get('active_relationships', [])}",
            f"recent_feedback: {recent}",
            "need: applicable SemMemory schemas, prior cases, object knowledge, failure rules, warnings, missing checks",
        ]
    )


def _compact_context(sem_context: str, epi_context: str, phase: str) -> str:
    parts = ["### Long-Term SemMemory", f"Current phase: {phase or 'unknown'}"]
    if sem_context and "fail_response" not in sem_context.lower():
        parts.extend(["", "Semantic memory:", sem_context.strip()])
    if epi_context and "fail_response" not in epi_context.lower():
        parts.extend(["", "Relevant prior cases:", epi_context.strip()])
    if len(parts) == 2:
        parts.append("\nNo relevant long-term SemMemory was retrieved.")
    return "\n".join(parts)


class SemanticRetriever:
    def __init__(self, episodic_rag: Any, semantic_rag: Any):
        self.episodic_rag = episodic_rag
        self.semantic_rag = semantic_rag

    async def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = _state_text(payload)
        top_k = int(payload.get("top_k") or 8)
        param = QueryParam(mode="hybrid", only_need_context=True, top_k=top_k)
        sem_context = ""
        epi_context = ""
        errors: list[str] = []

        async def _query(rag: Any, label: str) -> str:
            if rag is None:
                return ""
            try:
                return await rag.aquery(query, param=param)
            except Exception as exc:
                errors.append(f"{label}_retrieve_failed: {exc}")
                return ""

        sem_context, epi_context = await asyncio.gather(
            _query(self.semantic_rag, "mem_sem"),
            _query(self.episodic_rag, "mem_epi"),
        )
        long_term_context = _compact_context(sem_context, epi_context, payload.get("phase", "unknown"))
        return {
            "status": "ok",
            "long_term_context": long_term_context,
            "structured_long_term_memory": {
                "relevant_experiences": [],
                "applicable_schemas": [],
                "hypotheses": [],
                "corrections": [],
                "warnings": [],
                "missing_checks": [],
            },
            "debug": {
                "query": query,
                "sources": ["mem_sem", "mem_epi"],
                "errors": errors,
                "raw_context_lengths": {
                    "mem_sem": len(sem_context or ""),
                    "mem_epi": len(epi_context or ""),
                },
            },
        }
