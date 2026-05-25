from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from .contracts import validate_episode_payload
from .consolidator import Consolidator
from .event_parser import EventParser
from .exporter import case_to_epi_chunks, graph_to_sem_chunks, schemas_to_sem_chunks
from .retriever import SemanticRetriever
from .store import SemanticStore


class SemMemoryService:
    def __init__(
        self,
        storage_dir: str | Path,
        episodic_rag: Any,
        semantic_rag: Any,
        llm_client: Any,
        insert_chunks_from_json: Callable[[Any, str], Awaitable[Any]],
    ):
        self.store = SemanticStore(storage_dir)
        self.episodic_rag = episodic_rag
        self.semantic_rag = semantic_rag
        self.insert_chunks_from_json = insert_chunks_from_json
        self.parser = EventParser(llm_client)
        self.consolidator = Consolidator(llm_client)
        self.retriever = SemanticRetriever(episodic_rag, semantic_rag)

    def start_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_id = payload.get("episode_id")
        errors = validate_episode_payload(payload)
        if not episode_id:
            return {"status": "error", "message": "missing episode_id", "errors": errors}
        self.store.start_episode(episode_id, payload)
        return {"status": "ok", "episode_id": episode_id, "debug": {"validation_errors": errors}}

    async def retrieve_long_term(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.retriever.retrieve(payload)

    async def end_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_id = payload.get("episode_id") or "unknown_episode"
        validation_errors = validate_episode_payload(payload)
        report: dict[str, Any] = {
            "episode_id": episode_id,
            "validation_errors": validation_errors,
            "used_generic_insert": False,
            "used_build_memory": False,
            "used_insert_chunks_from_json": True,
            "used_lightrag_ainsert": True,
            "used_lightrag_custom_kg": False,
            "semmemory_llm_calls": 0,
            "light_rag_exports": {},
        }

        self.store.save_raw_episode(episode_id, payload)

        parse_result = await asyncio.to_thread(self.parser.parse_episode, payload)
        report["semmemory_llm_calls"] += 1
        self.store.save_prompt_and_output(episode_id, "event_parser", parse_result.get("prompt", ""), parse_result.get("raw_output", ""))
        raw_case = parse_result["raw_case"]
        graph = parse_result["experience_graph"]
        self.store.save_case(raw_case)
        self.store.save_experience_graph(graph)
        report["parser_errors"] = parse_result.get("errors", [])
        report["num_graph_nodes"] = len(graph.get("nodes", []))
        report["num_graph_edges"] = len(graph.get("edges", []))

        existing_schemas = self.store.load_schemas()
        consolidation = await asyncio.to_thread(self.consolidator.consolidate, graph, raw_case, existing_schemas)
        report["semmemory_llm_calls"] += 1
        self.store.save_prompt_and_output(episode_id, "consolidator", consolidation.get("prompt", ""), consolidation.get("raw_output", ""))
        self.store.save_consolidation_report(episode_id, consolidation)

        schema_records = (
            consolidation.get("schemas", [])
            + consolidation.get("hypotheses", [])
            + consolidation.get("corrections", [])
            + consolidation.get("invalidated", [])
        )
        self.store.upsert_schemas(consolidation.get("schemas", []))
        for op in consolidation.get("ops", []):
            self.store.append_schema_update(op)
        report["schema_updates"] = consolidation.get("ops", [])
        report["consolidator_errors"] = consolidation.get("debug", {}).get("validation_errors", [])

        epi_chunks = case_to_epi_chunks(raw_case)
        sem_graph_chunks = graph_to_sem_chunks(graph)
        sem_schema_chunks = schemas_to_sem_chunks(schema_records, episode_id)

        epi_path = self.store.save_chunk_jsonl(f"{episode_id}_episodic.jsonl", epi_chunks)
        sem_graph_path = self.store.save_chunk_jsonl(f"{episode_id}_semantic_graph.jsonl", sem_graph_chunks)
        sem_schema_path = self.store.save_chunk_jsonl(f"{episode_id}_semantic_schema.jsonl", sem_schema_chunks)

        exports = {
            "mem_epi_chunks": len(epi_chunks),
            "mem_sem_graph_chunks": len(sem_graph_chunks),
            "mem_sem_schema_chunks": len(sem_schema_chunks),
        }
        try:
            await self.insert_chunks_from_json(self.episodic_rag, epi_path)
            await self.insert_chunks_from_json(self.semantic_rag, sem_graph_path)
            if sem_schema_chunks:
                await self.insert_chunks_from_json(self.semantic_rag, sem_schema_path)
        except Exception as exc:
            report.setdefault("export_errors", []).append(str(exc))

        report["light_rag_exports"] = exports
        self.store.remove_active_episode(episode_id)
        self.store.save_report(episode_id, report)

        return {
            "status": "ok",
            "episode_id": episode_id,
            "num_graph_nodes": report["num_graph_nodes"],
            "num_graph_edges": report["num_graph_edges"],
            "schema_updates": report["schema_updates"],
            "exported_to_lightrag": {
                "mem_epi_chunks": exports["mem_epi_chunks"],
                "mem_sem_chunks": exports["mem_sem_graph_chunks"] + exports["mem_sem_schema_chunks"],
            },
            "debug": {
                "semmemory_llm_calls": report["semmemory_llm_calls"],
                "used_generic_insert": False,
                "used_build_memory": False,
                "used_insert_chunks_from_json": True,
                "used_lightrag_ainsert": True,
                "validation_errors": validation_errors,
                "parser_errors": report["parser_errors"],
                "consolidator_errors": report["consolidator_errors"],
                "export_errors": report.get("export_errors", []),
            },
        }
