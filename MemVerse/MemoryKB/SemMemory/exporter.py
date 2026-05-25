from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact(value: Any) -> str:
    if value in (None, "", [], {}):
        return "N/A"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _row(row_id: str, input_text: str, output_text: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "timestamp": _now(),
        "input_text": input_text,
        "output_text": output_text,
    }


def _edge_lines(edges: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for edge in edges:
        source = edge.get("source")
        relation = edge.get("relation")
        target = edge.get("target")
        if not source or not relation or not target:
            continue
        lines.append(
            f"relationship: {source} {relation} {target}; "
            f"relationship_keywords: {relation}, event_centric_edge; "
            f"evidence: {_compact(edge.get('evidence'))}; "
            f"confidence: {_compact(edge.get('confidence'))}"
        )
    return lines


def _support_event_ids(record: dict[str, Any]) -> list[str]:
    support = record.get("positive_support") or record.get("support") or []
    event_ids: list[str] = []
    if isinstance(support, list):
        for item in support:
            if isinstance(item, dict) and item.get("event_id"):
                event_ids.append(str(item["event_id"]))
            elif isinstance(item, str):
                event_ids.append(item)

    for event_id in record.get("support_event_ids") or []:
        event_ids.append(str(event_id))
    return event_ids


def _schema_relationship_lines(record: dict[str, Any], record_id: str) -> list[str]:
    lines: list[str] = []
    for precondition in record.get("preconditions") or []:
        lines.append(
            f"relationship: {record_id} has_precondition {precondition}; "
            "relationship_keywords: has_precondition, requires, rule_semantics"
        )
    for effect in record.get("effects") or []:
        lines.append(
            f"relationship: {record_id} has_effect {effect}; "
            "relationship_keywords: has_effect, causes, rule_semantics"
        )
    for failure_condition in record.get("failure_conditions") or []:
        lines.append(
            f"relationship: {record_id} fails_when {failure_condition}; "
            "relationship_keywords: fails_when, violates, failure_rule"
        )
    for phase in record.get("applicable_phases") or []:
        lines.append(
            f"relationship: {record_id} applies_to phase:{phase}; "
            "relationship_keywords: applies_to, phase"
        )
    for event_id in _support_event_ids(record):
        lines.append(
            f"relationship: {record_id} supported_by {event_id}; "
            "relationship_keywords: supported_by, evidence"
        )
    return lines


def case_to_epi_chunks(case: dict[str, Any]) -> list[dict[str, Any]]:
    episode_id = case.get("episode_id", "unknown_episode")
    case_id = case.get("id", f"case:{episode_id}")
    lines = [
        "[SemMemory episodic_case]",
        f"layer: raw_case",
        f"case_id: {case_id}",
        f"episode_id: {episode_id}",
        f"instruction: {case.get('instruction', '')}",
        f"task_type: {case.get('task_type', 'unknown')}",
        f"result: {_compact(case.get('result'))}",
        f"ordered_steps: {_compact(case.get('ordered_steps'))}",
        f"key_lessons: {_compact(case.get('key_lessons'))}",
        f"failure_modes: {_compact(case.get('failure_modes'))}",
        f"summary_for_retrieval: {case.get('summary_for_retrieval', '')}",
        f"relationship: {case_id} has_instruction instruction:{episode_id}; relationship_keywords: has_instruction, task_context",
        f"relationship: {case_id} has_outcome result:{episode_id}; relationship_keywords: has_outcome, task_result",
    ]
    return [_row(f"semmemory:epi:case:{episode_id}", f"episode_id={episode_id}", "\n".join(lines))]


def graph_to_sem_chunks(graph: dict[str, Any]) -> list[dict[str, Any]]:
    episode_id = graph.get("episode_id", "unknown_episode")
    rows: list[dict[str, Any]] = []
    for index, node in enumerate(graph.get("nodes", [])):
        dims = node.get("semantic_dimensions", {})
        node_id = node.get("id", f"node:{index}")
        related_edges = [
            edge for edge in graph.get("edges", [])
            if edge.get("source") == node_id or edge.get("target") == node_id
        ]
        lines = [
            "[SemMemory semantic]",
            "layer: experience_graph",
            f"episode_id: {episode_id}",
            f"graph_id: {graph.get('id', f'graph:{episode_id}')}",
            f"node_id: {node_id}",
            f"node_type: {node.get('node_type', 'unknown')}",
            f"label: {node.get('label', node_id)}",
            f"entity_semantics: {_compact(dims.get('entity_semantics'))}",
            f"spatial_semantics: {_compact(dims.get('spatial_semantics'))}",
            f"temporal_semantics: {_compact(dims.get('temporal_semantics'))}",
            f"rule_semantics: {_compact(dims.get('rule_semantics'))}",
            f"provenance: {_compact(node.get('provenance'))}",
            f"edges: {_compact(related_edges)}",
        ]
        lines.extend(_edge_lines(related_edges))
        rows.append(_row(f"semmemory:sem:graph:{episode_id}:{node_id}", f"node_id={node_id}", "\n".join(lines)))

    if not rows:
        lines = [
            "[SemMemory semantic]",
            "layer: experience_graph",
            f"episode_id: {episode_id}",
            f"graph_id: {graph.get('id', f'graph:{episode_id}')}",
            "nodes: []",
            "edges: []",
        ]
        rows.append(_row(f"semmemory:sem:graph:{episode_id}:empty", f"episode_id={episode_id}", "\n".join(lines)))
    return rows


def schemas_to_sem_chunks(records: list[dict[str, Any]], episode_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        record_id = record.get("id") or record.get("target_id") or f"schema_record:{episode_id}:{index}"
        record_type = record.get("type") or record.get("target_type") or "schema_record"
        lines = [
            "[SemMemory semantic]",
            f"layer: {record_type}",
            f"episode_id: {episode_id}",
            f"record_id: {record_id}",
            f"name: {record.get('name', record_id)}",
            f"action_type: {record.get('action_type', 'N/A')}",
            f"entity_roles: {_compact(record.get('entity_roles'))}",
            f"applicable_phases: {_compact(record.get('applicable_phases'))}",
            f"preconditions: {_compact(record.get('preconditions'))}",
            f"effects: {_compact(record.get('effects'))}",
            f"failure_conditions: {_compact(record.get('failure_conditions'))}",
            f"status: {record.get('status', 'active')}",
            f"source_operators: {_compact(record.get('source_operators') or record.get('source_operator'))}",
            f"support: {_compact(record.get('positive_support') or record.get('support_event_ids'))}",
            f"payload: {_compact(record.get('payload'))}",
            f"full_record: {_compact(record)}",
        ]
        lines.extend(_schema_relationship_lines(record, record_id))
        rows.append(_row(f"semmemory:sem:schema:{episode_id}:{record_id}", f"record_id={record_id}", "\n".join(lines)))
    return rows
