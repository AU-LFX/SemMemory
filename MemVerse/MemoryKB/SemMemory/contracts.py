from __future__ import annotations

from typing import Any


SEMANTIC_DIMENSION_KEYS = [
    "entity_semantics",
    "spatial_semantics",
    "temporal_semantics",
    "rule_semantics",
]


def ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_episode_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ["episode_id", "instruction"]:
        if not payload.get(key):
            errors.append(f"missing required field: {key}")
    if "step_records" in payload and not isinstance(payload["step_records"], list):
        errors.append("step_records must be a list")
    if "history" in payload and not isinstance(payload["history"], list):
        errors.append("history must be a list")
    return errors


def empty_experience_graph(episode_id: str, errors: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": f"graph:{episode_id}",
        "type": "semantic_experience_graph",
        "episode_id": episode_id,
        "nodes": [],
        "edges": [],
        "metadata": {
            "parser": "llm_v1",
            "semmemory_layer": "experience_graph",
            "errors": errors or [],
        },
    }


def empty_raw_case(payload: dict[str, Any]) -> dict[str, Any]:
    episode_id = payload.get("episode_id", "unknown_episode")
    return {
        "id": f"case:{episode_id}",
        "type": "episodic_case",
        "episode_id": episode_id,
        "instruction": payload.get("instruction", ""),
        "task_type": ensure_dict(payload.get("metadata")).get("task_type", "unknown"),
        "result": ensure_dict(payload.get("result")),
        "ordered_steps": ensure_list(payload.get("history")),
        "key_lessons": [],
        "failure_modes": [],
        "summary_for_retrieval": f"ALFRED episode {episode_id}: {payload.get('instruction', '')}",
    }


def normalize_node(node: dict[str, Any]) -> dict[str, Any]:
    dims = ensure_dict(node.get("semantic_dimensions"))
    normalized_dims = {key: ensure_dict(dims.get(key)) for key in SEMANTIC_DIMENSION_KEYS}
    normalized = dict(node)
    normalized["id"] = str(normalized.get("id") or normalized.get("label") or "unknown_node")
    normalized["node_type"] = str(normalized.get("node_type") or "unknown")
    normalized["label"] = str(normalized.get("label") or normalized["id"])
    normalized["semantic_dimensions"] = normalized_dims
    normalized["provenance"] = ensure_dict(normalized.get("provenance"))
    return normalized


def validate_experience_graph(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(graph, dict):
        return ["graph must be a dict"]
    if not graph.get("episode_id"):
        errors.append("graph missing episode_id")
    if not isinstance(graph.get("nodes"), list):
        errors.append("graph nodes must be a list")
    if not isinstance(graph.get("edges"), list):
        errors.append("graph edges must be a list")

    for idx, node in enumerate(ensure_list(graph.get("nodes"))):
        if not isinstance(node, dict):
            errors.append(f"node {idx} must be a dict")
            continue
        dims = ensure_dict(node.get("semantic_dimensions"))
        for key in SEMANTIC_DIMENSION_KEYS:
            if key not in dims:
                errors.append(f"node {node.get('id', idx)} missing semantic_dimensions.{key}")
    return errors


def normalize_graph(graph: dict[str, Any], episode_id: str) -> dict[str, Any]:
    normalized = dict(graph or {})
    normalized["id"] = normalized.get("id") or f"graph:{episode_id}"
    normalized["type"] = normalized.get("type") or "semantic_experience_graph"
    normalized["episode_id"] = normalized.get("episode_id") or episode_id
    normalized["nodes"] = [normalize_node(n) for n in ensure_list(normalized.get("nodes")) if isinstance(n, dict)]
    normalized["edges"] = [e for e in ensure_list(normalized.get("edges")) if isinstance(e, dict)]
    metadata = ensure_dict(normalized.get("metadata"))
    metadata.setdefault("parser", "llm_v1")
    metadata.setdefault("semmemory_layer", "experience_graph")
    normalized["metadata"] = metadata
    return normalized


def validate_consolidation_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["consolidation report must be a dict"]
    for key in ["ops", "schemas", "hypotheses", "corrections", "invalidated"]:
        if key in report and not isinstance(report[key], list):
            errors.append(f"{key} must be a list")
    return errors


def normalize_consolidation_report(report: dict[str, Any], episode_id: str) -> dict[str, Any]:
    normalized = dict(report or {})
    normalized["episode_id"] = normalized.get("episode_id") or episode_id
    for key in ["ops", "schemas", "hypotheses", "corrections", "invalidated"]:
        normalized[key] = ensure_list(normalized.get(key))
    normalized["debug"] = ensure_dict(normalized.get("debug"))
    return normalized
