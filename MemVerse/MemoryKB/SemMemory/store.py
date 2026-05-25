from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SemanticStore:
    def __init__(self, storage_dir: str | Path):
        self.root = Path(storage_dir)
        self.active_dir = self.root / "active_episodes"
        self.raw_dir = self.root / "raw_episodes"
        self.case_dir = self.root / "cases"
        self.graph_dir = self.root / "graphs"
        self.schema_dir = self.root / "schemas"
        self.chunk_dir = self.root / "chunks"
        self.prompt_dir = self.root / "prompts"
        self.output_dir = self.root / "llm_outputs"
        self.report_dir = self.root / "reports"
        self.index_dir = self.root / "indexes"
        for directory in [
            self.active_dir,
            self.raw_dir,
            self.case_dir,
            self.graph_dir,
            self.schema_dir,
            self.chunk_dir,
            self.prompt_dir,
            self.output_dir,
            self.report_dir,
            self.index_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict[str, Any] | list[Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default

    def start_episode(self, episode_id: str, payload: dict[str, Any]) -> None:
        self._write_json(self.active_dir / f"{episode_id}.json", payload)

    def remove_active_episode(self, episode_id: str) -> None:
        path = self.active_dir / f"{episode_id}.json"
        if path.exists():
            path.unlink()

    def save_raw_episode(self, episode_id: str, payload: dict[str, Any]) -> Path:
        path = self.raw_dir / f"{episode_id}.json"
        self._write_json(path, payload)
        return path

    def save_case(self, case: dict[str, Any]) -> Path:
        episode_id = case.get("episode_id", "unknown_episode")
        path = self.case_dir / f"{episode_id}_case.json"
        self._write_json(path, case)
        return path

    def save_experience_graph(self, graph: dict[str, Any]) -> Path:
        episode_id = graph.get("episode_id", "unknown_episode")
        path = self.graph_dir / f"{episode_id}_graph.json"
        self._write_json(path, graph)
        return path

    def load_schemas(self) -> list[dict[str, Any]]:
        data = self._read_json(self.schema_dir / "schemas.json", [])
        return data if isinstance(data, list) else []

    def upsert_schemas(self, schemas: list[dict[str, Any]]) -> None:
        if not schemas:
            return
        existing = {item.get("id"): item for item in self.load_schemas() if item.get("id")}
        for schema in schemas:
            schema_id = schema.get("id")
            if schema_id:
                merged = dict(existing.get(schema_id, {}))
                merged.update(schema)
                merged["last_updated"] = datetime.now(timezone.utc).isoformat()
                existing[schema_id] = merged
        self._write_json(self.schema_dir / "schemas.json", list(existing.values()))

    def append_schema_update(self, update: dict[str, Any]) -> None:
        path = self.schema_dir / "schema_updates.jsonl"
        row = dict(update)
        row.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def save_consolidation_report(self, episode_id: str, report: dict[str, Any]) -> Path:
        path = self.schema_dir / f"{episode_id}_consolidation.json"
        self._write_json(path, report)
        return path

    def save_chunk_jsonl(self, name: str, rows: list[dict[str, Any]]) -> str:
        path = self.chunk_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return str(path)

    def save_prompt_and_output(self, episode_id: str, stage: str, prompt: str, output: str) -> None:
        safe_stage = stage.replace("/", "_")
        (self.prompt_dir / f"{episode_id}_{safe_stage}.txt").write_text(prompt, encoding="utf-8")
        (self.output_dir / f"{episode_id}_{safe_stage}.txt").write_text(output or "", encoding="utf-8")

    def save_report(self, episode_id: str, report: dict[str, Any]) -> Path:
        path = self.report_dir / f"{episode_id}_report.json"
        self._write_json(path, report)
        return path
