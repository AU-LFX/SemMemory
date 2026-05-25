from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .contracts import (
    empty_experience_graph,
    empty_raw_case,
    normalize_graph,
    validate_experience_graph,
)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


class EventParser:
    def __init__(self, llm_client: Any, prompt_path: str | Path | None = None, model: str | None = None):
        self.llm_client = llm_client
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.prompt_path = Path(prompt_path) if prompt_path else Path(__file__).parent / "prompts" / "event_extraction.txt"

    def _load_prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")

    def _build_user_payload(self, payload: dict[str, Any]) -> str:
        compact = {
            "episode_id": payload.get("episode_id"),
            "instruction": payload.get("instruction"),
            "metadata": payload.get("metadata", {}),
            "available_actions": payload.get("available_actions", []),
            "result": payload.get("result", {}),
            "history": payload.get("history", []),
            "step_records": payload.get("step_records", []),
            "final_working_memory": payload.get("final_working_memory", {}),
        }
        return json.dumps(compact, ensure_ascii=False, indent=2)

    def parse_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        episode_id = payload.get("episode_id", "unknown_episode")
        system_prompt = self._load_prompt()
        user_content = self._build_user_payload(payload)
        raw_output = ""
        errors: list[str] = []

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )
            raw_output = response.choices[0].message.content or ""
            parsed = _extract_json_object(raw_output)
        except Exception as exc:
            errors.append(f"event_parser_failed: {exc}")
            return {
                "episode_id": episode_id,
                "raw_case": empty_raw_case(payload),
                "experience_graph": empty_experience_graph(episode_id, errors),
                "prompt": system_prompt + "\n\n" + user_content,
                "raw_output": raw_output,
                "errors": errors,
            }

        raw_case = parsed.get("raw_case") if isinstance(parsed.get("raw_case"), dict) else empty_raw_case(payload)
        graph = parsed.get("experience_graph") if isinstance(parsed.get("experience_graph"), dict) else parsed.get("graph", {})
        graph = normalize_graph(graph, episode_id)
        errors.extend(validate_experience_graph(graph))
        graph.setdefault("metadata", {}).setdefault("validation_errors", errors)

        return {
            "episode_id": episode_id,
            "raw_case": raw_case,
            "experience_graph": graph,
            "prompt": system_prompt + "\n\n" + user_content,
            "raw_output": raw_output,
            "errors": errors,
        }
