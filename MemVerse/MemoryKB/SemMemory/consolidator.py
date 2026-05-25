from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from .contracts import normalize_consolidation_report, validate_consolidation_report


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


class Consolidator:
    def __init__(self, llm_client: Any, prompt_path: str | Path | None = None, model: str | None = None):
        self.llm_client = llm_client
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.prompt_path = Path(prompt_path) if prompt_path else Path(__file__).parent / "prompts" / "schema_consolidation.txt"

    def _load_prompt(self) -> str:
        return self.prompt_path.read_text(encoding="utf-8")

    def consolidate(
        self,
        experience_graph: dict[str, Any],
        raw_case: dict[str, Any],
        existing_schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        episode_id = experience_graph.get("episode_id") or raw_case.get("episode_id") or "unknown_episode"
        system_prompt = self._load_prompt()
        user_payload = json.dumps(
            {
                "raw_case": raw_case,
                "experience_graph": experience_graph,
                "existing_schemas": existing_schemas,
            },
            ensure_ascii=False,
            indent=2,
        )
        raw_output = ""
        errors: list[str] = []

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                temperature=0,
            )
            raw_output = response.choices[0].message.content or ""
            parsed = _extract_json_object(raw_output)
            report = normalize_consolidation_report(parsed, episode_id)
        except Exception as exc:
            errors.append(f"consolidator_failed: {exc}")
            report = normalize_consolidation_report({"debug": {"errors": errors}}, episode_id)

        errors.extend(validate_consolidation_report(report))
        report.setdefault("debug", {})["validation_errors"] = errors
        report["prompt"] = system_prompt + "\n\n" + user_payload
        report["raw_output"] = raw_output
        return report
