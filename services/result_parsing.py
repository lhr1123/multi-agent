"""Reusable parsing helpers for LLM and workflow responses."""

import json
import re
from typing import Any, Dict


def strip_code_fence(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return stripped


def safe_json_loads_from_text(content: str) -> Dict[str, Any]:
    text = strip_code_fence(content)
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            return {}


def extract_workflow_step_result(workflow_result: Dict[str, Any]) -> Dict[str, Any]:
    for _, step_result in workflow_result.get("results", {}).items():
        return step_result
    return {}
