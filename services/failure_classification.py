"""Classify retry failures without using benchmark labels."""

import re

from services.result_parsing import safe_json_loads_from_text


_CONTEXT_PATTERNS = (
    r"missing (?:the )?(?:question|options?|context|problem)",
    r"(?:question|options?|context|problem).{0,30}(?:not provided|not given|unavailable)",
    r"without.{0,30}(?:question|options?|context|problem)",
    r"(?:缺少|未提供|没有)(?:完整)?(?:题目|问题|选项|上下文|信息)",
    r"无法(?:作答|回答|判断).{0,30}(?:题目|选项|上下文|信息)",
)


def classify_failure(step, workflow, action_id=""):
    """Return a controller-facing failure kind for one failed workflow."""
    kind = str(workflow.get("failure_kind", "") or "")
    if kind in {"transport", "budget", "deadline"}:
        return kind

    structured = step.get("structured_response") or {}
    text = "\n".join(str(value or "") for value in (
        step.get("error"), step.get("response"), structured.get("result_text"),
        structured.get("final_answer"), workflow.get("error"),
    ))
    lowered = text.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in _CONTEXT_PATTERNS):
        return "context"
    if any(marker in lowered for marker in (
        "requires json", "not valid json", "不是合法 json", "missing required params",
        "缺少title字段", "缺少description字段",
    )):
        return "format"
    response = str(step.get("response", "") or "").strip()
    if response and not safe_json_loads_from_text(response) and structured.get("status") == "partial":
        return "format"
    if action_id in {"execute_python", "read_file", "write_file", "delete_file", "list_directory",
                     "visit_website", "extract_content", "arxiv_search", "bing_search"}:
        return "tool"
    return "capability"
