"""Assignment helpers for semantic matching and Hungarian-core allocation."""

import re
from typing import Any, Dict, List, Tuple

from question_solution.hungarian_algorithm import hungarian_maximum_assignment


EXTERNAL_AGENT_IDS = {"arxivagent", "bingagent", "websiteagent"}
FILE_AGENT_IDS = {"fileagent"}
COMPUTE_AGENT_IDS = {"pythonagent"}
REASONING_AGENT_IDS = {
    "reasoningagent",
    "criticagent",
    "reflectagent",
    "questionagent",
    "summerizeagent",
    "concludeagent",
    "modifieragent",
}

EXTERNAL_NEED_PATTERNS = [
    r"\b(arxiv|paper|论文|文献|citation|reference|引用)\b",
    r"\b(search|web|website|url|http|online|internet|网页|网站|搜索|检索|新闻|最新|实时)\b",
    r"\b(download|crawl|scrape|爬取|抓取|浏览)\b",
]

LOCAL_FILE_PATTERNS = [
    r"\b(file|directory|path|read|write|csv|json|xml|txt|md|文件|目录|读取|写入|表格)\b",
    r"[A-Za-z]:\\",
    r"\b\S+\.(?:py|txt|md|json|csv|xml|yaml|yml)\b",
]

MATH_PATTERNS = [
    r"\b(calculate|compute|solve|equation|math|arithmetic|number|numeric|percent|ratio)\b",
    r"\b(计算|求解|算出|数学|方程|数字|数值|百分比|比例|总数|答案)\b",
    r"[-+]?\d+(?:\.\d+)?\s*(?:%|percent|美元|元|kg|km|m|cm|小时|天|年)?",
]

MULTIPLE_CHOICE_PATTERNS = [
    r"\bmultiple-choice\b",
    r"\boption letter\b",
    r"\boptions?:\b",
    r"\([A-J]\)",
    r"\b(A to J|A-J)\b",
    r"\b(选择题|选项|单选|多选)\b",
]


def _has_any_pattern(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def infer_routing_context(task_title: str = "", task_description: str = "") -> Dict[str, bool]:
    """Infer broad routing needs from the whole task, without benchmark-specific whitelists."""
    text = f"{task_title}\n{task_description}"
    needs_external = _has_any_pattern(text, EXTERNAL_NEED_PATTERNS)
    needs_file = _has_any_pattern(text, LOCAL_FILE_PATTERNS)
    is_math_like = _has_any_pattern(text, MATH_PATTERNS)
    is_multiple_choice = _has_any_pattern(text, MULTIPLE_CHOICE_PATTERNS)
    is_closed_book_like = (is_math_like or is_multiple_choice) and not needs_external and not needs_file
    return {
        "needs_external": needs_external,
        "needs_file": needs_file,
        "is_math_like": is_math_like,
        "is_multiple_choice": is_multiple_choice,
        "is_closed_book_like": is_closed_book_like,
    }


def _score_adjustment(agent, subtask, routing_context: Dict[str, bool]) -> int:
    """Apply soft, general routing preferences on top of embedding similarity."""
    agent_id = getattr(agent, "agent_id", "")
    subtask_text = " ".join(
        [
            str(getattr(subtask, "title", "") or ""),
            str(getattr(subtask, "description", "") or ""),
            " ".join(str(x) for x in (getattr(subtask, "required_skills", None) or [])),
        ]
    )
    subtask_needs_external = _has_any_pattern(subtask_text, EXTERNAL_NEED_PATTERNS)
    subtask_needs_file = _has_any_pattern(subtask_text, LOCAL_FILE_PATTERNS)
    subtask_is_math = _has_any_pattern(subtask_text, MATH_PATTERNS)

    adjustment = 0

    if routing_context.get("is_closed_book_like") and not subtask_needs_external and agent_id in EXTERNAL_AGENT_IDS:
        adjustment -= 6
    if routing_context.get("is_closed_book_like") and not subtask_needs_file and agent_id in FILE_AGENT_IDS:
        adjustment -= 5

    if routing_context.get("needs_external") or subtask_needs_external:
        if agent_id in EXTERNAL_AGENT_IDS:
            adjustment += 4
    if routing_context.get("needs_file") or subtask_needs_file:
        if agent_id in FILE_AGENT_IDS:
            adjustment += 4

    if routing_context.get("is_math_like") or subtask_is_math:
        if agent_id in COMPUTE_AGENT_IDS:
            adjustment += 3
        if agent_id in {"reasoningagent", "criticagent", "concludeagent", "modifieragent"}:
            adjustment += 2
        if agent_id in EXTERNAL_AGENT_IDS and not subtask_needs_external:
            adjustment -= 3

    if routing_context.get("is_multiple_choice"):
        if agent_id in {"reasoningagent", "criticagent", "concludeagent"}:
            adjustment += 2
        if agent_id in EXTERNAL_AGENT_IDS and not subtask_needs_external:
            adjustment -= 2

    return adjustment


def build_score_matrix(
    subtasks,
    agents,
    semantic_matcher,
    task_title: str = "",
    task_description: str = "",
) -> List[List[int]]:
    routing_context = infer_routing_context(task_title=task_title, task_description=task_description)
    matrix = []
    for subtask in subtasks:
        required_skills = subtask.required_skills or []
        row = []
        for agent in agents:
            capabilities = agent.capabilities or []
            score = semantic_matcher.get_match_score(required_skills, capabilities)
            score += _score_adjustment(agent, subtask, routing_context)
            score = max(0, min(10, score))
            row.append(score)
        matrix.append(row)
    return matrix


def assign_subtasks_with_hungarian_core(
    score_matrix: List[List[int]],
    n_agents: int,
    slots_per_agent: int = None,
) -> Tuple[List[int], int, Dict[str, Any]]:
    """
    Always use Hungarian as the core assignment engine.

    To allow one agent to handle multiple subtasks, expand each agent into
    multiple virtual slots and run Hungarian on the expanded matrix.
    """
    n_subtasks = len(score_matrix)
    if n_subtasks == 0 or n_agents == 0:
        return [], 0, {"slots_per_agent": 0, "expanded_agents": 0}

    if slots_per_agent is None:
        slots_per_agent = max(1, n_subtasks)
    else:
        slots_per_agent = max(1, int(slots_per_agent))

    slot_to_agent_idx: List[int] = []
    for agent_idx in range(n_agents):
        for _ in range(slots_per_agent):
            slot_to_agent_idx.append(agent_idx)

    expanded_matrix: List[List[int]] = []
    for row in score_matrix:
        expanded_row = [row[agent_idx] for agent_idx in slot_to_agent_idx]
        expanded_matrix.append(expanded_row)

    slot_assignments, _ = hungarian_maximum_assignment(expanded_matrix)

    assignments: List[int] = []
    total_score = 0
    for subtask_idx in range(n_subtasks):
        slot_idx = slot_assignments[subtask_idx] if subtask_idx < len(slot_assignments) else -1
        if slot_idx < 0 or slot_idx >= len(slot_to_agent_idx):
            assignments.append(-1)
            continue
        agent_idx = slot_to_agent_idx[slot_idx]
        assignments.append(agent_idx)
        total_score += score_matrix[subtask_idx][agent_idx]

    return assignments, total_score, {
        "slots_per_agent": slots_per_agent,
        "expanded_agents": len(slot_to_agent_idx),
    }
