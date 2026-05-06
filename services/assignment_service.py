"""Assignment helpers for semantic matching and Hungarian-core allocation."""

from typing import Any, Dict, List, Tuple

from question_solution.hungarian_algorithm import hungarian_maximum_assignment


def build_score_matrix(subtasks, agents, semantic_matcher) -> List[List[int]]:
    matrix = []
    for subtask in subtasks:
        required_skills = subtask.required_skills or []
        row = []
        for agent in agents:
            capabilities = agent.capabilities or []
            score = semantic_matcher.get_match_score(required_skills, capabilities)
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
