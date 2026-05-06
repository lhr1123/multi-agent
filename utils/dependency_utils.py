"""Dependency-graph cleanup helpers."""

from typing import Dict


def sanitize_subtask_dependencies(subtasks) -> Dict[str, int]:
    """
    Softly sanitize dependency graph:
    1) remove invalid/self/duplicate dependencies
    2) remove DFS back-edges to break cycles
    """
    task_ids = {st.id for st in subtasks}
    removed_invalid = 0

    for st in subtasks:
        clean = []
        for dep in st.dependencies or []:
            if dep == st.id:
                removed_invalid += 1
                continue
            if dep not in task_ids:
                removed_invalid += 1
                continue
            if dep in clean:
                removed_invalid += 1
                continue
            clean.append(dep)
        st.dependencies = clean

    graph = {st.id: list(st.dependencies or []) for st in subtasks}
    color: Dict[str, int] = {}
    removed_cycle_edges = set()

    def dfs(u: str):
        color[u] = 1
        for v in graph.get(u, []):
            c = color.get(v, 0)
            if c == 0:
                dfs(v)
            elif c == 1:
                removed_cycle_edges.add((u, v))
        color[u] = 2

    for st in subtasks:
        if color.get(st.id, 0) == 0:
            dfs(st.id)

    if removed_cycle_edges:
        for st in subtasks:
            st.dependencies = [d for d in (st.dependencies or []) if (st.id, d) not in removed_cycle_edges]

    return {
        "removed_invalid": removed_invalid,
        "removed_cycle_edges": len(removed_cycle_edges),
    }

