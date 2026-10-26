"""TaskBench inventory and fixed-DAG, model-call-free assignment comparison.

TaskBench tool labels are reference data. This module never treats them as agent
assignments or as evidence that an external tool can actually be executed.
"""

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation.runtime import append_checkpoint_sample, prepare_checkpoint, sort_samples
from models import SchedulerConfig, SubTask
from services.assignment_service import build_score_matrix
from services.execution_controller import ExecutionController
from services.usage_tracker import UsageTracker


POLICIES = ("hungarian", "greedy", "random")
SCHEMA_VERSION = 1


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def _write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def load_samples(path, offset=0, limit=None):
    """TaskBench's data.json is JSONL; skip byte-equivalent duplicate samples."""
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError("Offset must be nonnegative and limit must be positive")
    selected = []
    seen = {}
    duplicate_rows = 0
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample_id = str(sample.get("id", ""))
            if not sample_id:
                raise ValueError(f"Missing TaskBench id at line {line_number}")
            if sample_id in seen:
                if sample != seen[sample_id]:
                    raise ValueError(f"Conflicting TaskBench id at line {line_number}: {sample_id}")
                duplicate_rows += 1
                continue
            seen[sample_id] = sample
            if len(seen) <= offset:
                continue
            if limit is not None and len(selected) >= limit:
                break
            selected.append(sample)
    return selected, duplicate_rows


def load_agents(persona_path):
    agents = []
    with open(persona_path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            persona = json.loads(line)
            if persona.get("category") == "terminate":
                continue
            agents.append(SimpleNamespace(
                agent_id=persona["agent_id"],
                description=persona.get("description", ""),
                capabilities=persona.get("capabilities", persona.get("capability", [])),
            ))
    return agents


def load_tool_map(path, agent_ids):
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if mapping.get("version") != 1 or not isinstance(mapping.get("tools"), dict):
        raise ValueError("Tool map must have version=1 and a tools object")
    for tool, entry in mapping["tools"].items():
        allowed = entry.get("allowed_agent_ids") if isinstance(entry, dict) else None
        if not isinstance(allowed, list) or not allowed or any(not isinstance(a, str) for a in allowed):
            raise ValueError(f"Tool {tool!r} needs a nonempty allowed_agent_ids list")
        unknown = set(allowed) - agent_ids
        if unknown:
            raise ValueError(f"Tool {tool!r} names unknown agents: {sorted(unknown)}")
        if entry.get("reviewed") is not True:
            raise ValueError(f"Tool {tool!r} must be explicitly reviewed")
    return mapping["tools"]


def load_descriptions(path):
    if not path or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(node["id"]).replace("_", " ").casefold(): node.get("desc", "")
            for node in data.get("nodes", [])}


def graph_type(tasks):
    if len(tasks) == 1:
        return "node"
    children = Counter(dep for task in tasks for dep in task.dependencies)
    edges = sum(len(task.dependencies) for task in tasks)
    if edges == len(tasks) - 1 and all(len(task.dependencies) <= 1 for task in tasks) \
            and all(children[task.id] <= 1 for task in tasks):
        return "chain"
    return "dag"


def _validate_dag(tasks):
    """Reject malformed reference graphs before they enter the controller."""
    by_id = {task.id: task for task in tasks}
    visiting, visited = set(), set()

    def visit(task_id):
        if task_id not in by_id or task_id in visiting:
            raise ValueError("invalid_dag")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dep in by_id[task_id].dependencies:
            visit(dep)
        visiting.remove(task_id)
        visited.add(task_id)

    for task in tasks:
        visit(task.id)


def make_fixed_tasks(sample, tool_map, descriptions, max_nodes):
    """Convert gold tool nodes to fixed work items without choosing an agent."""
    nodes = sample.get("task_nodes")
    links = sample.get("task_links")
    if not isinstance(nodes, list) or not nodes or not isinstance(links, list):
        raise ValueError("missing_or_invalid_graph")
    if any(not isinstance(node, dict) for node in nodes):
        raise ValueError("missing_or_invalid_graph")
    if len(nodes) > max_nodes:
        raise ValueError("too_many_nodes")
    names = [str(node.get("task", "")) for node in nodes]
    if any(not name for name in names):
        raise ValueError("missing_tool_name")
    if any(name not in tool_map for name in names):
        raise ValueError("unmapped_tool")
    task_ids = [f"{sample['id']}:node:{i}" for i in range(len(nodes))]
    indexes = {}
    for i, name in enumerate(names):
        indexes.setdefault(name, []).append(i)

    def resolve(reference):
        if isinstance(reference, int) or (isinstance(reference, str) and reference.isdecimal()):
            index = int(reference)
            if 0 <= index < len(nodes):
                return index
        if isinstance(reference, str) and reference.startswith("<node-") and reference.endswith(">"):
            index = reference[6:-1]
            if index.isdecimal() and int(index) < len(nodes):
                return int(index)
        matches = indexes.get(str(reference), [])
        if len(matches) == 1:
            return matches[0]
        raise ValueError("ambiguous_or_unknown_link")

    tasks = [SubTask(
        id=task_ids[i], title=name,
        description=descriptions.get(name.replace("_", " ").casefold(), name),
        metadata={"required": True, "tool_id": name,
                  "allowed_agent_ids": tool_map[name]["allowed_agent_ids"]},
    ) for i, name in enumerate(names)]
    for link in links:
        if not isinstance(link, dict) or "source" not in link or "target" not in link:
            raise ValueError("invalid_link")
        source, target = resolve(link["source"]), resolve(link["target"])
        if source == target or task_ids[source] in tasks[target].dependencies:
            raise ValueError("invalid_link")
        tasks[target].dependencies.append(task_ids[source])
    try:
        _validate_dag(tasks)
    except ValueError as exc:
        raise ValueError("invalid_dag") from exc
    return tasks


def run_policy(tasks, agents, scores, base_config, policy):
    # A successful local stub isolates matching and scheduling from tool/LLM outcomes.
    config = replace(base_config, mode="rolling-hungarian", policy=policy,
                     feedback=False, max_attempts=1, early_stop_optional=False,
                     max_tokens=0, plan_cache_path="")
    feasible = [[agent.agent_id in task.metadata["allowed_agent_ids"]
                 and config.capacities.get(agent.agent_id, config.max_parallel) > 0
                 for agent in agents] for task in tasks]
    controller = ExecutionController(tasks, agents, scores, config, UsageTracker(config), feasible)
    state = controller.run(lambda *_: {"success": True})
    index_by_id = {task.id: i for i, task in enumerate(tasks)}
    score_sum = 0.0
    assignments = []
    for task in tasks:
        attempts = state.tasks[task.id].attempts
        if attempts:
            agent_id = attempts[-1]["agent_id"]
            agent_index = next(i for i, agent in enumerate(agents) if agent.agent_id == agent_id)
            score_sum += scores[index_by_id[task.id]][agent_index]
            assignments.append({"task_id": task.id, "agent_id": agent_id})
    rounds = state.rounds
    return {
        "stop_reason": state.stop_reason,
        "assignment_score_sum": score_sum,
        "assigned_count": len(assignments),
        "round_count": len(rounds),
        "parallel_ready_rounds": sum(len(row["ready_ids"]) > 1 for row in rounds),
        "joint_assignment_rounds": sum(len(row["candidate_ids"]) > 1 for row in rounds),
        "capacity_conflict_rounds": sum(bool(row["candidate_capacity_conflict"]) for row in rounds),
        "deferred_task_rounds": sum(bool(row["deferred_ids"]) for row in rounds),
        "assignments": assignments,
        "rounds": rounds,
    }


def evaluate_taskbench_offline(data_path, save_path, config, tool_map_path=None,
                               tool_desc_path=None, offset=0, limit=None,
                               checkpoint_path=None, resume=False, matcher=None):
    """Inventory all selected samples; compare policies only on reviewed mappings."""
    if resume and not checkpoint_path:
        raise ValueError("--resume requires --checkpoint-path")
    if config.mode != "rolling-hungarian":
        raise ValueError("TaskBench offline comparison uses rolling-hungarian mode")
    if Path(save_path).exists() and Path(save_path).stat().st_size and not resume:
        raise ValueError("Output already exists; choose a new --save-path or use --resume")
    if config.max_parallel < 2 and tool_map_path:
        raise ValueError("TaskBench assignment comparison requires --max-parallel >= 2")
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"TaskBench data file is missing: {data_path.resolve()}. "
            "Download the selected domain's data.json from "
            "https://github.com/microsoft/JARVIS/tree/main/taskbench "
            "and place it at this path. The inventory command does not download data."
        )
    persona_path = Path(__file__).resolve().parents[1] / "multi_agent_pool" / "personas.jsonl"
    agents = load_agents(persona_path)
    agent_ids = {agent.agent_id for agent in agents}
    if set(config.capacities) - agent_ids:
        raise ValueError(f"Unknown capacity agents: {sorted(set(config.capacities) - agent_ids)}")
    if tool_desc_path is None:
        candidate = data_path.parent / "tool_desc.json"
        tool_desc_path = candidate if candidate.exists() else None
    descriptions = load_descriptions(tool_desc_path)
    tool_map = load_tool_map(tool_map_path, agent_ids) if tool_map_path else None
    samples, duplicate_rows = load_samples(data_path, offset=offset, limit=limit)
    inventory = Counter(name for sample in samples for name in
                        (str(node.get("task", "")) for node in (sample.get("task_nodes") or [])
                         if isinstance(node, dict)))
    source_types = Counter(str(sample.get("type", "unknown")) for sample in samples)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "taskbench",
        "data_sha256": _digest(data_path),
        "tool_map_sha256": _digest(tool_map_path),
        "tool_desc_sha256": _digest(tool_desc_path),
        "personas_sha256": _digest(persona_path),
        "code_sha256": hashlib.sha256(b"".join(
            path.read_bytes() for path in (
                Path(__file__), Path(__file__).resolve().parents[1] / "services" / "assignment_service.py",
                Path(__file__).resolve().parents[1] / "services" / "execution_controller.py",
                Path(__file__).resolve().parents[1] / "pipelines" / "multi_agent_pipeline.py",
                Path(__file__).resolve().parents[1] / "semantic_matcher.py",
                Path(__file__).resolve().parents[1] / "models.py",
            ))).hexdigest(),
        "offset": offset, "limit": limit,
        "scheduler": asdict(config),
        "score_source": "all-MiniLM-L6-v2" if tool_map else None,
        "mode": "fixed_dag_offline" if tool_map else "inventory",
    }
    metadata["fingerprint"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    if checkpoint_path and tool_map:
        prior, done = prepare_checkpoint(checkpoint_path, "taskbench", resume, metadata)
    else:
        prior, done = [], set()
    entries = list(prior)
    excluded = Counter()
    graph_types = Counter()
    scorer = matcher
    for sample in samples:
        sample_id = str(sample["id"])
        if sample_id in done:
            continue
        entry = {"taskbench_id": sample_id, "experiment": metadata,
                 "user_request": sample.get("user_request", "")}
        if not tool_map:
            continue
        try:
            tasks = make_fixed_tasks(sample, tool_map, descriptions, config.max_plan_tasks)
        except ValueError as exc:
            entry["exclusion_reason"] = str(exc)
            excluded[str(exc)] += 1
        else:
            if scorer is None:
                from semantic_matcher import SemanticMatcher
                scorer = SemanticMatcher()
            scores = build_score_matrix(tasks, agents, scorer,
                                        task_description=entry["user_request"])
            entry.update({"graph_type": graph_type(tasks), "tool_ids": [t.title for t in tasks],
                          "fixed_dag": [t.to_dict() for t in tasks], "score_matrix": scores,
                          "policies": {policy: run_policy(tasks, agents, scores, config, policy)
                                       for policy in POLICIES}})
            graph_types[entry["graph_type"]] += 1
        entries.append(entry)
        if checkpoint_path:
            append_checkpoint_sample(checkpoint_path, entry)
    if tool_map:
        excluded = Counter(entry.get("exclusion_reason") for entry in entries
                           if entry.get("exclusion_reason"))
        graph_types = Counter(entry.get("graph_type") for entry in entries if entry.get("graph_type"))
    accepted = [entry for entry in entries if "policies" in entry]
    policy_totals = {}
    for policy in POLICIES if tool_map else ():
        runs = [entry["policies"][policy] for entry in accepted]
        policy_totals[policy] = {
            key: sum(run[key] for run in runs)
            for key in ("assignment_score_sum", "assigned_count", "round_count",
                        "parallel_ready_rounds", "joint_assignment_rounds",
                        "capacity_conflict_rounds", "deferred_task_rounds")
        }
        total_rounds = policy_totals[policy]["round_count"]
        policy_totals[policy]["parallel_ready_round_rate"] = (
            policy_totals[policy]["parallel_ready_rounds"] / total_rounds if total_rounds else 0.0)
        policy_totals[policy]["joint_assignment_round_rate"] = (
            policy_totals[policy]["joint_assignment_rounds"] / total_rounds if total_rounds else 0.0)
        policy_totals[policy]["capacity_conflict_round_rate"] = (
            policy_totals[policy]["capacity_conflict_rounds"] / total_rounds if total_rounds else 0.0)
        policy_totals[policy]["samples_with_capacity_conflict"] = sum(
            run["capacity_conflict_rounds"] > 0 for run in runs)
    choice_samples = sum(any(len(task["metadata"]["allowed_agent_ids"]) > 1
                             for task in entry["fixed_dag"]) for entry in accepted)
    report = {
        "schema_version": SCHEMA_VERSION, "experiment": metadata,
        "summary": {"selected_samples": len(samples), "accepted_samples": len(accepted),
                    "source_duplicate_rows_skipped": duplicate_rows,
                    "samples_with_agent_choice": choice_samples,
                    "excluded_by_reason": dict(excluded), "graph_types": dict(graph_types),
                    "source_types": dict(source_types),
                    "tool_inventory": dict(inventory.most_common()),
                    "policy_totals": policy_totals,
                    "outcome_accuracy": None,
                    "note": "Offline stub success measures assignment behavior, not TaskBench execution accuracy."},
        "samples": sort_samples(entries, "taskbench") if tool_map else [],
    }
    _write_json(save_path, report)
    return report
