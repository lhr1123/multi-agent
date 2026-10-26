"""Fixed-DAG multi-agent execution with static or rolling assignment."""

import json
from dataclasses import asdict

from llm.llm_config import SUB_AGENT_MODEL
from models import SchedulerConfig, TaskDecompositionResult, TaskPriority
from multi_agent_pool.workflow import WorkflowManager
from question_solution.task_decomposer import TaskDecomposer
from semantic_matcher import SemanticMatcher
from services.action_selection import select_action_and_input
from services.assignment_service import build_score_matrix
from services.decomposition_cache import DecompositionCache, plan_cache_key
from services.evidence_store import EvidenceStore
from services.execution_controller import ExecutionController
from services.failure_classification import classify_failure
from services.result_parsing import extract_workflow_step_result
from services.usage_tracker import UsageTracker, usage_scope, BudgetExceeded, DeadlineExceeded


def _validate_dag(tasks):
    by_id = {t.id: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate task IDs")
    visiting, done = set(), set()

    def visit(task_id):
        if task_id not in by_id:
            raise ValueError(f"Unknown dependency: {task_id}")
        if task_id in visiting:
            raise ValueError("Cyclic task dependencies")
        if task_id in done:
            return
        visiting.add(task_id)
        for dependency in by_id[task_id].dependencies:
            visit(dependency)
        visiting.remove(task_id)
        done.add(task_id)

    for task in tasks:
        visit(task.id)


def _feasibility(tasks, agents, config):
    unknown = set(config.capacities) - {a.agent_id for a in agents}
    if unknown:
        raise ValueError(f"Unknown capacity agent IDs: {sorted(unknown)}")
    matrix = []
    for task in tasks:
        allowed = task.metadata.get("allowed_agent_ids")
        required = set(task.metadata.get("required_capabilities", []))
        matrix.append([
            (allowed is None or agent.agent_id in allowed)
            and required.issubset(set(agent.capabilities))
            and config.capacities.get(agent.agent_id, config.max_parallel) > 0
            for agent in agents
        ])
    return matrix


def run_multi_agent_flow(
    llm, task_title, task_description, sub_agent_model_name=SUB_AGENT_MODEL,
    final_summary_instruction="", semantic_matcher=None, scheduler_config=None,
    fixed_subtasks=None,
):
    config = scheduler_config or SchedulerConfig()
    tracker = UsageTracker(config)
    evidence_store = EvidenceStore(config.evidence_max_chars)
    result = {"schema_version": 2, "final_result": "", "step_outputs": {}, "step_structured": {},
              "terminate_step": {}, "scheduler_config": asdict(config), "error": ""}
    controller = None
    with usage_scope(tracker):
        try:
            decomposer = TaskDecomposer(llm_interface=llm)
            task = decomposer.create_task(task_title, task_description, TaskPriority.HIGH)
            if fixed_subtasks is None:
                cache = DecompositionCache(config.plan_cache_path)
                cache_key = plan_cache_key(
                    task_title, task_description, getattr(llm, "model_name", ""),
                    config.max_plan_tasks, config.allow_direct_plan,
                )
                cached_plan = cache.get(cache_key)
                decomposition = decomposer.decompose(
                    task,
                    max_subtasks=config.max_plan_tasks,
                    allow_direct=config.allow_direct_plan,
                    llm_response=cached_plan,
                )
                if cached_plan is None and decomposer.last_llm_response:
                    cache.put(cache_key, decomposer.last_llm_response)
                result["plan_cache"] = {
                    "enabled": bool(config.plan_cache_path),
                    "hit": cached_plan is not None,
                    "key": cache_key,
                }
                result["plan_source"] = "planner"
            else:
                # Fixed plans are supplied by an evaluation adapter; no gold plan reaches the planner.
                if not fixed_subtasks:
                    raise ValueError("Fixed plan must contain at least one subtask")
                decomposition = TaskDecompositionResult(task, list(fixed_subtasks), "External fixed DAG")
                result["plan_cache"] = {"enabled": False, "hit": False, "key": ""}
                result["plan_source"] = "fixed"
            result["decomposition"] = decomposition.to_dict()
            _validate_dag(decomposition.sub_tasks)
            manager = WorkflowManager(model_name=sub_agent_model_name)
            agents = manager.get_tool_agents() + manager.get_reasoning_agents()
            matcher = semantic_matcher or SemanticMatcher()
            scores = build_score_matrix(decomposition.sub_tasks, agents, matcher,
                                        task_title=task_title, task_description=task_description)
            controller = ExecutionController(decomposition.sub_tasks, agents, scores, config, tracker,
                                             _feasibility(decomposition.sub_tasks, agents, config))
            result["score_matrix"] = scores

            def execute(subtask, agent, dependencies):
                local = WorkflowManager(model_name=sub_agent_model_name)
                dependency_evidence = evidence_store.dependencies(subtask.dependencies, dependencies)
                dependency_text = {
                    item["task_id"]: json.dumps(item, ensure_ascii=False)
                    for item in dependency_evidence
                }
                attempt = len(controller.state.tasks[subtask.id].attempts)
                previous_attempts = []
                for previous in controller.state.tasks[subtask.id].attempts:
                    previous_result = previous.get("result", {})
                    structured = previous_result.get("structured_response") or {}
                    previous_attempts.append({
                        "agent_id": previous.get("agent_id", ""),
                        "failure_kind": previous.get("failure_kind", ""),
                        "error": str(previous_result.get("error", ""))[:500],
                        "result_text": str(structured.get("result_text", ""))[:800],
                    })
                usage = tracker.snapshot()
                remaining_tokens = None
                if config.max_tokens:
                    remaining_tokens = max(0, config.max_tokens - usage["budget_charged_tokens"])
                task_context = {
                    "original_task": {"title": task_title, "description": task_description},
                    "constraints": task.metadata.get("constraints", []),
                    "current_work_item": {
                        "id": subtask.id,
                        "title": subtask.title,
                        "description": subtask.description,
                        "work_type": subtask.metadata.get("work_type", "solve"),
                        "required_skills": subtask.required_skills,
                    },
                    "dependency_evidence": dependency_evidence,
                    "candidate_answers": evidence_store.candidates(),
                    "previous_attempts": previous_attempts,
                    "remaining_budget": {
                        "tokens": remaining_tokens,
                        "seconds": max(0.0, tracker.remaining_seconds()),
                        "attempts": max(0, config.max_attempts - attempt),
                    },
                }
                with usage_scope(tracker, f"{subtask.id}#{attempt}|{agent.agent_id}"):
                    action, payload = select_action_and_input(
                        llm, subtask, agent, task_description, dependency_text,
                        task_context=task_context)
                    local.create_workflow(f"{subtask.id}_{len(controller.state.tasks[subtask.id].attempts)}",
                                          subtask.description)
                    local.add_step(agent.agent_id, action, payload)
                    workflow = local.execute_workflow()
                step = extract_workflow_step_result(workflow)
                outcome = {**step, "success": bool(workflow.get("success")),
                           "retryable": workflow.get("retryable", True),
                           "failure_kind": workflow.get("failure_kind", ""),
                           "action_id": action, "agent_id": agent.agent_id,
                           "workflow": workflow}
                if not outcome["success"]:
                    outcome["failure_kind"] = classify_failure(step, workflow, action)
                else:
                    evidence_store.add(subtask.id, outcome)
                return outcome

            state = controller.run(execute)
            result["stop_reason"] = state.stop_reason
            result["execution"] = controller.to_dict()
            for task_id, runtime in state.tasks.items():
                if runtime.status == "succeeded":
                    result["step_outputs"][task_id] = runtime.result.get("response", "")
                    result["step_structured"][task_id] = runtime.result.get("structured_response", {})
                    evidence_store.add(task_id, runtime.result)
            result["evidence"] = evidence_store.all()
            result["assignments"] = [
                {"task_id": task_id, "agent_id": rt.attempts[-1]["agent_id"]}
                for task_id, rt in state.tasks.items() if rt.attempts
            ]
            result["assignment_meta"] = {"mode": config.mode, "rounds": len(state.rounds)}
            if state.stop_reason not in {"completed", "completed_early"}:
                result["error"] = state.stop_reason

            terminators = manager.get_terminate_agents()
            if result["step_structured"] and terminators:
                payload = {
                    "original_task": task_description,
                    "evidence": evidence_store.all(),
                    "candidate_answers": evidence_store.candidates(),
                    "unresolved_tasks": [
                        {"id": task_id, "status": rt.status, "error": rt.result.get("error", "")}
                        for task_id, rt in state.tasks.items() if rt.status != "succeeded"
                    ],
                }
                manager.create_workflow("final_terminate", "Extract final answer")
                manager.add_step(terminators[0].agent_id, "terminate_task", {
                    "result": json.dumps(payload, ensure_ascii=False),
                    "summary": final_summary_instruction or (
                        "Extract the answer to the original task. Preserve units and exact values. "
                        "Do not treat incomplete evidence as verified. Return strict JSON with "
                        "status/final_answer/result_text/key_facts/confidence."
                    ),
                })
                with usage_scope(tracker, "final"):
                    final = manager.execute_workflow()
                result["terminate_step"] = extract_workflow_step_result(final)
                structured = result["terminate_step"].get("structured_response", {})
                result["final_result"] = str(structured.get("final_answer", "") or "")
                result["finalization_status"] = "completed" if final.get("success") and result["final_result"] else "failed"
                if result["finalization_status"] == "failed":
                    result["error"] = result["error"] or "finalization_failed"
            else:
                result["finalization_status"] = "unavailable"
                result["error"] = result["error"] or "no_final_answer"
        except (BudgetExceeded, DeadlineExceeded) as exc:
            reason = "budget_exhausted" if isinstance(exc, BudgetExceeded) else "deadline_exceeded"
            if result.get("stop_reason") not in {None, "completed", "completed_early"}:
                result["finalization_status"] = reason
            else:
                result["stop_reason"] = reason
            result["error"] = str(exc)
        except Exception as exc:
            result["stop_reason"] = "execution_error"
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if controller:
                result["execution"] = controller.to_dict()
            result["usage"] = tracker.snapshot()
            result["total_tokens"] = result["usage"]["known_total_tokens"]
    return result
