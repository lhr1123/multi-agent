"""Multi-agent orchestration pipeline."""

import json
from typing import Any, Dict

from llm.llm_config import SUB_AGENT_MODEL
from models import TaskPriority
from multi_agent_pool.workflow import WorkflowManager
from question_solution.task_decomposer import TaskDecomposer
from reporters.console_reporter import print_decomposition_result, print_final_result
from semantic_matcher import SemanticMatcher
from services.action_selection import select_action_and_input
from services.assignment_service import assign_subtasks_with_hungarian_core, build_score_matrix
from services.result_parsing import extract_workflow_step_result, safe_json_loads_from_text
from utils.dependency_utils import sanitize_subtask_dependencies


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def run_multi_agent_flow(
    llm,
    task_title: str,
    task_description: str,
    sub_agent_model_name: str = SUB_AGENT_MODEL,
    final_summary_instruction: str = "",
) -> Dict[str, Any]:
    print("多智能体任务分配系统")
    print("=" * 60)

    print("\n【步骤1】任务分解")
    print("-" * 60)
    decomposer = TaskDecomposer(llm_interface=llm)
    task = decomposer.create_task(title=task_title, description=task_description, priority=TaskPriority.HIGH)
    decomposition_result = decomposer.decompose(task)
    dep_fix_stats = sanitize_subtask_dependencies(decomposition_result.sub_tasks)
    if dep_fix_stats["removed_invalid"] or dep_fix_stats["removed_cycle_edges"]:
        print(
            "Dependency graph normalized: "
            f"removed_invalid={dep_fix_stats['removed_invalid']}, "
            f"removed_cycle_edges={dep_fix_stats['removed_cycle_edges']}"
        )
    print_decomposition_result(decomposition_result)

    print("\n【步骤2】Embedding + 匈牙利分配")
    print("-" * 60)
    workflow_manager = WorkflowManager(model_name=sub_agent_model_name)
    tool_agents = workflow_manager.get_tool_agents()
    reasoning_agents = workflow_manager.get_reasoning_agents()
    all_agents = tool_agents + reasoning_agents
    print(f"可用智能体数量: {len(all_agents)}")
    for agent in all_agents:
        print(f"  - {agent.agent_id}: {agent.agent_name} ({agent.category.value})")

    semantic_matcher = SemanticMatcher()
    score_matrix = build_score_matrix(decomposition_result.sub_tasks, all_agents, semantic_matcher)
    print("\n得分矩阵:")
    for i, row in enumerate(score_matrix):
        print(f"  {decomposition_result.sub_tasks[i].title[:20]}... -> {row}")

    assignments, total_score, assignment_meta = assign_subtasks_with_hungarian_core(
        score_matrix=score_matrix,
        n_agents=len(all_agents),
    )
    print(
        f"Hungarian-core assignment config: "
        f"slots_per_agent={assignment_meta.get('slots_per_agent', 0)}, "
        f"expanded_agents={assignment_meta.get('expanded_agents', 0)}"
    )
    print(f"\n匹配结果 (总分: {total_score}):")

    matched_by_id: Dict[str, Dict[str, Any]] = {}
    for i, subtask in enumerate(decomposition_result.sub_tasks):
        agent_idx = assignments[i] if i < len(assignments) else -1
        if agent_idx < 0 or agent_idx >= len(all_agents):
            if all_agents:
                retry_assignments, _, _ = assign_subtasks_with_hungarian_core(
                    score_matrix=[score_matrix[i]],
                    n_agents=len(all_agents),
                    slots_per_agent=1,
                )
                agent_idx = retry_assignments[0] if retry_assignments else -1
                if agent_idx < 0 or agent_idx >= len(all_agents):
                    print(f"  Subtask[{i}] {subtask.title} -> Hungarian fallback failed")
                    continue
                print(f"  子任务[{i}] {subtask.title} -> 未分配，回退到最高分智能体 {all_agents[agent_idx].agent_id}")
            else:
                print(f"  子任务[{i}] {subtask.title} -> 无可用智能体")
                continue
        agent = all_agents[agent_idx]
        print(f"  子任务[{i}] {subtask.title} -> {agent.agent_id} ({agent.agent_name})")
        matched_by_id[subtask.id] = {"subtask": subtask, "agent": agent, "score": score_matrix[i][agent_idx]}

    print("\n【步骤3】按依赖关系执行")
    print("-" * 60)

    subtask_map = {subtask.id: subtask for subtask in decomposition_result.sub_tasks}
    executed = set()
    executing = set()
    step_outputs: Dict[str, str] = {}
    step_structured: Dict[str, Dict[str, Any]] = {}
    total_tokens = 0

    def execute_step(subtask_id: str) -> None:
        nonlocal total_tokens
        if subtask_id in executed:
            return
        if subtask_id in executing:
            print(f"Warning: dependency cycle hit at {subtask_id}, skipping recursive edge.")
            return
        if subtask_id not in matched_by_id:
            print(f"  跳过未分配任务: {subtask_id}")
            executed.add(subtask_id)
            return

        executing.add(subtask_id)
        subtask = subtask_map[subtask_id]
        agent = matched_by_id[subtask_id]["agent"]

        for dependency_id in subtask.dependencies:
            execute_step(dependency_id)

        print(f"\n执行子任务: {subtask.title}")
        print(f"  分配智能体: {agent.agent_name}")
        print(f"  所需技能: {subtask.required_skills}")

        action_id, input_data = select_action_and_input(
            llm=llm,
            subtask=subtask,
            agent=agent,
            original_task_description=task_description,
            step_outputs=step_outputs,
        )

        workflow_manager.create_workflow(f"step_{subtask.id}", subtask.description)
        workflow_manager.add_step(agent.agent_id, action_id, input_data)
        workflow_result = workflow_manager.execute_workflow()

        total_tokens += workflow_result.get("total_tokens", 0)
        step_result = extract_workflow_step_result(workflow_result)
        response = _safe_text(step_result.get("response", ""))
        structured = step_result.get("structured_response")
        if not isinstance(structured, dict) or not structured:
            parsed = safe_json_loads_from_text(response)
            structured = {
                "status": "partial",
                "final_answer": str(parsed.get("final_answer", "") or ""),
                "result_text": response,
                "key_facts": parsed.get("key_facts", []) if isinstance(parsed.get("key_facts"), list) else [],
                "confidence": parsed.get("confidence", 0.5) if parsed else 0.5,
            }

        step_outputs[subtask.id] = response
        step_structured[subtask.id] = structured

        print(f"  Action: {action_id}")
        print(f"  结果: {response[:120]}...")
        print(f"  本步 token: {workflow_result.get('total_tokens', 0)}")

        executed.add(subtask_id)
        executing.remove(subtask_id)

    for subtask in decomposition_result.sub_tasks:
        execute_step(subtask.id)

    print(f"\n总 token 消耗: {total_tokens}")

    print("\n【步骤4】结果汇总")
    print("-" * 60)
    all_dependency_ids = set()
    for subtask in decomposition_result.sub_tasks:
        all_dependency_ids.update(subtask.dependencies)
    sink_task_ids = [subtask.id for subtask in decomposition_result.sub_tasks if subtask.id not in all_dependency_ids]
    if not sink_task_ids:
        sink_task_ids = list(step_outputs.keys())

    sink_outputs = []
    for sink_task_id in sink_task_ids:
        if sink_task_id in step_outputs:
            sink_outputs.append(f"[{sink_task_id}] {step_outputs[sink_task_id]}")
    all_step_outputs = [f"[{step_id}] {text}" for step_id, text in step_outputs.items()]
    structured_facts = []
    for step_id, info in step_structured.items():
        facts = info.get("key_facts", [])
        if isinstance(facts, list):
            for fact in facts:
                if isinstance(fact, dict):
                    structured_facts.append({"step_id": step_id, **fact})
    merged_result = "\n\n".join(sink_outputs) if sink_outputs else "\n\n".join(all_step_outputs)

    final_result = merged_result
    terminate_step_result: Dict[str, Any] = {}
    terminate_agents = workflow_manager.get_terminate_agents()
    if terminate_agents and (merged_result or all_step_outputs):
        terminate_agent = terminate_agents[0]
        summary_instruction = final_summary_instruction or (
            "You are responsible for final answer extraction. "
            "For math word problems, compute and provide the final numeric answer. "
            "Return strict JSON with status/final_answer/result_text/key_facts/confidence. "
            "final_answer must contain only the final answer value (number or short text) with no extra narration."
        )
        aggregate_payload = {
            "original_task": task_description,
            "sink_outputs": sink_outputs,
            "all_step_outputs": all_step_outputs,
            "structured_facts": structured_facts,
            "step_structured": step_structured,
        }
        workflow_manager.create_workflow("final_terminate", "Aggregate final result")
        workflow_manager.add_step(
            terminate_agent.agent_id,
            "terminate_task",
            {
                "result": json.dumps(aggregate_payload, ensure_ascii=False),
                "summary": summary_instruction,
            },
        )
        terminate_result = workflow_manager.execute_workflow()
        total_tokens += terminate_result.get("total_tokens", 0)
        terminate_step_result = extract_workflow_step_result(terminate_result)
        terminate_response = _safe_text(terminate_step_result.get("response", ""))
        terminate_structured = terminate_step_result.get("structured_response", {})
        if not isinstance(terminate_structured, dict):
            terminate_structured = {}

        candidate = _safe_text(terminate_structured.get("final_answer", ""))
        if candidate:
            final_result = candidate
        elif terminate_response:
            parsed = safe_json_loads_from_text(terminate_response)
            final_result = _safe_text(parsed.get("final_answer", "") or terminate_response)

    if not final_result:
        final_result = merged_result

    print_final_result(final_result)

    return {
        "decomposition": decomposition_result,
        "assignments": assignments,
        "assignment_meta": assignment_meta,
        "score_matrix": score_matrix,
        "step_outputs": step_outputs,
        "step_structured": step_structured,
        "terminate_step": terminate_step_result,
        "final_result": final_result,
        "total_tokens": total_tokens,
    }
