"""
任务执行器 - 使用LangGraph根据匹配结果调度通用智能体池中的子智能体
"""
from typing import TypedDict, Dict, Any, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from models import TaskDecompositionResult, SubTask, TaskStatus
from multi_agent_pool.agent_pool import AgentPool


class TaskExecutionState(TypedDict):
    """任务执行状态（LangGraph 使用的状态结构）"""

    # 原始任务和子任务
    original_task: Dict[str, Any]
    sub_tasks: List[SubTask]

    # 子任务 -> 智能体 分配结果（subtask_id -> agent_id）
    assignments: Dict[str, str]

    # 执行结果
    task_results: Dict[str, Any]          # subtask_id -> result
    task_statuses: Dict[str, str]         # subtask_id -> status
    completed_tasks: List[str]            # 已完成子任务ID

    # token 统计：subtask_id -> {prompt_tokens, completion_tokens, total_tokens}
    token_usage: Dict[str, Dict[str, int]]
    total_tokens: int

    # 错误信息：subtask_id -> error
    errors: Dict[str, str]

    # 当前正在执行的任务ID（调试用）
    current_task_id: str


class TaskGraphExecutor:
    """基于 LangGraph 的任务执行器：按照依赖顺序执行匹配好的子任务"""

    def __init__(self, agent_pool: AgentPool):
        self.agent_pool = agent_pool
        self.checkpointer = MemorySaver()
        self._graph = None

    # ==================== 图构建 ====================

    def _build_graph(self) -> StateGraph:
        """构建只有一个循环节点的执行图：每次执行一个“就绪”的子任务"""
        workflow = StateGraph(TaskExecutionState)

        workflow.add_node("execute_next", self._execute_next_node)

        # 入口
        workflow.set_entry_point("execute_next")

        # 条件边：继续执行或结束
        workflow.add_conditional_edges(
            "execute_next",
            self._should_continue,
            {
                "continue": "execute_next",
                "end": END,
            },
        )

        return workflow.compile(checkpointer=self.checkpointer)

    # ==================== 节点逻辑 ====================

    def _execute_next_node(self, state: TaskExecutionState) -> TaskExecutionState:
        """执行下一个就绪的子任务"""
        sub_tasks = state["sub_tasks"]
        assignments = state["assignments"]
        completed = set(state["completed_tasks"])

        # 构建 id -> SubTask 映射
        subtask_map: Dict[str, SubTask] = {st.id: st for st in sub_tasks}

        # 找到一个“就绪”的子任务：
        # - 还没完成
        # - 不在错误列表中
        # - 所有依赖都已完成
        next_subtask: SubTask | None = None
        for st in sub_tasks:
            if st.id in completed:
                continue
            if st.id in state["errors"]:
                continue
            deps = st.dependencies or []
            if all(dep_id in completed for dep_id in deps):
                next_subtask = st
                break

        # 如果没有就绪任务，直接返回状态
        if next_subtask is None:
            return state

        sub_id = next_subtask.id
        state["current_task_id"] = sub_id

        # 找到对应的智能体
        agent_id = assignments.get(sub_id)
        if not agent_id:
            state["errors"][sub_id] = "未找到匹配的智能体"
            state["task_statuses"][sub_id] = TaskStatus.FAILED.value
            return state

        agent = self.agent_pool.get_agent(agent_id)
        if agent is None:
            state["errors"][sub_id] = f"智能体 {agent_id} 不存在"
            state["task_statuses"][sub_id] = TaskStatus.FAILED.value
            return state

        # 构建上下文（依赖任务的结果）
        dependency_results = {
            dep_id: state["task_results"].get(dep_id, "")
            for dep_id in next_subtask.dependencies or []
        }
        context = {"dependency_results": dependency_results}

        # 调用智能体执行子任务（使用子任务描述作为输入）
        exec_result = agent.execute(
            task_description=next_subtask.description,
            task_context=context,
        )

        # 更新状态
        state["task_results"][sub_id] = exec_result.get("result", "")
        status = exec_result.get("status", "failed")
        state["task_statuses"][sub_id] = status

        token_info = exec_result.get("token_usage", {}) or {}
        # 确保字段存在且为 int
        prompt_tokens = int(token_info.get("prompt_tokens", 0) or 0)
        completion_tokens = int(token_info.get("completion_tokens", 0) or 0)
        total_tokens = int(token_info.get("total_tokens", prompt_tokens + completion_tokens) or 0)

        state["token_usage"][sub_id] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        state["total_tokens"] += total_tokens

        error_msg = exec_result.get("error")
        if error_msg:
            state["errors"][sub_id] = error_msg

        if status == TaskStatus.COMPLETED.value or status == "success":
            if sub_id not in state["completed_tasks"]:
                state["completed_tasks"].append(sub_id)
        else:
            # 标记为失败
            if sub_id not in state["completed_tasks"]:
                state["completed_tasks"].append(sub_id)

        return state

    def _should_continue(self, state: TaskExecutionState) -> str:
        """判断是否还有可执行的子任务"""
        sub_tasks = state["sub_tasks"]
        completed = set(state["completed_tasks"])
        errors = set(state["errors"].keys())

        for st in sub_tasks:
            if st.id in completed or st.id in errors:
                continue
            deps = st.dependencies or []
            if all(dep_id in completed for dep_id in deps):
                return "continue"

        return "end"

    # ==================== 对外执行接口 ====================

    def execute(
        self,
        decomposition_result: TaskDecompositionResult,
        match_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行任务分解结果

        Args:
            decomposition_result: 任务分解结果（包含子任务和依赖）
            match_result: AgentPool.find_matching_agents 的返回结果

        Returns:
            {
              "final_state": TaskExecutionState,
              "total_tokens": int,
              "task_results": {subtask_id: result},
              "errors": {subtask_id: error},
              "token_usage_by_task": {subtask_id: {...}},
            }
        """
        # 构建 subtask_id -> agent_id 映射
        assignments: Dict[str, str] = {}
        for item in match_result.get("assignments", []):
            sub_id = item["subtask_id"]
            agent_id = item["agent_id"]
            assignments[sub_id] = agent_id

        # 初始化状态
        initial_state: TaskExecutionState = {
            "original_task": decomposition_result.original_task.to_dict(),
            "sub_tasks": decomposition_result.sub_tasks,
            "assignments": assignments,
            "task_results": {},
            "task_statuses": {},
            "completed_tasks": [],
            "token_usage": {},
            "total_tokens": 0,
            "errors": {},
            "current_task_id": "",
        }

        # 构建图（懒加载）
        if self._graph is None:
            self._graph = self._build_graph()

        config = {"configurable": {"thread_id": "task_executor_thread"}}

        final_state_wrapper = None
        for step in self._graph.stream(initial_state, config):
            final_state_wrapper = step

        if final_state_wrapper:
            last_key = list(final_state_wrapper.keys())[-1]
            final_state: TaskExecutionState = final_state_wrapper[last_key]
        else:
            final_state = initial_state
        
        serializable_state = dict(final_state)
        serializable_state["sub_tasks"] = [
            st.to_dict() for st in final_state["sub_tasks"]
        ]

        return {
            # 注意：final_state 含有 SubTask 对象，无法直接 json.dump。
            # 这里返回可序列化版本，便于落盘与复现实验结果。
            "final_state": serializable_state,
            "total_tokens": final_state.get("total_tokens", 0),
            "task_results": final_state.get("task_results", {}),
            "errors": final_state.get("errors", {}),
            "token_usage_by_task": final_state.get("token_usage", {}),
        }

    # ==================== 并行执行优化 ====================

    def _get_ready_tasks(
        self,
        sub_tasks: List[SubTask],
        completed: set,
        errors: set,
    ) -> List[SubTask]:
        """获取所有就绪（无依赖且未完成）的子任务"""
        ready = []
        for st in sub_tasks:
            if st.id in completed or st.id in errors:
                continue
            deps = st.dependencies or []
            if all(dep_id in completed for dep_id in deps):
                ready.append(st)
        return ready

    def execute_parallel(
        self,
        decomposition_result: TaskDecompositionResult,
        match_result: Dict[str, Any],
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """并行执行任务 - 无依赖的子任务同时执行"""
        assignments: Dict[str, str] = {}
        for item in match_result.get("assignments", []):
            assignments[item["subtask_id"]] = item["agent_id"]

        sub_tasks = decomposition_result.sub_tasks
        subtask_map: Dict[str, SubTask] = {st.id: st for st in sub_tasks}

        completed: set = set()
        errors: set = set()
        task_results: Dict[str, Any] = {}
        task_statuses: Dict[str, str] = {}
        token_usage: Dict[str, Dict[str, int]] = {}
        total_tokens = 0
        task_errors: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                ready_tasks = self._get_ready_tasks(sub_tasks, completed, errors)
                if not ready_tasks:
                    break

                futures = {}
                for st in ready_tasks:
                    future = executor.submit(
                        self._execute_single_task,
                        st,
                        assignments,
                        subtask_map,
                        task_results,
                    )
                    futures[future] = st.id

                for future in as_completed(futures):
                    sub_id = futures[future]
                    try:
                        result = future.result()
                        task_results[sub_id] = result.get("result", "")
                        task_statuses[sub_id] = result.get("status", "failed")

                        token_info = result.get("token_usage", {}) or {}
                        prompt_tokens = int(token_info.get("prompt_tokens", 0) or 0)
                        completion_tokens = int(token_info.get("completion_tokens", 0) or 0)
                        total = prompt_tokens + completion_tokens
                        token_usage[sub_id] = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total,
                        }
                        total_tokens += total

                        if result.get("error"):
                            task_errors[sub_id] = result["error"]
                            errors.add(sub_id)

                        completed.add(sub_id)
                    except Exception as e:
                        task_errors[sub_id] = str(e)
                        errors.add(sub_id)
                        completed.add(sub_id)

        return {
            "final_state": {
                "original_task": decomposition_result.original_task.to_dict(),
                "sub_tasks": [st.to_dict() for st in sub_tasks],
                "assignments": assignments,
                "task_results": task_results,
                "task_statuses": task_statuses,
                "completed_tasks": list(completed),
                "token_usage": token_usage,
                "total_tokens": total_tokens,
                "errors": task_errors,
            },
            "total_tokens": total_tokens,
            "task_results": task_results,
            "errors": task_errors,
            "token_usage_by_task": token_usage,
        }

    def _execute_single_task(
        self,
        sub_task: SubTask,
        assignments: Dict[str, str],
        subtask_map: Dict[str, SubTask],
        completed_results: Dict[str, str],
    ) -> Dict[str, Any]:
        """执行单个子任务"""
        agent_id = assignments.get(sub_task.id)
        if not agent_id:
            return {"result": "", "status": "failed", "error": "未找到匹配的智能体"}

        agent = self.agent_pool.get_agent(agent_id)
        if agent is None:
            return {"result": "", "status": "failed", "error": f"智能体 {agent_id} 不存在"}

        dependency_results = {
            dep_id: completed_results.get(dep_id, "")
            for dep_id in sub_task.dependencies or []
        }
        context = {"dependency_results": dependency_results}

        return agent.execute(
            task_description=sub_task.description,
            task_context=context,
        )

