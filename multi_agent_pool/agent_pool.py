"""
通用智能体池 - 使用langgraph封装的智能体
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from llm.llm_config import SUB_AGENT_MODEL, sub_agent_llm_model
from question_solution.hungarian_algorithm import hungarian_maximum_assignment
from models import SubTask
from semantic_matcher import SemanticMatcher


# ======================= 智能体状态定义 =======================
class AgentState(TypedDict):
    """智能体状态 - 所有智能体共享的状态结构"""

    # 输入任务信息
    task_description: str
    task_context: Dict[str, Any]  # 任务上下文信息（如依赖任务的结果）

    # 执行结果
    result: str
    status: str  # "success" | "failure" | "processing"

    # token统计
    token_usage: Dict[
        str, int
    ]  # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}

    # 错误信息
    error: str

    # 中间步骤
    intermediate_steps: Annotated[List[Dict[str, Any]], add]


# ==================== 智能体基类 ====================


class BaseAgent(ABC):
    """智能体基类 - 包含所有智能体的抽象接口"""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        client=None,
        model_name: str = SUB_AGENT_MODEL,
    ):
        """
        args:
            agent_id: 智能体唯一ID
            agent_name: 智能体名称
            client: 模型客户端，默认使用子智能体 endpoint
            model_name: 模型名称，默认为"Pro/zai-org/GLM-4.7"
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.client = client or sub_agent_llm_model
        self.model_name = model_name
        self.graph = None
        self.checkpointer = MemorySaver()

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        返回智能体能力列表（用于匈牙利算法进行子任务匹配）

        returns:
            List[str]: 智能体能力列表，例如["文本提取", "信息抽取"]
        """
        pass

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        构建langgraph工作流

        returns:
            StateGraph: 构建好的langgraph工作流
        """
        pass

    @abstractmethod
    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        returns:
            str: 系统提示词
        """
        pass

    def execute(
        self,
        task_description: str,
        task_context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        执行任务
        args:
            task_description: 任务描述
            task_context: 任务上下文

        returns:
            执行结果字典
        """
        if self.graph is None:
            self.graph = self.build_graph()

        # 初始化状态
        initial_state: AgentState = {
            "task_description": task_description,
            "task_context": task_context or {},
            "result": "",
            "status": "processing",
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "error": "",
            "intermediate_steps": [],
        }

        # 执行图
        config = {"configurable": {"thread_id": f"{self.agent_id}_thread"}}

        try:
            final_state = None
            for state in self.graph.stream(initial_state, config):
                final_state = state
                # 获取最后一个节点的状态
                if state:
                    last_key = list(state.keys())[-1]
                    final_state = state[last_key]

            if final_state:
                return {
                    "result": final_state.get("result", ""),
                    "status": final_state.get("status", "failed"),
                    "token_usage": final_state.get("token_usage", {}),
                    "error": final_state.get("error", ""),
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                }
        except Exception as e:
            return {
                "result": "",
                "status": "failed",
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "error": str(e),
                "agent_id": self.agent_id,
                "agent_name": self.agent_name,
            }

        return {
            "result": "",
            "status": "failed",
            "token_usage": {},
            "error": "执行失败：未返回结果",
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
        }


# 封装智能体池
# ===================== 文本提取智能体 ===========================


class TextExtractionAgent(BaseAgent):
    """文本提取智能体 - 从文本中提取结构化信息"""

    def __init__(
        self,
        client=None,
        model_name: str = SUB_AGENT_MODEL,
    ):
        super().__init__(
            agent_id="text_extraction",
            agent_name="文本提取智能体",
            client=client,
            model_name=model_name,
        )

    def get_capabilities(self) -> List[str]:
        """返回文本提取智能体的能力tags"""
        return [
            "文本提取", "信息抽取", "数据提取", "文本解析", "结构化提取",
            "阅读理解", "理解", "提取", "分析", "数学信息提取", "信息提取"
        ]

    def build_graph(self) -> StateGraph:
        """构建文本提取工作流"""
        workflow = StateGraph(AgentState)

        # 添加执行节点
        workflow.add_node("extract", self._extract_node)

        # 设置入口和出口
        workflow.set_entry_point("extract")
        workflow.add_edge("extract", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的文本提取和信息抽取专家。
        你的任务是：
        1. 从给定的文本中提取关键信息
        2. 识别数字、实体、关系等结构化数据
        3. 将提取的信息以清晰、结构化的方式呈现
        4. 如果文本包含计算问题，提取出所有相关的数值和运算关系
        """

    def _extract_node(self, state: AgentState) -> AgentState:
        """执行文本提取节点"""
        prompt = f"""{self._build_system_prompt()}
        任务描述：{state['task_description']}
        """
        # 如果有上下文，添加到prompt
        if state.get("task_context"):
            prompt += "\n上下文信息：\n"
            for key, value in state["task_context"].items():
                prompt += f"- {key}: {value}\n"

        prompt += "\n请执行文本提取任务，提取所有关键信息和数值。"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
            )

            result = response.choices[0].message.content

            # 更新任务状态
            state["result"] = result
            state["status"] = "success"
            state["token_usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        except Exception as e:
            state["status"] = "failed"
            state["error"] = str(e)

        return state


# ======================== 基础运算智能体 =========================


class BasicComputationAgent(BaseAgent):
    """基础运算智能体 - 执行数学计算和基础运算"""

    def __init__(self, client=None, model_name: str = SUB_AGENT_MODEL):
        super().__init__(
            agent_id="basic_computation",
            agent_name="基础运算智能体",
            client=client,
            model_name=model_name,
        )

    def get_capabilities(self) -> List[str]:
        """返回基础运算智能体的能力tags"""
        return [
            "基础运算", "数学计算", "算数运算", "数值计算", "计算",
            "加法", "减法", "乘法", "除法", "加法运算", "算术运算",
            "数学运算", "数值运算", "算术", "运算"
        ]

    def build_graph(self) -> StateGraph:
        """构建基础运算工作流"""
        workflow = StateGraph(AgentState)

        # 添加两个节点：解析和计算
        workflow.add_node("parse", self._parse_node)
        workflow.add_node("compute", self._compute_node)

        # 添加边
        workflow.set_entry_point("parse")
        workflow.add_edge("parse", "compute")
        workflow.add_edge("compute", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的数学计算专家。
        你的任务是：
        1. 解析数学表达式和计算问题
        2. 执行精确的算数运算
        3. 确保计算结果的准确性
        """

    def _parse_node(self, state: AgentState) -> AgentState:
        """解析节点 - 提取需要计算的表达式"""
        prompt = f"""{self._build_system_prompt()}

        任务描述： {state['task_description']}
        请先解析这个计算问题，提取需要计算的数值和运算关系。
        """

        if state.get("task_context"):
            prompt += "\n上下文信息：\n"
            for key, value in state["task_context"].items():
                prompt += f"- {key}: {value}\n"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            parsed_info = response.usage.prompt_tokens  # 临时存储信息

            state["intermediate_steps"].append(
                {"step": "parse", "result": response.choices[0].message.content}
            )
            state["token_usage"]["prompt_tokens"] = response.usage.prompt_tokens

        except Exception as e:
            state["status"] = "failed"
            state["error"] = f"解析失败：{str(e)}"

        return state

    def _compute_node(self, state: AgentState) -> AgentState:
        """计算节点 - 执行实际计算"""
        parse_result = (
            state["intermediate_steps"][-1]["result"]
            if state["intermediate_steps"]
            else ""
        )

        prompt = f"""{self._build_system_prompt()}
        原始问题： {state['task_description']}
        解析结果： {parse_result}
        请基于解析结果执行计算，给出最终答案。
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            result = response.choices[0].message.content

            state["result"] = result
            state["status"] = "success"

            # 统计token使用数
            state["token_usage"]["prompt_tokens"] += response.usage.prompt_tokens
            state["token_usage"]["completion_tokens"] = response.usage.completion_tokens
            state["token_usage"]["total_tokens"] = state["token_usage"]["prompt_tokens"] + state["token_usage"]["completion_tokens"]

        except Exception as e:
            state["status"] = "failed"
            state["error"] = f"计算失败: {str(e)}"

        return state


# =========================== 逻辑推理智能体 ==============================


class LogicalReasoningAgent(BaseAgent):
    """逻辑推理智能体 - 执行逻辑推理和问题解决"""

    def __init__(
        self,
        client=None,
        model_name: str = SUB_AGENT_MODEL,
    ):
        super().__init__(
            agent_id="logical_reasoning",
            agent_name="逻辑推理智能体",
            client=client,
            model_name=model_name,
        )

    def get_capabilities(self) -> List[str]:
        """返回逻辑推理智能体的能力"""
        return ["逻辑", "问题解决", "推理", "逻辑分析", "推理判断"]

    def build_graph(self) -> StateGraph:
        """构建逻辑推理工作流"""
        workflow = StateGraph(AgentState)

        # 添加三个节点
        workflow.add_node("understand", self._understand_node)
        workflow.add_node("reason", self._reason_node)
        workflow.add_node("verify", self._verify_node)

        # 设置流程
        workflow.set_entry_point("understand")
        workflow.add_edge("understand", "reason")
        workflow.add_edge("reason", "verify")
        workflow.add_edge("verify", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个专业的逻辑推理专家。
        你的任务是：
        1. 理解问题的逻辑结构
        2. 进行多部逻辑推理
        3. 验证过程的正确性
        4. 给出清晰的推理步骤和最终结论
        """

    def _understand_node(self, state: AgentState) -> AgentState:
        """理解问题 - 分析问题"""
        prompt = f"""{self._build_system_prompt()}
        任务描述：{state['task_description']}
        请先理解这个问题的逻辑结构，识别关键信息和推理要求。
        """
        
        if state.get('task_context'):
            prompt += "\n上下文信息\n"
            for key, value in state['task_context'].items():
                prompt += f"- {key}: {value}\n"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ],
            )

            state["intermediate_steps"].append(
                {
                    "step": "understand",
                    "result": response.choices[0].message.content
                }
            )
            state["token_usage"]["prompt_tokens"] = response.usage.prompt_tokens
        
        except Exception as e:
            state["status"] = "failed"
            state["error"] = f"理解失败：{str(e)}"
        
        return state

    def _reason_node(self, state: AgentState) -> AgentState:
        """推理节点 - 执行逻辑推理"""
        understand_result = state["intermediate_steps"][-1]["result"] if state["intermediate_steps"] else ""

        prompt = f"""{self._build_system_prompt()}
        原始问题：{state['task_description']}
        理解结果：{understand_result}
        请基于理解结果进行逻辑推理，给出详细的推理步骤。
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )

            state["intermediate_steps"].append(
                {
                    "step": "reason",
                    "result": response.choices[0].message.content
                }
            )
            
            # 统计token
            state["token_usage"]["prompt_tokens"] += response.usage.prompt_tokens
            state["token_usage"]["completion_tokens"] = response.usage.completion_tokens

        except Exception as e:
            state["status"] = "failed"
            state["error"] = f"推理失败：{str(e)}"
        
        return state
    
    def _verify_node(self, state: AgentState) -> AgentState:
        """验证节点 - 验证推理结果"""
        understand_result = state["intermediate_steps"][0]["result"] if len(state["intermediate_steps"]) > 0 else ""
        reason_result = state["intermediate_steps"][-1]["result"] if state["intermediate_steps"] else ""

        prompt = f"""{self._build_system_prompt()}
        原始问题： {state['task_description']}
        理解过程： {understand_result}
        推理过程： {reason_result}
        请验证推理过程的正确性，给出最终答案和验证结论。
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                ]
            )

            result = response.choices[0].message.content

            state["result"] = result
            state["status"] = "success"

            # 累计token
            state["token_usage"]["prompt_tokens"] += response.usage.prompt_tokens
            state["token_usage"]["completion_tokens"] = response.usage.completion_tokens
            state["token_usage"]["total_tokens"] = state["token_usage"]["prompt_tokens"] + state["token_usage"]["completion_tokens"]

        except Exception as e:
            state["status"] = "failed"
            state["error"] = f"验证失败：{str(e)}"
        
        return state

# ========================== 智能体管理池 ========================

class AgentPool:
    """智能体池 - 管理所有通用智能体"""

    def __init__(self, client=None, model_name: str = SUB_AGENT_MODEL):
        """
        Args:
            client: llm客户端
            model_name:模型名称
        """
        self.client = client or sub_agent_llm_model
        self.model_name = model_name
        self.agents: Dict[str, BaseAgent] = {}
        
        self.semantic_matcher = SemanticMatcher()

        # 初始化智能体
        self._initialize_agents()
    
    def _initialize_agents(self):
        """初始化所有智能体"""
        self.agents["text_extraction"] = TextExtractionAgent(
            client = self.client,
        )

        self.agents["basic_computation"] = BasicComputationAgent(
            client = self.client
        )

        self.agents["logical_reasoning"] = LogicalReasoningAgent(
            client = self.client
        )
    
    def get_agent(self, agent_id: str) -> BaseAgent:
        """根据id获取智能体"""
        return self.agents.get(agent_id)
    
    def get_all_agent(self) -> Dict[str, BaseAgent]:
        """获取所有智能体"""
        return self.agents

    def _get_capabilities(self, agent: BaseAgent) -> List[str]:
        """安全获取能力列表（已归一化）"""
        caps = agent.get_capabilities()
        return list(self._normalize_tags(caps))

    def _build_score_matrix(
        self, subtasks: List[SubTask], agents: List[BaseAgent]
    ) -> List[List[int]]:
        """
        构建得分矩阵：
        - 行：子任务
        - 列：智能体
        - 分值：使用语义嵌入计算匹配得分（1-10）
        """
        matrix: List[List[int]] = []
        for sub in subtasks:
            req = sub.required_skills or []
            row: List[int] = []
            for agent in agents:
                caps = agent.get_capabilities()
                score = self.semantic_matcher.get_match_score(req, caps)
                row.append(score)
            matrix.append(row)
        return matrix

    def _normalize_tags(self, tags: List[str]) -> set[str]:
        """
        归一化技能/能力标签，减少 LLM 输出与内置标签不一致导致的 0 分匹配。
        """
        synonym_map = {
            "基本算术": "基础运算",
            "基本算术运算": "基础运算",
            "算术运算": "基础运算",
            "数学运算": "数学计算",
            "数值运算": "数值计算",
            "单位换算": "计算",
            "数学验证": "推理判断",
            "逻辑推理": "推理",
            "逻辑分析": "逻辑分析",
            "信息抽取": "信息抽取",
            "文本提取": "文本提取",
            "阅读理解": "文本提取",
            "数学理解": "文本解析",
            "问题分析": "信息抽取",
            "数值提取": "数据提取",
            "计算": "数学计算",
            "算术": "基础运算",
            "运算": "数值计算",
            "理解": "文本解析",
            "分析": "逻辑分析",
            "提取": "信息抽取",
            "加法": "基础运算",
            "减法": "基础运算",
            "乘法": "基础运算",
            "除法": "基础运算",
        }

        normed: set[str] = set()
        for t in tags or []:
            if not t:
                continue
            s = str(t).strip()
            if not s:
                continue
            s = synonym_map.get(s, s)
            normed.add(s)
        return normed

    def find_matching_agents(
        self, subtasks: List[SubTask], reusable: bool = True
    ) -> Dict[str, Any]:
        """
        根据所需技能查找匹配的智能体，使用匈牙利算法求解最大权匹配问题。

        Args:
            subtasks: 子任务列表（需包含 required_skills）
            reusable: 是否允许智能体可复用（一个智能体分配多个子任务）。
                      - True（默认）：对每个子任务独立选取得分最高的智能体（多对一）。
                      - False：使用匈牙利算法求最大权一对一匹配（子任务数>智能体数时会出现 unmatched）。

        Returns:
            {
              "assignments": [
                 {
                    "subtask_id": ...,
                    "subtask_title": ...,
                    "required_skills": [...],
                    "agent_id": ...,
                    "agent_name": ...,
                    "score": int
                 }, ...
              ],
              "unmatched_subtasks": [...],  # 如果智能体数量不足
              "total_score": int,
              "score_matrix": [[...], ...]  # 行：subtask，列：agent
            }
        """
        agents = list(self.agents.values())
        if not subtasks or not agents:
            return {
                "assignments": [],
                "unmatched_subtasks": [st.id for st in subtasks] if subtasks else [],
                "total_score": 0,
                "score_matrix": [],
            }

        score_matrix = self._build_score_matrix(subtasks, agents)

        # 构建分配结果
        assignments = []
        matched_subtasks = set()
        n_sub = len(subtasks)
        n_ag = len(agents)

        if reusable:
            # 可复用分配：每个子任务独立选择最优智能体（允许同一智能体重复出现）
            total_score = 0
            for i, sub in enumerate(subtasks):
                row = score_matrix[i] if i < len(score_matrix) else []
                if not row:
                    continue
                best_idx = max(range(n_ag), key=lambda j: row[j])
                agent = agents[best_idx]
                score = row[best_idx]
                assignments.append(
                    {
                        "subtask_id": sub.id,
                        "subtask_title": sub.title,
                        "required_skills": sub.required_skills,
                        "agent_id": agent.agent_id,
                        "agent_name": agent.agent_name,
                        "score": score,
                    }
                )
                matched_subtasks.add(sub.id)
                total_score += score

            unmatched_subtasks = []
        else:
            # 一对一分配：使用匈牙利算法求解最大权匹配
            assignments_list, total_score = hungarian_maximum_assignment(score_matrix)
            for i, agent_idx in enumerate(assignments_list):
                if i < n_sub and agent_idx >= 0 and agent_idx < n_ag:
                    sub = subtasks[i]
                    agent = agents[agent_idx]
                    assignments.append(
                        {
                            "subtask_id": sub.id,
                            "subtask_title": sub.title,
                            "required_skills": sub.required_skills,
                            "agent_id": agent.agent_id,
                            "agent_name": agent.agent_name,
                            "score": score_matrix[i][agent_idx],
                        }
                    )
                    matched_subtasks.add(sub.id)

            unmatched_subtasks = [st.id for st in subtasks if st.id not in matched_subtasks]

        return {
            "assignments": assignments,
            "unmatched_subtasks": unmatched_subtasks,
            "total_score": total_score,
            "score_matrix": score_matrix,
        }

    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有智能体及其能力"""
        return [
            {
                "agent_id": agent.agent_id,
                "agent_name": agent.agent_name,
                "capabilities": agent.get_capabilities(),
            }
            for agent in self.agents.values()
        ]
