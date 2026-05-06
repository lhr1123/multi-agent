"""
操作定义 - 定义智能体可执行的操作
"""
from typing import Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    """操作类型枚举"""
    FILE_OPERATION = "file_operation"
    SEARCH = "search"
    WEB_ACCESS = "web_access"
    CODE_EXECUTION = "code_execution"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    REFLECTION = "reflection"
    QUESTION = "question"
    SUMMARIZE = "summarize"
    CONCLUDE = "conclude"
    MODIFY = "modify"
    TERMINATE = "terminate"


@dataclass
class Action:
    """操作定义"""
    action_id: str
    action_name: str
    action_type: ActionType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    execute_func: Callable = None


class ActionRegistry:
    """操作注册表 - 管理所有可用操作"""

    def __init__(self):
        self._actions: Dict[str, Action] = {}
        self._register_default_actions()

    def _register_default_actions(self):
        """注册默认操作"""
        file_actions = [
            Action(
                action_id="read_file",
                action_name="读取文件",
                action_type=ActionType.FILE_OPERATION,
                description="读取指定文件的内容",
                parameters={"path": "", "encoding": "utf-8"},
                required_params=["path"]
            ),
            Action(
                action_id="write_file",
                action_name="写入文件",
                action_type=ActionType.FILE_OPERATION,
                description="写入内容到指定文件",
                parameters={"path": "", "content": "", "encoding": "utf-8", "mode": "w"},
                required_params=["path", "content"]
            ),
            Action(
                action_id="list_directory",
                action_name="列出目录",
                action_type=ActionType.FILE_OPERATION,
                description="列出指定目录下的文件和子目录",
                parameters={"path": "."},
                required_params=["path"]
            ),
            Action(
                action_id="delete_file",
                action_name="删除文件",
                action_type=ActionType.FILE_OPERATION,
                description="删除指定的文件",
                parameters={"path": ""},
                required_params=["path"]
            ),
        ]
        
        search_actions = [
            Action(
                action_id="arxiv_search",
                action_name="学术搜索",
                action_type=ActionType.SEARCH,
                description="在arXiv上搜索学术论文",
                parameters={"query": "", "max_results": 10},
                required_params=["query"]
            ),
            Action(
                action_id="bing_search",
                action_name="网页搜索",
                action_type=ActionType.SEARCH,
                description="使用Bing进行网页搜索",
                parameters={"query": "", "max_results": 10},
                required_params=["query"]
            ),
        ]

        web_actions = [
            Action(
                action_id="visit_website",
                action_name="访问网站",
                action_type=ActionType.WEB_ACCESS,
                description="访问指定URL并获取页面内容",
                parameters={"url": "", "timeout": 30},
                required_params=["url"]
            ),
            Action(
                action_id="extract_content",
                action_name="提取内容",
                action_type=ActionType.WEB_ACCESS,
                description="从网页中提取指定内容",
                parameters={"url": "", "selector": ""},
                required_params=["url"]
            ),
        ]

        code_actions = [
            Action(
                action_id="execute_python",
                action_name="执行Python代码",
                action_type=ActionType.CODE_EXECUTION,
                description="执行Python代码并返回结果",
                parameters={"code": "", "timeout": 60},
                required_params=["code"]
            ),
        ]

        reasoning_actions = [
            Action(
                action_id="logical_reasoning",
                action_name="逻辑推理",
                action_type=ActionType.REASONING,
                description="进行逻辑推理和分析",
                parameters={"problem": "", "context": {}},
                required_params=["problem"]
            ),
            Action(
                action_id="critical_analysis",
                action_name="批判分析",
                action_type=ActionType.ANALYSIS,
                description="对信息进行批判性分析",
                parameters={"content": "", "criteria": []},
                required_params=["content"]
            ),
            Action(
                action_id="reflect",
                action_name="反思改进",
                action_type=ActionType.REFLECTION,
                description="反思过程并提出改进建议",
                parameters={"process": "", "result": ""},
                required_params=["process"]
            ),
            Action(
                action_id="decompose_question",
                action_name="问题分解",
                action_type=ActionType.QUESTION,
                description="将复杂问题分解为子问题",
                parameters={"question": ""},
                required_params=["question"]
            ),
            Action(
                action_id="summarize",
                action_name="总结信息",
                action_type=ActionType.SUMMARIZE,
                description="对信息进行总结和归纳",
                parameters={"content": "", "max_length": 500},
                required_params=["content"]
            ),
            Action(
                action_id="conclude",
                action_name="得出结论",
                action_type=ActionType.CONCLUDE,
                description="从信息中得出结论",
                parameters={"evidence": "", "context": {}},
                required_params=["evidence"]
            ),
            Action(
                action_id="modify_error",
                action_name="修正错误",
                action_type=ActionType.MODIFY,
                description="识别和修正错误",
                parameters={"error": "", "context": {}},
                required_params=["error"]
            ),
        ]

        terminate_actions = [
            Action(
                action_id="terminate_task",
                action_name="终止任务",
                action_type=ActionType.TERMINATE,
                description="结束任务并提取最终结果",
                parameters={"result": "", "summary": ""},
                required_params=["result"]
            ),
        ]

        all_actions = (file_actions + search_actions + web_actions + 
                      code_actions + reasoning_actions + terminate_actions)
        
        for action in all_actions:
            self._actions[action.action_id] = action

    def get_action(self, action_id: str) -> Action:
        """获取指定操作"""
        return self._actions.get(action_id)

    def get_actions_by_type(self, action_type: ActionType) -> List[Action]:
        """获取指定类型的所有操作"""
        return [a for a in self._actions.values() if a.action_type == action_type]

    def get_all_actions(self) -> Dict[str, Action]:
        """获取所有操作"""
        return self._actions

    def register_action(self, action: Action):
        """注册新操作"""
        self._actions[action.action_id] = action