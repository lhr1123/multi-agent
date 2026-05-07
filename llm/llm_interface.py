"""
LLM接口抽象类与默认实现

- `LLMInterface`: 抽象接口，定义大模型调用规范
- `MockLLMInterface`: 简单的Mock实现，方便本地测试
- `SiliconFlowLLMInterface`: 使用 OpenAI 兼容客户端进行真实调用
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import json
import re

from llm.llm_config import (
    DEFAULT_MODEL,
    TASK_ORCHESTRATOR_TEMPERATURE,
    TASK_ORCHESTRATOR_TOP_P,
    task_orchestrator_llm_model,
)


class LLMInterface(ABC):
    """LLM接口抽象基类"""
    
    @abstractmethod
    def decompose_task(self, task_title: str, task_description: str,
                      task_priority: str) -> Dict[str, Any]:
        """
        调用LLM进行任务分解
        
        Args:
            task_title: 任务标题
            task_description: 任务描述
            task_priority: 任务优先级 (low/medium/high/critical)
            
        Returns:
            Dict[str, Any]: LLM返回的JSON格式分解结果，格式如下：
            {
                "sub_tasks": [
                    {
                        "title": "子任务标题",
                        "description": "子任务详细描述",
                        "priority": "high/medium/low",
                        "estimated_time": 2.0,  # 小时，可选
                        "required_skills": ["技能1", "技能2"],  # 可选
                        "dependencies": [],  # 依赖的其他子任务索引（从0开始），可选
                        "metadata": {}  # 额外元数据，可选
                    },
                    ...
                ],
                "decomposition_strategy": "分解策略说明",
                "reasoning": "分解思路和理由"  # 可选
            }
            
        Raises:
            Exception: LLM调用失败时抛出异常
        """
        raise NotImplementedError
    
    @abstractmethod
    def extract_skills(self, text: str) -> List[str]:
        """
        从文本中提取所需技能（可选方法，如果LLM不支持可以返回空列表）
        
        Args:
            text: 待分析的文本
            
        Returns:
            List[str]: 提取出的技能列表
        """
        raise NotImplementedError


class MockLLMInterface(LLMInterface):
    """
    Mock LLM接口实现 - 用于测试和开发
    返回示例数据，不进行实际的LLM调用
    """
    
    def decompose_task(self, task_title: str, task_description: str,
                      task_priority: str) -> Dict[str, Any]:
        """Mock实现，返回示例分解结果"""
        return {
            "sub_tasks": [
                {
                    "title": "任务规划",
                    "description": f"规划任务：{task_title}",
                    "priority": "high",
                    "estimated_time": 2.0,
                    "required_skills": ["规划"],
                    "dependencies": [],
                    "metadata": {}
                },
                {
                    "title": "任务执行",
                    "description": task_description,
                    "priority": "high",
                    "estimated_time": 4.0,
                    "required_skills": [],
                    "dependencies": [0],
                    "metadata": {}
                },
                {
                    "title": "任务审查",
                    "description": "审查任务完成情况",
                    "priority": "medium",
                    "estimated_time": 1.0,
                    "required_skills": ["审查"],
                    "dependencies": [1],
                    "metadata": {}
                }
            ],
            "decomposition_strategy": "Mock LLM分解策略",
            "reasoning": "这是一个Mock实现，请替换为真实的LLM接口"
        }
    
    def extract_skills(self, text: str) -> List[str]:
        """Mock实现，返回空列表"""
        return []


class SiliconFlowLLMInterface(LLMInterface):
    """
    使用 OpenAI 兼容客户端的真实LLM接口实现（SiliconFlow / vLLM 兼容）
    
    默认复用任务总控层配置好的客户端；也可以显式传入其他客户端，
    例如单模型 baseline 的独立 endpoint。
    """
    
    def __init__(self, client=None, model_name: str = DEFAULT_MODEL):
        """
        Args:
            client: 已初始化的 OpenAI 兼容客户端
            model_name: 模型名称（例如 SiliconFlow 上的 Qwen/Qwen2.5-32B-Instruct）
        """
        self.client = client or task_orchestrator_llm_model
        self.model_name = model_name or DEFAULT_MODEL
    
    def decompose_task(self, task_title: str, task_description: str,
                      task_priority: str) -> Dict[str, Any]:
        """
        调用真实LLM进行任务分解
        """
        prompt = self._build_decomposition_prompt(
            task_title, task_description, task_priority
        )
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个多智能体系统中的任务分解专家。"},
                {"role": "user", "content": prompt},
            ],
            temperature=TASK_ORCHESTRATOR_TEMPERATURE,
            top_p=TASK_ORCHESTRATOR_TOP_P,
        )
        
        content = response.choices[0].message.content
        
        try:
            llm_output = self._safe_json_loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM 返回内容不是合法 JSON，请检查 prompt 或模型输出。原始内容：{content}"
            ) from e
        
        return self._validate_response(llm_output)
    
    def extract_skills(self, text: str) -> List[str]:
        """
        可选：如果你希望也用LLM来提取技能，可以在这里实现。
        当前简单返回空列表，留给你以后扩展。
        """
        return []
    
    def _build_decomposition_prompt(self, title: str, description: str,
                                   priority: str) -> str:
        """构建任务分解prompt"""
        prompt = f"""请将以下任务分解为多个子任务。使用中文回答

任务标题：{title}
任务描述：{description}
任务优先级：{priority}

请按照以下JSON格式返回分解结果：
{{
    "sub_tasks": [
        {{
            "title": "子任务标题",
            "description": "子任务详细描述",
            "priority": "high/medium/low",
            "estimated_time": 2.0,
            "required_skills": ["技能1", "技能2"],
            "dependencies": [],
            "metadata": {{}}
        }}
    ],
    "decomposition_strategy": "分解策略说明",
    "reasoning": "分解思路和理由"
}}

要求：
1. 子任务应该具体、可执行
2. dependencies字段使用子任务在数组中的索引（从0开始）
3. 合理设置优先级和预计时间
4. 识别所需的技能
5. 确保依赖关系合理（无循环依赖）

请直接返回JSON，不要包含其他文字。"""
        return prompt

    def _safe_json_loads(self, content: str) -> Dict[str, Any]:
        """
        尽可能从模型输出中解析出 JSON 对象。

        常见情况：模型会把 JSON 包在 ```json ... ``` 代码块里，或者在 JSON 前后夹带说明文字。
        """
        if content is None:
            raise json.JSONDecodeError("Empty content", "", 0)

        text = content.strip()

        # 1) 去掉 Markdown fenced code block（```json ... ``` 或 ``` ... ```）
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```"):
                # 找到末尾的 ```
                end_idx = None
                for i in range(len(lines) - 1, 0, -1):
                    if lines[i].strip() == "```":
                        end_idx = i
                        break
                if end_idx is not None and end_idx > 0:
                    text = "\n".join(lines[1:end_idx]).strip()

        # 2) 直接尝试解析（理想情况：纯 JSON）
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 3) 从文本中提取第一个 JSON 对象（从第一个 { 到最后一个 }）
        #    这对“前后有解释文字”的输出很有效，但假设输出里只有一个 JSON 对象。
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        return json.loads(m.group(0))
    
    def _validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """验证和规范化LLM返回结果"""
        if "sub_tasks" not in response:
            raise ValueError("LLM返回结果缺少sub_tasks字段")
        if not isinstance(response["sub_tasks"], list):
            raise ValueError("sub_tasks必须是列表类型")
        if len(response["sub_tasks"]) == 0:
            raise ValueError("sub_tasks不能为空")
        
        for i, sub_task in enumerate(response["sub_tasks"]):
            if "title" not in sub_task:
                raise ValueError(f"子任务 {i} 缺少title字段")
            if "description" not in sub_task:
                raise ValueError(f"子任务 {i} 缺少description字段")
            
            sub_task.setdefault("priority", "medium")
            sub_task.setdefault("required_skills", [])
            sub_task.setdefault("dependencies", [])
            sub_task.setdefault("metadata", {})
        
        response.setdefault("decomposition_strategy", "基于LLM语义理解的任务分解")
        return response
