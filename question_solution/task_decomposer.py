"""
任务分配层 - 基于LLM语义理解的任务分解器实现
"""
import uuid
import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from models import (
    Task, SubTask, TaskDecompositionResult,
    TaskPriority, TaskStatus
)
from llm.llm_interface import LLMInterface, MockLLMInterface


class TaskDecomposer:
    """任务分解器 - 基于LLM语义理解将用户任务拆分为子任务"""
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """
        初始化任务分解器
        
        Args:
            llm_interface: LLM接口实现，如果为None则使用MockLLMInterface（仅用于测试）
        """
        self.llm = llm_interface if llm_interface is not None else MockLLMInterface()
    
    def decompose(self, task: Task) -> TaskDecompositionResult:
        """
        分解任务为子任务
        
        Args:
            task: 用户输入的主任务
            
        Returns:
            TaskDecompositionResult: 包含子任务列表的分解结果
        """
        # 调用LLM进行任务分解
        llm_response = self.llm.decompose_task(
            task_title=task.title,
            task_description=task.description,
            task_priority=task.priority.value
        )
        
        # 解析LLM返回的结果
        sub_tasks = self._parse_llm_response(llm_response, task.id)
        
        # 计算总预计时间
        total_time = sum(st.estimated_time for st in sub_tasks if st.estimated_time)
        
        # 获取分解策略说明
        strategy = llm_response.get(
            "decomposition_strategy", 
            "基于LLM语义理解的任务分解"
        )
        
        return TaskDecompositionResult(
            original_task=task,
            sub_tasks=sub_tasks,
            decomposition_strategy=strategy,
            total_estimated_time=total_time if total_time > 0 else None
        )
    
    def _parse_llm_response(self, llm_response: Dict[str, Any], 
                           task_id_base: str) -> List[SubTask]:
        """
        解析LLM返回的JSON结果，转换为SubTask对象列表
        
        Args:
            llm_response: LLM返回的分解结果字典
            task_id_base: 任务ID基础前缀，用于生成子任务ID
            
        Returns:
            List[SubTask]: 子任务对象列表
        """
        sub_tasks = []
        sub_tasks_data = llm_response.get("sub_tasks", [])
        
        if not sub_tasks_data:
            raise ValueError("LLM返回结果中未包含sub_tasks字段或sub_tasks为空")
        
        # 先创建所有子任务对象（不包含依赖关系）
        for i, sub_task_data in enumerate(sub_tasks_data):
            # 解析优先级
            priority_str = sub_task_data.get("priority", "medium").lower()
            priority = self._parse_priority(priority_str)
            
            # 创建子任务对象
            sub_task = SubTask(
                id=f"{task_id_base}_subtask_{i}",
                title=sub_task_data.get("title", f"子任务 {i+1}"),
                description=sub_task_data.get("description", ""),
                priority=priority,
                status=TaskStatus.PENDING,
                estimated_time=sub_task_data.get("estimated_time"),
                required_skills=sub_task_data.get("required_skills", []),
                dependencies=[],  # 先设为空，后续处理
                metadata=sub_task_data.get("metadata", {})
            )
            
            # 保存依赖索引信息（LLM返回的是索引，需要转换为ID）
            dependency_indices = sub_task_data.get("dependencies", [])
            sub_task.metadata["_dependency_indices"] = dependency_indices
            
            sub_tasks.append(sub_task)
        
        # 处理依赖关系：将索引转换为实际的子任务ID
        for i, sub_task in enumerate(sub_tasks):
            dependency_indices = sub_task.metadata.get("_dependency_indices", [])
            dependency_ids = []
            
            for dep_idx in dependency_indices:
                if isinstance(dep_idx, int) and 0 <= dep_idx < len(sub_tasks):
                    dependency_ids.append(sub_tasks[dep_idx].id)
                else:
                    # 如果依赖索引无效，记录警告但不中断
                    print(f"警告: 子任务 {i} 的依赖索引 {dep_idx} 无效，已忽略")
            
            sub_task.dependencies = dependency_ids
            # 清理临时元数据
            if "_dependency_indices" in sub_task.metadata:
                del sub_task.metadata["_dependency_indices"]
        
        return sub_tasks
    
    def _parse_priority(self, priority_str: str) -> TaskPriority:
        """解析优先级字符串为TaskPriority枚举"""
        priority_map = {
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL,
        }
        return priority_map.get(priority_str.lower(), TaskPriority.MEDIUM)
    
    def create_task(self, title: str, description: str, 
                   priority: TaskPriority = TaskPriority.MEDIUM) -> Task:
        """
        创建任务对象
        
        Args:
            title: 任务标题
            description: 任务描述
            priority: 任务优先级
            
        Returns:
            Task: 任务对象
        """
        return Task(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            priority=priority,
            created_at=datetime.now().isoformat()
        )
