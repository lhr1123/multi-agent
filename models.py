"""
任务分配层的数据模型定义
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TaskPriority(Enum):
    """任务优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    """子任务数据模型"""
    id: str
    title: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    estimated_time: Optional[float] = None  # 预计完成时间（小时）
    required_skills: List[str] = field(default_factory=list)  # 所需技能
    dependencies: List[str] = field(default_factory=list)  # 依赖的其他子任务ID
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "estimated_time": self.estimated_time,
            "required_skills": self.required_skills,
            "dependencies": self.dependencies,
            "metadata": self.metadata
        }


@dataclass
class Task:
    """主任务数据模型"""
    id: str
    title: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "metadata": self.metadata
        }


@dataclass
class TaskDecompositionResult:
    """任务分解结果"""
    original_task: Task
    sub_tasks: List[SubTask]
    decomposition_strategy: str  # 分解策略说明
    total_estimated_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "original_task": self.original_task.to_dict(),
            "sub_tasks": [st.to_dict() for st in self.sub_tasks],
            "decomposition_strategy": self.decomposition_strategy,
            "total_estimated_time": self.total_estimated_time
        }
    
    def get_task_graph(self) -> Dict[str, List[str]]:
        """获取任务依赖图"""
        graph = {}
        for sub_task in self.sub_tasks:
            graph[sub_task.id] = sub_task.dependencies
        return graph
