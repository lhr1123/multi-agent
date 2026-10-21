"""
任务分配层的数据模型定义
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import math


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
class SchedulerConfig:
    mode: str = "rolling-hungarian"
    policy: str = "hungarian"
    max_parallel: int = 1
    service_concurrency: int = 1
    capacities: Dict[str, int] = field(default_factory=dict)
    max_attempts: int = 2
    max_tokens: int = 0
    max_output_tokens: int = 4096
    final_reserve_tokens: int = 8192
    deadline_seconds: float = 600.0
    seed: int = 0
    feedback: bool = True
    fit_weight: float = 1.0
    reliability_weight: float = 0.2
    cost_weight: float = 0.1
    failure_weight: float = 0.5
    information_gain_weight: float = 0.15
    diversity_weight: float = 0.1
    reliability_priors: Dict[str, float] = field(default_factory=dict)
    reliability_prior_strength: float = 2.0
    max_plan_tasks: int = 3
    allow_direct_plan: bool = True
    early_stop_optional: bool = True
    optional_utility_threshold: float = 1.3
    evidence_max_chars: int = 6000
    plan_cache_path: str = ""

    def __post_init__(self):
        if self.mode not in {"static", "rolling-hungarian"}:
            raise ValueError("Unknown scheduler mode")
        if self.policy not in {"hungarian", "greedy", "random"}:
            raise ValueError("Unknown assignment policy")
        if min(self.max_parallel, self.service_concurrency, self.max_attempts,
               self.max_output_tokens, self.max_plan_tasks, self.evidence_max_chars) < 1:
            raise ValueError("Concurrency, attempts and output limit must be positive")
        if self.max_tokens < 0 or self.final_reserve_tokens < 0:
            raise ValueError("Token budgets must be nonnegative")
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("Deadline must be finite and positive")
        if any(not isinstance(v, int) or v < 0 for v in self.capacities.values()):
            raise ValueError("Agent capacities must be nonnegative integers")
        weights = (self.fit_weight, self.reliability_weight, self.cost_weight,
                   self.failure_weight, self.information_gain_weight, self.diversity_weight,
                   self.reliability_prior_strength, self.optional_utility_threshold)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Weights must be finite and nonnegative")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.reliability_priors.values()):
            raise ValueError("Reliability priors must lie in [0, 1]")


@dataclass
class TaskRuntime:
    task_id: str
    status: str = "pending"
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    result: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionState:
    tasks: Dict[str, TaskRuntime]
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    stop_reason: Optional[str] = None
    termination: Dict[str, Any] = field(default_factory=dict)


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
