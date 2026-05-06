# Architecture Overview

## Goal

这个项目实现的是一个“由大模型驱动分工、由 embedding + 匈牙利算法完成分配、由子智能体执行、由终止智能体汇总结果”的多智能体任务处理框架。

它的核心不是简单的“工具链调用”，而是：

1. 先让总控模型理解任务结构。
2. 再让分配模块选择最合适的子智能体。
3. 再让子智能体自主决定是否调用工具。
4. 最后统一提取最终答案与评测指标。

## High-Level Pipeline

```text
User Task
  -> TaskDecomposer
  -> Subtasks + dependencies + required skills
  -> SemanticMatcher
  -> score matrix
  -> Hungarian assignment
  -> WorkflowManager
  -> step execution by sub-agents
  -> terminate agent aggregation
  -> final answer / evaluation result
```

## Main Modules

### `main.py`

统一 CLI 入口，负责：

- 参数解析
- 初始化 orchestrator / baseline LLM
- 调度到 `pipelines/` 或 `evaluation/`

入口文件本身不再承载具体的分配、执行、评测实现。

### `pipelines/`

负责单次运行主流程。

- `multi_agent_pipeline.py`: 任务分解、agent 分配、依赖执行、终止汇总

### `evaluation/`

负责数据集级评测流程。

- `gsmhard_evaluator.py`: GSM-hard 样本循环、指标统计、结果落盘、图表生成
- `mmlu_pro_evaluator.py`: MMLU-Pro 样本循环、多选答案抽取、分类别统计、结果落盘

### `services/`

负责可复用的业务服务与策略层逻辑。

- `assignment_service.py`: 得分矩阵与匈牙利核心分配
- `action_selection.py`: 子任务到 action 的选择策略
- `result_parsing.py`: LLM / workflow 响应解析
- `single_llm_solver.py`: 单模型基线运行

### `reporters/`

负责面向 CLI 的展示逻辑。

- `console_reporter.py`: 任务分解展示、最终答案展示

### `llm/`

负责模型配置和接口封装。

- `llm_config.py`: 集中维护 orchestrator / sub-agent / baseline 模型名
- `llm_interface.py`: SiliconFlow/OpenAI-compatible 客户端封装

### `question_solution/`

负责任务分解与分配算法相关逻辑。

- `task_decomposer.py`: 调用 LLM 产出子任务结构
- `hungarian_algorithm.py`: 核心分配算法
- `task_executor.py`: 执行辅助逻辑

### `multi_agent_pool/`

负责智能体池、persona、动作定义和工作流执行。

- `agent_pool.py`: 智能体定义与分组
- `actions.py`: 可执行动作和工具绑定
- `workflow.py`: step 级执行框架
- `personas.jsonl`: 智能体能力画像

### `utils/`

负责和主流程正交的辅助逻辑。

- `dataset_utils.py`: 数据集读取
- `dependency_utils.py`: 依赖图清洗
- `answer_utils.py`: 结果数值提取和准确率判定
- `eval_visualization.py`: 评测图表生成
- `utils.py`: Python 代码执行等通用工具

## Why the helper split matters

重构前，`main.py` 同时承载：

- 任务流程编排
- 数据集加载
- 依赖清洗
- 最终答案提取
- 评测图表生成

这会让入口文件过长，也会让开源读者难以快速理解“主流程”和“辅助功能”的边界。

当前版本进一步把“流程级职责”也拆分到了 `pipelines/`、`evaluation/`、`services/` 和 `reporters/`，好处是：

- `main.py` 更接近真正的入口脚本，而不是巨型脚本
- 单次运行、数据集评测、策略路由可以独立演进
- 单独测试辅助函数与流程函数更容易
- 后续替换答案提取策略或可视化方案时影响面更小

## Design Notes

### 1. Hungarian algorithm stays at the core

你的系统设计要求“匈牙利算法是分配核心，不能只是可选项”。当前结构仍然保留这个原则：

- 先通过 embedding 生成得分矩阵
- 再用匈牙利算法做全局最优分配
- 子智能体执行阶段不重新改写分配结果

### 2. Sub-agents decide whether to use tools

系统不是把工具直接映射为答案生成器，而是让子智能体在执行子任务时判断：

- 是否需要工具
- 需要哪种工具
- 如何解释工具输出

这样能保留系统灵活性，也更符合“智能体层”和“工具层”分离的设计目标。

### 3. Final answer extraction is separate from execution

执行结果与最终答案提取分离，是为了提高稳定性：

- 单步输出可能包含中间推理、解释、工具原文
- 最终评测往往只需要一个标准答案

因此项目使用终止智能体和 `answer_utils.py` 对结果进行二次提取。

## Recommended Next Refactor

如果你下一步继续做结构优化，建议按下面顺序推进：

1. 为 `pipelines/multi_agent_pipeline.py` 和 `services/action_selection.py` 补单元测试。
2. 把 action 选择规则进一步数据化，减少 if/else 链。
3. 为数据集评测输出定义更严格的 schema，便于后续误差分析。
4. 视需要引入更正式的日志层，替代当前的 `print` 式 CLI 输出。
