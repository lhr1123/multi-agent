# Multi-Agent Task Allocation System

一个面向数学推理与任务分配实验的多智能体系统。当前代码基线为 `1.4` 版本，后续迭代均建议基于这一版本继续开发。

系统主流程是：

1. 大模型对原始任务做分解，生成子任务、依赖关系和技能需求。
2. embedding 模型计算子任务与子智能体的语义匹配分数。
3. 匈牙利算法基于匹配分数完成核心分配。
4. 子智能体在工作流中执行子任务，并自主决定是否调用工具。
5. 终止智能体汇总中间结果，产出最终答案。

## Project Structure

```text
multi_agent_1.4/
├── main.py                       # 轻量 CLI 入口，只负责参数解析和调度
├── models.py                     # Task / SubTask 等数据结构
├── semantic_matcher.py           # embedding 语义匹配
├── llm/
│   ├── llm_config.py             # 全局模型与 API 配置
│   └── llm_interface.py          # LLM 接口与具体客户端封装
├── pipelines/
│   └── multi_agent_pipeline.py   # 多智能体单次运行主流程
├── evaluation/
│   ├── gsmhard_evaluator.py      # GSM-hard 数据集评测入口
│   └── mmlu_pro_evaluator.py     # MMLU-Pro 数据集评测入口
├── services/
│   ├── action_selection.py       # 子任务到 action 的路由策略
│   ├── assignment_service.py     # 得分矩阵与匈牙利分配
│   ├── result_parsing.py         # LLM / workflow 结果解析
│   └── single_llm_solver.py      # 单模型基线求解
├── reporters/
│   └── console_reporter.py       # 控制台输出格式化
├── multi_agent_pool/
│   ├── actions.py                # 工具动作定义
│   ├── agent_pool.py             # 子智能体池与 persona 定义
│   ├── personas.jsonl            # 智能体画像
│   └── workflow.py               # 工作流编排与执行
├── question_solution/
│   ├── hungarian_algorithm.py    # 匈牙利算法实现
│   ├── task_decomposer.py        # 任务分解器
│   └── task_executor.py          # 执行辅助逻辑
├── utils/
│   ├── dataset_utils.py          # 数据集加载
│   ├── dependency_utils.py       # 子任务依赖关系清洗
│   ├── answer_utils.py           # 最终答案提取与准确率判断
│   ├── eval_visualization.py     # 评测图表生成
│   └── utils.py                  # Python 执行等通用工具
├── dataset/                      # GSM-hard、MMLU-Pro 等数据集
├── result/                       # JSON 结果与图表输出
└── docs/
    └── ARCHITECTURE.md           # 架构说明
```

## Runtime Flow

### 1. Task orchestration

`main.py` 只负责初始化模型、解析 CLI 参数，并把请求转交给 `pipelines/` 或 `evaluation/`。

真正的单次运行流程位于 `pipelines/multi_agent_pipeline.py`，其中会初始化任务分解层 LLM，并调用 `TaskDecomposer` 生成：

- 子任务列表
- 依赖关系
- 所需技能标签

随后会通过 `sanitize_subtask_dependencies()` 对依赖图进行轻量清洗，避免无效依赖、自依赖和简单环。

### 2. Semantic matching and assignment

`SemanticMatcher` 使用 embedding 模型计算：

- 子任务标题 / 描述 / 技能需求
- 子智能体名称 / 类别 / persona

之间的相似度分数，生成得分矩阵。

`question_solution/hungarian_algorithm.py` 中的匈牙利算法是任务分配核心。当前实现允许通过扩展 agent slots 的方式处理“子任务数量大于智能体数量”的场景，同时仍保持由匈牙利算法给出全局最优分配。

### 3. Workflow execution

每个子任务会被分配给一个子智能体。子智能体不是“等于工具”，而是：

- 先由大模型判断当前子任务该怎么做
- 再决定是否调用绑定工具
- 最后返回结构化或半结构化结果

这部分逻辑主要位于：

- `pipelines/multi_agent_pipeline.py`
- `services/action_selection.py`
- `multi_agent_pool/agent_pool.py`
- `multi_agent_pool/workflow.py`
- `multi_agent_pool/actions.py`

### 4. Result aggregation

所有步骤执行完成后，终止智能体会汇总：

- sink 子任务输出
- 全部 step 输出
- 结构化 key facts

并给出最终答案。评测模式下，会根据 benchmark 分别通过 `evaluation/gsmhard_evaluator.py` 或 `evaluation/mmlu_pro_evaluator.py` 调用对应的答案提取逻辑并计算准确率。

## Configuration

统一在 `llm/llm_config.py` 中维护模型配置。

- `TASK_ORCHESTRATOR_MODEL`: 任务分配层 / 总控层使用的模型
- `SUB_AGENT_MODEL`: 子智能体执行层使用的模型
- `SINGLE_BASELINE_MODEL`: 单模型对照实验使用的模型

环境变量：

```bash
SILICONFLOW_API_KEY=...
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

## Usage

### 单条多智能体运行

```bash
python main.py --mode multi
```

### 单模型基线

```bash
python main.py --mode single
```

### 对比实验

```bash
python main.py --mode both
```

### GSM-hard 数据集评测

```bash
python main.py --mode dataset --benchmark gsm-hard --dataset-path dataset/gsm-hard/gsmhardv2.jsonl --dataset-limit 10 --save-path result/gsmhard_eval_output.json
```

### 同时运行单模型基线对比

```bash
python main.py --mode dataset --benchmark gsm-hard --dataset-path dataset/gsm-hard/gsmhardv2.jsonl --dataset-limit 10 --save-path result/gsmhard_eval_output.json --compare-single
```

### MMLU-Pro 数据集评测

```bash
python main.py --mode dataset --benchmark mmlu-pro --dataset-path dataset/MMLU-Pro --dataset-limit 100 --save-path result/mmlu_pro_eval_output.json
```

### MMLU-Pro 同时运行单模型基线对比

```bash
python main.py --mode dataset --benchmark mmlu-pro --dataset-path dataset/MMLU-Pro --dataset-limit 100 --save-path result/mmlu_pro_eval_output.json --compare-single
```

## Outputs

评测模式会在 `result/` 下产出：

- 评测摘要 JSON
- 每条样本的预测结果
- 统计图表

图表默认保存到：

```text
result/charts/<output_file_name>/
```

如果本地可用 `matplotlib`，输出 PNG；否则自动降级为 SVG。

## Refactor Notes for 1.4

这一版在保持现有功能不变的前提下，进一步减轻了 `main.py` 的负担：

- 把多智能体单次运行流程下沉到 `pipelines/`
- 把 GSM-hard / MMLU-Pro 评测入口下沉到 `evaluation/`
- 把分配、action 路由、结果解析、单模型基线拆到 `services/`
- 把控制台打印逻辑拆到 `reporters/`
- 保留 `main.py` 作为统一 CLI 入口，避免业务细节继续堆积

当前 `MMLU-Pro` 适配说明：

- 使用 `--benchmark mmlu-pro` 进入多选题评测链路
- 如果本地存在 `validation-*.parquet`，会自动按 category 组装 few-shot CoT 前缀
- 如果本地只有 `test-*.parquet`，则自动退化为无 validation prefix 的评测方式

后续如果继续做结构优化，比较自然的下一步会是：

- 为 `services/action_selection.py` 增加更清晰的策略对象或规则表
- 为 `pipelines/multi_agent_pipeline.py` 补测试，尤其是依赖执行与终止汇总部分
- 继续把 workflow 执行期的日志输出从 `print` 升级为更稳定的日志接口
