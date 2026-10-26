# SAA 硕士论文实验项目交接记录

更新日期：2026-10-26（Asia/Shanghai）。本文件用于在新设备、新 Codex 会话中快速恢复项目上下文；**当前仓库代码和实验结果始终是事实来源**。开始工作时先核对 Git 提交、数据与结果文件是否仍与此记录一致。

## 新会话先读这里

在仓库根目录打开项目，并告诉 Codex：

> 请先阅读 `docs/PROJECT_HANDOFF.md`、`docs/ROLLING_SCHEDULER.md`，核对当前代码、Git 状态及本地结果，再继续 SAA 论文实验。不要把交接记录当作已验证的最新状态。未经我明确要求，不运行付费模型或大规模评测。

本次会话给出的仓库工作规则（若新设备没有对应的 `AGENTS.md`，仍按这里执行）：每次改代码须同步更新相关功能注释；依赖与语法遵循项目已有版本；不影响无关样式和功能；只检查改动界面的语法。**本交接记录不是代码修改。**

## 研究目标与当前结论

论文核心方法 SAA：LLM 将任务分解成带依赖的子任务 → 语义匹配得到任务—智能体得分矩阵 → Agent Slot 匈牙利指派 → DAG 执行。现有代码还包含滚动指派、执行反馈和有限重试。研究问题分别是：端到端任务质量；语义匹配相对规则/随机匹配；匈牙利相对贪心/随机指派；滚动重指派与失败反馈；Token、耗时和模型调用成本。

**目前不能宣称匈牙利、语义匹配或反馈提升了质量。**已运行的 10 道 GSM-Hard 冒烟题中，9 道成功规划的题均为 `DIRECT`、各只有 1 个子任务；另 1 道在分解调用时遭遇服务端 503，没有计划。因此观察到的联合指派轮次为 0，尚无检验有限容量匹配优势的自然场景。这只描述该 10 题样本，不能外推到全部 GSM-Hard。

GSM-Hard 和 MMLU-Pro 继续用于端到端质量与成本对照；还需补充能观察 DAG 并行、智能体技能差异、容量竞争和执行失败的任务集。公开数据不一定自带与本项目智能体池对应的能力和容量标签，必须事先建立映射、固定筛选规则，不能根据 SAA 的结果挑题。

## 代码定位与已知机制

| 主题 | 当前实现及定位 |
| --- | --- |
| 配置默认值 | `models.py` 的 `SchedulerConfig`：默认 `rolling-hungarian`、`max_parallel=1`、`service_concurrency=1`、`max_attempts=2`、`max_plan_tasks=3`，允许 `DIRECT`，默认启用反馈。 |
| 任务分解 | `llm/llm_interface.py` 的 `_build_decomposition_prompt` 要求：若一个智能体可独立完成，应选择 `DIRECT`；`question_solution/task_decomposer.py` 解析依赖并可将超长计划折叠为 `DIRECT`。 |
| 语义匹配 | `semantic_matcher.py` 使用 `all-MiniLM-L6-v2`，任务标题、描述及技能对比智能体描述与能力；`services/assignment_service.py` 的 `build_score_matrix` 还叠加规则路由偏置。现有 CLI 没有“仅规则/随机分数”入口。 |
| 静态指派 | `services/assignment_service.py` 的 `assign_subtasks_with_hungarian_core` 默认给每个智能体扩展与子任务数相同的虚拟 Slot；`services/execution_controller.py` 在 `static` 模式一次指派后冻结智能体。在无额外容量约束时，这接近各任务独立选择最高分。 |
| 滚动指派 | `services/execution_controller.py` 先排序就绪任务，再截取前 `max_parallel` 个进行容量约束匹配。默认值为 1 时，每轮最多一个任务进入匹配；即使 DAG 同时有多个就绪任务，也不能在该轮体现多任务匹配优势。 |
| 反馈消融 | `--no-feedback` 仅关闭动态加权分数；按失败类型重试、能力失败换人等硬路由仍存在。测试“完全去反馈”需要独立开关。 |
| Trace 与成本 | `pipelines/multi_agent_pipeline.py` 生成 schema v2 的分解、得分矩阵、逐轮执行及 usage；`evaluation/runtime.py` 保存逐题 trace、调用记录与汇总。未知用量调用不能按 0 Token 成本理解。 |
| 评测入口 | `main.py` 目前仅有 GSM-Hard、MMLU-Pro 两种 `--benchmark`；`--mode multi` 使用固定示例题。自定义任务集不能直接用现有 CLI 批量运行。 |

工程测试在 `tests/test_scheduler.py`、`tests/test_pipeline.py`、`tests/test_usage.py`，覆盖模拟匹配、重试、容量、依赖及用量等行为；尚不能代替真实数据的效果实验。调度说明见 `docs/ROLLING_SCHEDULER.md`。

## 已完成的实际操作与结果

1. 只读检查现有 `result/*.json`：`debug_idx8.json`、`execution_output.json`、`gsmhard_eval_output.json`、`gsmhard_eval_output_after_fix.json`、`mmlu_pro_eval_output.json` 均不是当前 schema v2 逐题结果，没有足够 trace 计算并行或容量竞争比例。旧结果使用的模型也不同，不能与当前实现直接做同口径比较。
2. 本机 Python 3.12.6；随后创建 `.venv` 并成功运行 `main.py --help`。GSM-Hard 数据文件存在。`dataset/MMLU-Pro` 目录存在，但在本次检查时没有 `data/test-*.parquet` 或 `data/validation-*.parquet`；目录存在不代表 MMLU 数据可运行。
3. 运行 GSM-Hard 前 10 题的 schema v2 冒烟测试：第 0 题先运行，随后用同一 checkpoint 的 `--resume` 跳过第 0 题并跑第 1–9 题。结果见本地 `result/preflight_gsm_v2.json` 和 `result/preflight_gsm_v2.jsonl`。配置：`--run-kind multi --scheduler rolling-hungarian --assignment-policy hungarian --max-parallel 2 --service-concurrency 1 --max-attempts 2 --max-tokens 40000 --deadline-seconds 1200 --no-charts-during-run`。
4. 这 10 题的汇总为 8/10 正确、9/10 有可解析预测，已知 Token 总数 37,364（均值 3,736.4），平均每题约 95.54 秒。这只是冒烟观察，既无单智能体同题对照，也无算法消融，不可据此写论文效果结论。
5. 第 0–6、8–9 题各 1 个 `DIRECT` 子任务、1 轮且每轮 1 个候选；第 7 题在分解模型调用时返回 `InternalServerError 503`，没有分解与指派 trace。该请求记录 `unknown_usage_calls=1`、已知用量 0、预算估计扣减 6145；已知用量 0 **不等于实际费用为 0**。503 发生于规划阶段，不能计入“执行失败后的滚动恢复”。
6. 运行时曾看到 Hugging Face 未登录、Windows 缓存无法使用 symlink、模型权重中 `embeddings.position_ids` 为 unexpected 的提示；嵌入模型随后加载成功，其他 9 题完成了评测。不要将这些提示自动当作任务失败。

记录时仓库分支为 `main`，HEAD 为 `840e3731c74903a33b0dbf856f2248ce486bf08f`，工作区在新增本文档前无代码改动。上述新实验的 fingerprint 为 `a35d0db6e9e23e5a2df813e0960ce288fd9ed38ee5fa7915a823886222c3cb3b`；换设备后应以结果文件内的 `config.experiment` 复核。

**跨设备注意：** `.venv/`、`result/`、`dataset/MMLU-Pro/data/` 均被 `.gitignore` 排除。只拉取 Git 仓库不会带来虚拟环境、schema v2 冒烟结果、模型缓存或 MMLU 数据。需由用户另行备份/同步实验结果和合法取得的数据；勿将 API Key 写入交接文档或提交仓库。当前模型与环境变量名见 `llm/llm_config.py`，不要记录凭据值。

## 下一阶段实验设计与待办

**先做，不需要模型调用：**

1. 固定容量竞争的口径：并行就绪轮次、真正进入同轮联合指派的轮次、多个任务共享首选可行智能体且需求超过其容量的轮次，分别以全部轮次/全部题目为分母报告。现有滚动 trace 有 `candidate_ids`、`deferred_ids`、`raw_scores`、`feasible`、`capacities`，但最好补截断前的 `ready_ids`、`eligible_ids` 和明确的冲突标志。
2. 预先选定额外任务集与筛选标准，保留单节点、串行链、并行无冲突及并行有潜在容量冲突的任务。每题记录原题、资源、验收标准、预期依赖、技能需求、可行智能体与容量；不按任何算法的运行结果筛选。公开来源优先考察 [TaskBench](https://github.com/microsoft/JARVIS/blob/main/taskbench/README.md) 的工具 DAG；可考察 [MultiAgentBench/MARBLE](https://github.com/ulab-uiuc/MARBLE) 和 [FDABench](https://github.com/fdabench/FDAbench) 的端到端协作或数据分析任务。TaskBench 的工具节点不能直接视作本项目智能体，需人工审核映射与工具可执行性。
3. 最小代码工作（尚未授权或实施）：增加自定义任务集适配/CLI；增加规则及随机匹配分数对照；增加截断前就绪集日志和离线指派重放统计；将软分数反馈与失败类型硬路由拆开消融；必要时增加确定性执行故障注入。修改代码时遵守上面的仓库工作规则。
4. 后续实验区分两层：离线固定相同 DAG、分数、可行性与容量，比较随机/贪心/匈牙利的指派覆盖率、匹配得分、外部能力标签及冲突轮次；端到端再在同一批样本上比较静态、滚动、去语义、去反馈和单智能体的完成率、Token、时间及调用数。
5. 使用相同题目 ID、模型、提示词、工具权限、分解计划、预算、尝试数、并发、答案解析规则；每组独立输出与 checkpoint。共享分解缓存可使计划一致，但成本报告必须区分“缓存重放执行成本”和“含一次规划的完整系统成本”。随机方法预先固定多个种子，报告配对差值及不确定性。

**不要直接把样本量从 GSM 10 题扩大到 30–50 题来证明指派优势。**目前这批题没有产生联合指派。公开数据集也不能直接接入现有 `--benchmark`；先完成数据筛选、能力映射和适配方案，再进行少量冒烟。

## 新设备恢复检查

1. `git status --short`、`git rev-parse HEAD`，阅读本文件、`docs/ROLLING_SCHEDULER.md` 和相关代码；区分本文档中的历史观察与当前事实。
2. 使用项目锁文件建立兼容 Python 3.12 的环境，例如 `uv sync --frozen`；`main.py --help` 只验证入口，不调用模型。
3. 检查所需数据文件和本地结果是否已从旧设备同步。若 `result/preflight_gsm_v2.json` 不在新设备，不要假装已有 10 题 trace；若在，先核对 schema、fingerprint、题号及错误记录。
4. 在启动任何新评测前，先确认本轮任务是否允许模型费用、数据集范围和输出路径；不覆盖或混合旧实验结果。
