# Rolling Hungarian Scheduler

Updated 2026-10-27. TaskBench fixed-DAG offline replay uses this controller
with a local success stub and frozen per-sample semantic score matrix; it does
not exercise LLM planning, external tools, or answer correctness. See
[TaskBench offline assignment](TASKBENCH.md) for its separate entrypoint and
experiment limits.

## Scope

The default scheduler is `rolling-hungarian`. A minimal planner first chooses
`DIRECT` or a bounded `DECOMPOSE` plan, then the controller reassigns ready work
after execution feedback. Essential work is marked required; optional review
can be skipped after required evidence succeeds. This is a deterministic
control rule, not confidence-based answer correctness prediction.

`--scheduler static` freezes initial assignments using the historical
unlimited-slot assignment policy. Both modes use corrected accounting,
failure handling, prompts and output checks. Static mode is a corrected
baseline, not a byte-for-byte reproduction of historical results.

## Running

Existing model endpoint and credential environment variables remain supported.
All SiliconFlow chat requests explicitly send `enable_thinking=false`, configured
by `ENABLE_THINKING` in `llm/llm_config.py`; the value is recorded in experiment
metadata and per-call traces.
Start with a small sample and a new output/checkpoint path:

```powershell
.\.venv\Scripts\python.exe -B main.py --mode dataset --benchmark gsm-hard --dataset-path dataset/gsm-hard/gsmhardv2.jsonl --dataset-limit 10 --run-kind both --scheduler rolling-hungarian --max-attempts 2 --max-tokens 40000 --deadline-seconds 1200 --save-path result/rolling_gsm.json --checkpoint-path result/rolling_gsm.jsonl --no-charts-during-run
```

For MMLU-Pro:

```powershell
.\.venv\Scripts\python.exe -B main.py --mode dataset --benchmark mmlu-pro --dataset-path dataset/MMLU-Pro --dataset-limit 30 --sample-strategy stratified --run-kind both --scheduler rolling-hungarian --max-plan-tasks 3 --plan-cache-path result/mmlu_plans.jsonl --max-tokens 40000 --deadline-seconds 1200 --save-path result/rolling_mmlu.json --checkpoint-path result/rolling_mmlu.jsonl --no-charts-during-run
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--scheduler` | `rolling-hungarian` | `static` freezes the initial assignment |
| `--assignment-policy` | `hungarian` | `greedy` and `random` are matching ablations |
| `--max-parallel` | 1 | Maximum operations per sample in one wave |
| `--service-concurrency` | 1 | Process-wide concurrent requests per endpoint |
| `--agent-capacity reasoningagent=1` | max-parallel | Physical per-wave capacity |
| `--max-attempts` | 2 | Total attempts per task, including the first |
| `--max-tokens` | 0 | Per-sample budget; zero disables the token limit |
| `--max-output-tokens` | 4096 | Output limit on every model request |
| `--final-reserve-tokens` | 8192 | Budget withheld from non-final requests |
| `--deadline-seconds` | 600 | Deadline checked on dispatch and feedback |
| `--seed` | 0 | Random matching seed, not a provider generation seed |
| `--max-plan-tasks` | 3 | Maximum work items; oversized plans collapse to DIRECT |
| `--plan-cache-path` | none | Shared JSONL decomposition cache for paired scheduler runs |
| `--evidence-max-chars` | 6000 | Compact evidence budget passed downstream |
| `--optional-utility-threshold` | 1.3 | Minimum utility for optional work after required work completes |
| `--reliability-priors` | none | Frozen development-set JSON map keyed by agent ID |
| `--no-direct-plan` | off | Disable the planner's DIRECT route |
| `--no-early-stop-optional` | off | Force optional planned work to execute |
| `--sample-strategy` | sequential | Use `stratified` for category-balanced MMLU-Pro samples |
| `--no-feedback` | off | Fit-only scoring ablation |
| `--scheduler-config` | none | JSON with any SchedulerConfig fields |

CLI options override JSON fields. Capacities refer to concurrent operations,
never total lifetime uses. Set capacity to zero to disable an agent. All roles
sharing an endpoint share its request limiter, including dataset workers.
This limiter is process-local; separate shard processes need external limiting.
Parallelism requires both a larger wave and sufficient service concurrency.

## Scheduling and Scores

Only tasks with successful dependencies enter a wave. Candidates are ordered by
attempt count and original DAG order, then limited to max_parallel. This is a
bounded wave heuristic, not a globally optimal budget or latency schedule.

Scores combine normalized semantic fit, a development-set reliability prior,
within-run outcomes, observed operation cost, information-gain metadata,
agent diversity, and repeated capability failures. Scores and features are
logged. Gold labels never enter the controller; priors must be frozen before
test evaluation.

Required-task matching first maximizes feasible cardinality, then weighted score.
Every solve includes enough zero-valued defer columns; deferred tasks remain
pending. Incompatible pairs are masked. Task metadata allowed_agent_ids and
required_capabilities provides exact eligibility constraints.

Without resource competition, Hungarian can reduce to independent best-agent
selection. Single-ready-task waves have no global assignment advantage.
Embeddings now include descriptions and use a bounded cache. The embedding
model is unchanged; model replacement should be a separate ablation.

## Completion and Failures

States are pending, ready, running, succeeded, failed, skipped and blocked. Failed
upstream tasks never unlock dependents. Invalid dependency indices and cycles
are rejected rather than silently deleting edges.

Failures are classified as transport, format, context, tool, capability,
budget or deadline. Rolling retries keep transport/format/context/tool repairs
on the same agent. Capability failures move to another feasible agent when one
exists. All routes remain bounded by `max_attempts`.

Completion checks verify execution and structured output, not factual or
mathematical correctness. Optional work is stopped only after all required
work succeeds; confidence never triggers early stopping.
Attempted file writes/deletions are not automatically retried. Python executes
in a separate process with a timeout and isolated stdout; this is not a
security sandbox or an idempotency guarantee for generated code.

Stop reasons distinguish completed, budget_exhausted, deadline_exceeded,
blocked, attempts_exhausted and execution_error. Final answer generation has
a separate finalization_status. Incomplete runs may emit best-effort answers
with an explicit error; completing the DAG does not guarantee a final answer.

## Accounting and Checkpoints

All active pipeline model calls are tracked: decomposition, code generation,
tool decisions, summaries, reasoning and final extraction. Automatic SDK retries
are disabled on tracked requests. Operation retries appear in the trace.

Reservations use UTF-8 bytes plus overhead and output caps. This is conservative
text estimation, not a guarantee about provider tokenizers. Reservations are
atomic; actual provider usage settles them. Failed requests or missing usage
remain unknown and consume the reserved estimate for budget purposes.
Budget refusals before a request is sent consume no tokens.

Token totals are known provider usage, a lower bound when unknown_usage_calls
is nonzero. Budget charges are separate. The final reserve is a minimum
allowance, not a guarantee that an arbitrarily long summary fits. SDK/network
timeouts and uninterruptible local I/O can exceed the requested wall deadline;
Python subprocesses are explicitly terminated on timeout.

Schema-v2 samples retain decomposition, attempts, matching matrices, outcomes
and usage. Every sub-agent receives the original task, current work item,
dependency evidence, candidates, prior attempts and remaining budget. Final
aggregation reads compact structured evidence plus unresolved tasks; raw outputs
remain in the trace. MMLU-Pro no longer guesses random answers on parse failure.

Each rolling round also records `ready_ids` and `eligible_ids` before the
`max_parallel` cutoff, `candidate_ids` after the cutoff, and
`candidate_capacity_conflict` for candidates whose top feasible agent is
oversubscribed. The conflict flag describes the candidate wave, not all
ready tasks outside that wave. Report round and sample denominators separately.

Checkpoints include model/sampling settings, scheduler configuration, code hash,
Git commit and endpoint fingerprints. Resume rejects incompatible or legacy
checkpoints. Non-resume runs refuse to append to existing nonempty checkpoints.
Merge rejects mixed fingerprints. Use a separate path per experiment.

## Validation and Experiments

Use `--single-baseline python` for the capability-matched single-agent baseline.
The same baseline model decides whether to use Python, generates and repairs its
own code, executes through the shared local executor, and reads the result. It
does not use decomposition, semantic matching, Hungarian assignment, personas,
or the multi-agent finalizer. Dataset reference code and labels are never added
to its prompt.

```powershell
.\.venv\Scripts\python.exe -B main.py --mode dataset --benchmark gsm-hard --dataset-path dataset/gsm-hard/gsmhardv2.jsonl --dataset-limit 50 --run-kind single --single-baseline python --max-attempts 2 --save-path result/gsm_single_python50.json --checkpoint-path result/gsm_single_python50.jsonl --no-charts-during-run
```

Result metadata and each baseline sample record `single_baseline_mode`. Python
decisions, generated code, execution output, retries and errors are retained in
`tool_trace`. Use separate checkpoint files for `plain` and `python`; their
experiment fingerprints intentionally differ.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests use mock APIs without paid inference or model downloads. Coverage includes
matching against exhaustive search, reassignment, deferral, dependency failures,
failure routing, full task context, plan replay, stratified sampling, concurrent
accounting, subprocess timeouts, both evaluators and checkpoint compatibility.

Compare corrected static, rolling Hungarian, greedy, random and no-feedback with
identical samples, models and resources. Reuse one `--plan-cache-path` across
static and rolling runs so their task plans are identical. Use different output
and checkpoint paths for each run. Tune only on development data and
report paired uncertainty, token usage, failures and latency. A budget-matched
single-model multi-sample baseline is still needed for research conclusions.
Engineering tests do not establish accuracy improvements or token savings.
