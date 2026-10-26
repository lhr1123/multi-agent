# TaskBench offline assignment, first version

Updated: 2026-10-27 (Asia/Shanghai). This is an offline scheduler experiment,
not a TaskBench tool-execution or answer-accuracy benchmark.

TaskBench's `data.json` is JSONL. Each sample includes an original
`user_request`, reference `task_nodes`, and `task_links`. The reference nodes
name tools, not this project's agents. This adapter treats the reference graph
as a fixed DAG for an **offline assignment experiment**. It never invokes a
TaskBench tool or a paid chat model. It may load/download the local
`all-MiniLM-L6-v2` embedding model when a reviewed tool map is supplied.

Source: [TaskBench README](https://github.com/microsoft/JARVIS/blob/main/taskbench/README.md)
and [format_data.py](https://github.com/microsoft/JARVIS/blob/main/taskbench/format_data.py).

## Prepare the data

Place one official domain's `data.json` and `tool_desc.json` under
`dataset/taskbench/source/`, for example
`dataset/taskbench/source/data_huggingface/`. The source directory is ignored
by Git. Keep a record of the downloaded source revision; result metadata also
records SHA-256 hashes of the exact files.
The inventory command reads local files only; it does not download TaskBench.

On 2026-10-26, the Hugging Face tools and daily-life API domains were obtained from the Microsoft
[`microsoft/Taskbench`](https://huggingface.co/datasets/microsoft/Taskbench)
dataset at revision `d12764a8fa650f49f50507a02c3e707c71c67621` via
`hf-mirror.com`, because direct GitHub and Hugging Face connections timed out
on this machine. The Hugging Face-domain `data.json` has SHA-256
`df81bbdc6fc069eeb855e4fce3308bf089c34003e9e2e6c319845f2c171cecd3`;
`tool_desc.json` has SHA-256
`0eb0654faea9a5f07c0102ca57bfc6de3f9e03d8a00c2776b82e22b1b7c31fbd`.
The daily-life API-domain files have SHA-256
`6e5c5fffd743fe79ce086d9f6457efdb9ab114802dcfeab6e6c39578c0f51fd4`
and `64b3ed4f0f3ad199d817695bc3bb7e2f7d9bc2cae2f310821744a62ae41aee13`,
respectively. The multimedia domain and other official companion files have
not been downloaded. The current adapter only needs `data.json` and
`tool_desc.json` for inventory and offline assignment.
The source directory remains ignored by Git and must be downloaded again on
another machine. The source has one exact duplicate ID row: 7,458 lines become
7,457 unique samples. Conflicting rows with the same ID remain an error.

To restore the Hugging Face-domain files on another machine with access to
the mirror, run these commands from the repository root:

```powershell
New-Item -ItemType Directory -Force dataset/taskbench/source/data_huggingface | Out-Null
curl.exe -fL -o dataset/taskbench/source/data_huggingface/data.json https://hf-mirror.com/datasets/microsoft/Taskbench/resolve/d12764a8fa650f49f50507a02c3e707c71c67621/data_huggingface/data.json
curl.exe -fL -o dataset/taskbench/source/data_huggingface/tool_desc.json https://hf-mirror.com/datasets/microsoft/Taskbench/resolve/d12764a8fa650f49f50507a02c3e707c71c67621/data_huggingface/tool_desc.json
Get-FileHash dataset/taskbench/source/data_huggingface/data.json -Algorithm SHA256
Get-FileHash dataset/taskbench/source/data_huggingface/tool_desc.json -Algorithm SHA256
```

Inventory the chosen domain before defining eligibility:

```powershell
.\.venv\Scripts\python.exe -B main.py --mode dataset --benchmark taskbench --dataset-path dataset/taskbench/source/data_huggingface/data.json --save-path result/taskbench_inventory.json
```

The summary lists tool names and graph types. It does not run a matching
policy, load the embedding model, or make model API calls.

## Review the tool mapping

Create a separate, versioned JSON mapping. The repository includes
`dataset/taskbench/pilot_tool_map.json` for a narrow smoke test. Only tools explicitly present in the map
are eligible. Use **exact** tool names from the inventory. Every entry must
be manually reviewed and may list more than one feasible agent:

```json
{
  "version": 1,
  "tools": {
    "Summarization": {
      "allowed_agent_ids": ["summerizeagent", "concludeagent"],
      "reviewed": true
    }
  }
}
```

An offline eligibility judgment does **not** mean either agent implements
that external API. Do not map a tool solely because its name sounds similar
to an agent's persona. Record excluded tools and the review rule before
examining algorithm results. The adapter rejects missing, invalid, cyclic,
or ambiguous graphs instead of guessing their edges. Repeated tool names
with name-only dependency links are ambiguous and are excluded.

## Compare assignments

For a mapped subset, select a wave size of at least two and specify
capacities that can create real competition. The plan limit is a *sample
filter*, not an instruction to collapse the gold graph:

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
.\.venv\Scripts\python.exe -B main.py --mode dataset --benchmark taskbench --dataset-path dataset/taskbench/source/data_huggingface/data.json --taskbench-tool-map dataset/taskbench/pilot_tool_map.json --max-plan-tasks 3 --max-parallel 2 --agent-capacity summerizeagent=1 --agent-capacity reasoningagent=1 --agent-capacity concludeagent=1 --seed 0 --save-path result/taskbench_new_run.json --checkpoint-path result/taskbench_new_run.jsonl
```

Use `--resume` with the same checkpoint and output path after interruption.
Use new paths if the data, mapping, code, or configuration changes; a non-resume
run refuses to overwrite an existing result. The
checkpoint fingerprint includes source, mapping, persona, relevant code, and
configuration hashes. `--dataset-offset` and `--dataset-limit` select a fixed
slice before filtering.

Each accepted sample is scored once by the existing semantic matcher and
routing rules. All three policies receive the same DAG, score matrix,
eligibility mask, and capacities. A local success stub advances the scheduler;
this measures assignment and deferral behavior, **not** real TaskBench task
completion. Reported assignment score is the project's own score and is not
an independent quality label. The summary includes all selected samples,
exclusion reasons, graph types, and per-policy totals. Per-sample rounds
include pre-truncation ready and eligible IDs plus conflict flags for the
joint candidate wave.

The pipeline also accepts `fixed_subtasks=` for a later execution adapter.
When supplied, it skips the LLM planner and marks `plan_source=fixed` in the
trace. The current TaskBench CLI intentionally stops at offline replay;
running real tools requires a separately reviewed executable subset and
independent outcome checks. For an end-to-end planning evaluation, keep the
reference graph only in the evaluator and give the planner the original user
request.

## First local pilot, 2026-10-26

`dataset/taskbench/pilot_tool_map.json` is a conservative **offline-only** map
for `Summarization` and `Question Answering`, based on the existing reasoning
personas. It does not assert that these agents implement the corresponding
Hugging Face APIs. A full replay of the 7,457 unique Hugging Face-domain
samples accepted 227: 224 single-node and 3 serial chains. Across 230 rounds,
all three policies had zero parallel-ready, joint-assignment, and capacity-
conflict rounds. Hungarian and greedy therefore had identical assignment
scores on this pilot. See ignored local result
`result/taskbench_pilot_full.json` for per-sample traces.

The current map cannot test the central capacity-constrained matching claim.
Although the source has 795 records marked as DAG, none survives this
conservative mapping. The pilot's 3 chains each have two nodes, assigned
sequentially to `concludeagent` by both Hungarian and greedy; the successful
stub does not show that this agent can actually perform both Hugging Face
tool tasks.
The daily-life API domain was also inventoried locally (4,318 samples); its
three read-only web-style tools (`search_by_engine`, `get_weather`,
`get_news_for_topic`) yield only 90 single-node samples and one chain under
the three-node limit. Expanding eligibility needs independently reviewed
capabilities and executable tools, rather than relabeling unrelated agents.
