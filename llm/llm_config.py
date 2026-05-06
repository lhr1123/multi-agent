"""Centralized LLM configuration for the multi-agent system.

Modify model names here to affect the whole project.

- `TASK_ORCHESTRATOR_MODEL`:
  Used by the task-allocation layer for decomposition, orchestration,
  and final answer aggregation.
- `SUB_AGENT_MODEL`:
  Used by sub-agents inside the workflow when they reason about a
  subtask and decide whether to call a tool.
- `SINGLE_BASELINE_MODEL`:
  Used by the single-LLM baseline in comparison experiments.
"""

import os

from openai import OpenAI


# SiliconFlow / OpenAI-compatible client configuration.
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")


# Task orchestration layer model.
# Responsible for task decomposition, dependency planning,
# workflow coordination, and final answer extraction.
TASK_ORCHESTRATOR_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Sub-agent execution model.
# Responsible for subtask reasoning and deciding whether to use tools.
SUB_AGENT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Single-model baseline used in evaluation mode.
SINGLE_BASELINE_MODEL = TASK_ORCHESTRATOR_MODEL

# Default fallback model when no explicit model is passed.
DEFAULT_MODEL = TASK_ORCHESTRATOR_MODEL


# Shared API client instance.
llm_model = OpenAI(
    api_key=SILICONFLOW_API_KEY,
    base_url=SILICONFLOW_BASE_URL,
)
