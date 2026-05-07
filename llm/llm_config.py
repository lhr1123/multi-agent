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

This module also supports separate OpenAI-compatible endpoints for:

- task orchestration
- sub-agent execution
- single-model baseline

If the role-specific env vars are unset, they fall back to the shared
`SILICONFLOW_*` values for backward compatibility.
"""

import os

from openai import OpenAI


def _env_with_fallback(primary: str, fallback: str, default: str = "") -> str:
    value = os.getenv(primary)
    if value:
        return value
    value = os.getenv(fallback)
    if value:
        return value
    return default


def _float_env_with_fallback(primary: str, fallback: str, default: float) -> float:
    raw = _env_with_fallback(primary, fallback, "")
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


# Shared fallback configuration.
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")

# Role-specific endpoint configuration.
TASK_ORCHESTRATOR_API_KEY = _env_with_fallback("TASK_ORCHESTRATOR_API_KEY", "SILICONFLOW_API_KEY")
TASK_ORCHESTRATOR_BASE_URL = _env_with_fallback(
    "TASK_ORCHESTRATOR_BASE_URL",
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1",
)

SUB_AGENT_API_KEY = _env_with_fallback("SUB_AGENT_API_KEY", "SILICONFLOW_API_KEY")
SUB_AGENT_BASE_URL = _env_with_fallback(
    "SUB_AGENT_BASE_URL",
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1",
)

SINGLE_BASELINE_API_KEY = _env_with_fallback("SINGLE_BASELINE_API_KEY", "SILICONFLOW_API_KEY")
SINGLE_BASELINE_BASE_URL = _env_with_fallback(
    "SINGLE_BASELINE_BASE_URL",
    "SILICONFLOW_BASE_URL",
    "https://api.siliconflow.cn/v1",
)

# Shared fallback sampling configuration.
LLM_TEMPERATURE = _float_env_with_fallback("LLM_TEMPERATURE", "", 0.0)
LLM_TOP_P = _float_env_with_fallback("LLM_TOP_P", "", 1.0)

# Role-specific sampling configuration.
TASK_ORCHESTRATOR_TEMPERATURE = _float_env_with_fallback(
    "TASK_ORCHESTRATOR_TEMPERATURE",
    "LLM_TEMPERATURE",
    LLM_TEMPERATURE,
)
TASK_ORCHESTRATOR_TOP_P = _float_env_with_fallback(
    "TASK_ORCHESTRATOR_TOP_P",
    "LLM_TOP_P",
    LLM_TOP_P,
)

SUB_AGENT_TEMPERATURE = _float_env_with_fallback(
    "SUB_AGENT_TEMPERATURE",
    "LLM_TEMPERATURE",
    LLM_TEMPERATURE,
)
SUB_AGENT_TOP_P = _float_env_with_fallback(
    "SUB_AGENT_TOP_P",
    "LLM_TOP_P",
    LLM_TOP_P,
)

SINGLE_BASELINE_TEMPERATURE = _float_env_with_fallback(
    "SINGLE_BASELINE_TEMPERATURE",
    "LLM_TEMPERATURE",
    LLM_TEMPERATURE,
)
SINGLE_BASELINE_TOP_P = _float_env_with_fallback(
    "SINGLE_BASELINE_TOP_P",
    "LLM_TOP_P",
    LLM_TOP_P,
)


# Task orchestration layer model.
# Responsible for task decomposition, dependency planning,
# workflow coordination, and final answer extraction.
TASK_ORCHESTRATOR_MODEL = "Qwen3.6-35B-A3B"

# Sub-agent execution model.
# Responsible for subtask reasoning and deciding whether to use tools.
SUB_AGENT_MODEL = "llama3.1-8b"

# Single-model baseline used in evaluation mode.
SINGLE_BASELINE_MODEL = "llama3.1-8b"

# Default fallback model when no explicit model is passed.
DEFAULT_MODEL = TASK_ORCHESTRATOR_MODEL


# Role-specific API client instances.
task_orchestrator_llm_model = OpenAI(
    api_key=TASK_ORCHESTRATOR_API_KEY,
    base_url=TASK_ORCHESTRATOR_BASE_URL,
)

sub_agent_llm_model = OpenAI(
    api_key=SUB_AGENT_API_KEY,
    base_url=SUB_AGENT_BASE_URL,
)

single_baseline_llm_model = OpenAI(
    api_key=SINGLE_BASELINE_API_KEY,
    base_url=SINGLE_BASELINE_BASE_URL,
)

# Backward-compatible alias: historical callers defaulted to a single shared client.
llm_model = task_orchestrator_llm_model
