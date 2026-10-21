"""Single-agent baseline with optional access to the shared Python executor."""

from __future__ import annotations

import json
import time

from llm.llm_config import (
    SILICONFLOW_EXTRA_BODY,
    SINGLE_BASELINE_TEMPERATURE,
    SINGLE_BASELINE_TOP_P,
)
from services.code_execution import execute_python
from services.result_parsing import safe_json_loads_from_text
from services.usage_tracker import UsageTracker, tracked_completion, usage_scope


DEFAULT_OUTPUT_SCHEMA = (
    '{\n'
    '  "answer": "final answer",\n'
    '  "reasoning": "brief reasoning"\n'
    '}'
)


def _completion(llm, stage, system_prompt, user_prompt):
    response = tracked_completion(
        llm.client,
        stage=stage,
        model=llm.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=SINGLE_BASELINE_TEMPERATURE,
        top_p=SINGLE_BASELINE_TOP_P,
        extra_body=SILICONFLOW_EXTRA_BODY,
    )
    content = response.choices[0].message.content or ""
    parsed = safe_json_loads_from_text(content)
    if not parsed:
        raise ValueError(f"{stage} returned invalid JSON: {content[:500]}")
    return parsed, content


def _decision_prompt(task_description, output_schema):
    return f"""Solve the following task as one agent. You may use the local Python executor.

Task:
{task_description}

Rules:
1. Treat every literal number and constraint in the task as authoritative, even when unrealistic.
2. Never repair an apparent typo, clamp a negative value, or replace a value using common sense.
3. Use Python when exact arithmetic or a deterministic calculation would reduce error.
4. Python code must use only the standard library, put the answer in `result`, and print it.
5. Do not assume access to hidden dataset code, labels, reference answers, or other agents.
6. The required final response schema is shown below. When Python is unnecessary, put only the
   value intended for its `answer` field in `direct_answer`, not another JSON object.
{output_schema}

Return strict JSON only:
{{
  "use_python": true,
  "code": "result = 1 + 1\\nprint(result)",
  "direct_answer": "",
  "reasoning": "short tool decision rationale"
}}
"""


def _repair_prompt(task_description, code, error):
    return f"""Repair Python code for the original task.

Original task:
{task_description}

Failed code:
{code}

Execution error:
{error}

Preserve all literal numbers exactly. Use only the Python standard library. Put the final answer
in `result` and print it. Return strict JSON only:
{{"code": "corrected code", "reasoning": "short repair explanation"}}
"""


def _final_prompt(task_description, execution, output_schema):
    return f"""Return the final answer to the original task using the deterministic Python result.

Original task:
{task_description}

Python stdout:
{execution.get('output', '')}

Python return value:
{execution.get('return_value', '')}

Rules:
1. The Python return value was calculated from the literal task values and is authoritative.
2. Do not replace negative, fractional, large, or otherwise unrealistic values.
3. Keep the answer concise and preserve required units or multiple-choice formatting.

Return strict JSON only in this format:
{output_schema}
"""


def solve_with_single_tool_llm(
    llm,
    task_title: str,
    task_description: str,
    scheduler_config=None,
    system_prompt: str = "You are a careful single-agent problem solver with access to Python.",
    output_schema: str = DEFAULT_OUTPUT_SCHEMA,
):
    """Run a capability-matched single-agent baseline without orchestration."""
    tracker = UsageTracker(scheduler_config)
    started = time.time()
    tool_trace = {"used": False, "attempts": []}
    try:
        with usage_scope(tracker, "single_tool"):
            decision, decision_raw = _completion(
                llm,
                "single_tool_decision",
                system_prompt,
                _decision_prompt(task_description, output_schema),
            )
            use_python = decision.get("use_python")
            if not isinstance(use_python, bool):
                raise ValueError("single_tool_decision requires boolean use_python")

            if not use_python:
                answer = str(decision.get("direct_answer") or decision.get("answer") or "").strip()
                if not answer:
                    raise ValueError("Direct single-tool decision returned no answer")
                result = {
                    "answer": answer,
                    "reasoning": str(decision.get("reasoning", "") or ""),
                    "raw_response": decision_raw,
                    "decision_raw_response": decision_raw,
                    "tool_trace": tool_trace,
                    "baseline_mode": "python",
                }
            else:
                tool_trace["used"] = True
                code = str(decision.get("code", "") or "").strip()
                if not code:
                    raise ValueError("Python decision returned empty code")

                execution = None
                max_attempts = tracker.config.max_attempts
                for attempt_index in range(max_attempts):
                    execution = execute_python(
                        code,
                        timeout=max(0.001, min(30.0, tracker.remaining_seconds())),
                    )
                    tool_trace["attempts"].append({
                        "attempt": attempt_index + 1,
                        "code": code,
                        "success": bool(execution.get("success")),
                        "output": execution.get("output", ""),
                        "return_value": execution.get("return_value"),
                        "error": execution.get("error", ""),
                    })
                    if execution.get("success"):
                        break
                    if attempt_index + 1 >= max_attempts:
                        error = RuntimeError("Single-tool Python attempts exhausted")
                        error.tool_trace = tool_trace
                        raise error
                    repaired, _ = _completion(
                        llm,
                        "single_tool_repair",
                        system_prompt,
                        _repair_prompt(task_description, code, execution.get("error", "")),
                    )
                    code = str(repaired.get("code", "") or "").strip()
                    if not code:
                        raise ValueError("Single-tool repair returned empty code")

                final, final_raw = _completion(
                    llm,
                    "single_tool_final",
                    system_prompt,
                    _final_prompt(task_description, execution or {}, output_schema),
                )
                answer = str(final.get("answer") or final.get("final_answer") or "").strip()
                if not answer:
                    raise ValueError("Single-tool final response returned no answer")
                result = {
                    **final,
                    "answer": answer,
                    "raw_response": final_raw,
                    "decision_raw_response": decision_raw,
                    "tool_trace": tool_trace,
                    "baseline_mode": "python",
                }
    except Exception as exc:
        exc.usage_report = tracker.snapshot()
        if not hasattr(exc, "tool_trace"):
            exc.tool_trace = tool_trace
        raise

    result["elapsed_time"] = time.time() - started
    result["usage"] = tracker.snapshot()
    result["tokens_used"] = {
        "prompt_tokens": sum(call.get("prompt_tokens") or 0 for call in result["usage"]["calls"]),
        "completion_tokens": sum(call.get("completion_tokens") or 0 for call in result["usage"]["calls"]),
        "total_tokens": result["usage"]["known_total_tokens"],
    }
    return result
