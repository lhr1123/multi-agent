"""Single-LLM baseline execution helpers."""

import json
import time

from llm.llm_config import (
    SILICONFLOW_EXTRA_BODY,
    SINGLE_BASELINE_TEMPERATURE,
    SINGLE_BASELINE_TOP_P,
)
from services.result_parsing import strip_code_fence
from services.usage_tracker import UsageTracker, usage_scope, tracked_completion


def solve_with_single_llm(llm, task_title, task_description, *args, scheduler_config=None, **kwargs):
    tracker = UsageTracker(scheduler_config)
    with usage_scope(tracker, "single"):
        try:
            result = _solve_with_single_llm(llm, task_title, task_description, *args, **kwargs)
        except Exception as exc:
            exc.usage_report = tracker.snapshot()
            raise
    result["usage"] = tracker.snapshot()
    result["tokens_used"]["total_tokens"] = result["usage"]["known_total_tokens"]
    return result


def _solve_with_single_llm(
    llm,
    task_title: str,
    task_description: str,
    system_prompt: str = "你是一个数学问题求解专家。",
    output_schema: str = (
        '{\n'
        '  "answer": "最终答案",\n'
        '  "reasoning": "简要推理过程"\n'
        '}'
    ),
) -> dict:
    prompt = (
        f"{system_prompt}请直接解答以下问题，并给出 JSON 输出。\n"
        f"问题：{task_description}\n\n"
        f"请按以下 JSON 返回：\n{output_schema}\n"
        "只返回 JSON。"
    )

    start_time = time.time()
    response = tracked_completion(llm.client, stage="final",
        model=llm.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=SINGLE_BASELINE_TEMPERATURE,
        top_p=SINGLE_BASELINE_TOP_P,
        extra_body=SILICONFLOW_EXTRA_BODY,
    )
    end_time = time.time()

    content = response.choices[0].message.content or ""
    json_str = strip_code_fence(content)
    try:
        result = json.loads(json_str)
        if not isinstance(result, dict):
            raise ValueError("Expected object")
    except (json.JSONDecodeError, ValueError):
        result = {"answer": content, "reasoning": "JSON parse failed", "json_parse_failed": True}
    result["raw_response"] = content
    result["baseline_mode"] = "plain"

    result["tokens_used"] = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    result["elapsed_time"] = end_time - start_time
    return result
