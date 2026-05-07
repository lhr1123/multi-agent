"""Single-LLM baseline execution helpers."""

import json
import re
import time

from llm.llm_config import SINGLE_BASELINE_TEMPERATURE, SINGLE_BASELINE_TOP_P
from services.result_parsing import strip_code_fence


def solve_with_single_llm(
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
    response = llm.client.chat.completions.create(
        model=llm.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=SINGLE_BASELINE_TEMPERATURE,
        top_p=SINGLE_BASELINE_TOP_P,
    )
    end_time = time.time()

    content = response.choices[0].message.content or ""
    json_str = strip_code_fence(content)
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", content)
        answer = numbers[-1] if numbers else content
        result = {"answer": answer, "reasoning": "JSON parse failed"}

    result["tokens_used"] = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
    }
    result["elapsed_time"] = end_time - start_time
    return result
