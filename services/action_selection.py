"""Action routing helpers for subtask execution."""

import json
import re
from typing import Any, Dict, Optional, Tuple

from multi_agent_pool.workflow import AgentCategory

from llm.llm_config import TASK_ORCHESTRATOR_TEMPERATURE, TASK_ORCHESTRATOR_TOP_P
from services.result_parsing import strip_code_fence


def generate_python_code_for_subtask(
    llm,
    original_task_description: str,
    subtask_description: str,
    context_text: str,
) -> str:
    prompt = f"""Write executable Python code to solve this subtask.
Requirements:
1. Use only Python standard library.
2. Put final value in variable `result`.
3. Print the final value.
4. Return code only.
5. Follow the numbers in the task literally, even if they are unrealistic.
6. Do not clamp negative values to zero, do not use abs/max/min to make values realistic, and do not add common-sense corrections unless the task explicitly asks for them.
7. Do not round or cast to int unless the task explicitly requests rounding or an integer count.

Original task:
{original_task_description}

Subtask:
{subtask_description}

Dependency context:
{context_text}
"""
    try:
        response = llm.client.chat.completions.create(
            model=llm.model_name,
            messages=[
                {"role": "system", "content": "You generate safe Python code for subtask execution."},
                {"role": "user", "content": prompt},
            ],
            temperature=TASK_ORCHESTRATOR_TEMPERATURE,
            top_p=TASK_ORCHESTRATOR_TOP_P,
        )
        code = strip_code_fence(response.choices[0].message.content or "")
        if code.strip():
            return code
    except Exception:
        pass

    safe_context = json.dumps(context_text or original_task_description, ensure_ascii=False)
    return (
        "import re\n"
        f"text = {safe_context}\n"
        "nums = re.findall(r'-?\\d+(?:\\.\\d+)?', text)\n"
        "result = nums[-1] if nums else ''\n"
        "print(result)\n"
    )


def infer_file_path(subtask) -> str:
    meta_path = (subtask.metadata or {}).get("path")
    if isinstance(meta_path, str) and meta_path.strip():
        return meta_path.strip()
    match = re.search(r"([A-Za-z]:\\\\[^\\s]+|[^\\s]+\\.(py|txt|md|json|csv))", subtask.description or "")
    if match:
        return match.group(1)
    return "main.py"


def extract_candidate_url(*texts: str) -> Optional[str]:
    url_re = re.compile(r"https?://[^\s'\"<>()]+", re.IGNORECASE)
    domain_re = re.compile(r"\b(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+[^\s'\"<>()]*")

    def clean(raw: str) -> str:
        return str(raw or "").strip().rstrip(".,;:!?)]}>\"'")

    for text in texts:
        match = url_re.search(str(text or ""))
        if match:
            return clean(match.group(0))

    for text in texts:
        match = domain_re.search(str(text or ""))
        if not match:
            continue
        candidate = clean(match.group(0))
        if "." not in candidate:
            continue
        if not candidate.lower().startswith(("http://", "https://")):
            candidate = "https://" + candidate
        return candidate
    return None


def select_action_and_input(
    llm,
    subtask,
    agent,
    original_task_description: str,
    step_outputs: Dict[str, str],
) -> Tuple[str, Dict[str, Any]]:
    skills_text = " ".join([str(x) for x in (subtask.required_skills or [])]).lower()
    context_text = "\n".join(
        [f"Dependency[{dep}]: {step_outputs.get(dep, '')[:300]}" for dep in subtask.dependencies]
    )

    if agent.category == AgentCategory.TOOL:
        if agent.agent_id == "pythonagent":
            return "execute_python", {
                "code": generate_python_code_for_subtask(
                    llm,
                    original_task_description=original_task_description,
                    subtask_description=subtask.description,
                    context_text=context_text,
                )
            }
        if agent.agent_id == "fileagent":
            return "read_file", {"path": infer_file_path(subtask), "encoding": "utf-8"}
        if agent.agent_id == "arxivagent":
            return "arxiv_search", {"query": subtask.description, "max_results": 5}
        if agent.agent_id == "websiteagent":
            subtask_meta = subtask.metadata or {}
            url = extract_candidate_url(
                str(subtask_meta.get("url", "") or ""),
                subtask.description,
                context_text,
                original_task_description,
            )
            payload: Dict[str, Any] = {
                "task": subtask.description,
                "context": context_text,
                "timeout": 20,
            }
            if url:
                payload["url"] = url
            return "extract_content", payload
        if agent.agent_id == "bingagent":
            return "bing_search", {"query": subtask.description, "max_results": 5}

        if any(keyword in skills_text for keyword in ["代码", "python", "code", "计算", "算法", "script"]):
            return "execute_python", {
                "code": generate_python_code_for_subtask(
                    llm,
                    original_task_description=original_task_description,
                    subtask_description=subtask.description,
                    context_text=context_text,
                )
            }
        if any(keyword in skills_text for keyword in ["文件", "file", "read", "写入", "目录"]):
            return "read_file", {"path": infer_file_path(subtask), "encoding": "utf-8"}
        if "arxiv" in skills_text:
            return "arxiv_search", {"query": subtask.description, "max_results": 5}
        return "bing_search", {"query": subtask.description, "max_results": 5}

    if any(keyword in skills_text for keyword in ["提取", "拆解", "分解", "decompose"]):
        return "decompose_question", {"question": subtask.description}
    if any(keyword in skills_text for keyword in ["验证", "检查", "批判", "critic"]):
        return "critical_analysis", {"content": f"{subtask.description}\n{context_text}", "criteria": []}
    if any(keyword in skills_text for keyword in ["总结", "摘要", "summarize"]):
        return "summarize", {"content": context_text or subtask.description, "max_length": 500}
    if any(keyword in skills_text for keyword in ["结论", "conclude"]):
        return "conclude", {"evidence": context_text or subtask.description, "context": {"deps": context_text}}

    return "logical_reasoning", {
        "problem": subtask.description,
        "context": {"dependency_outputs": context_text},
    }
