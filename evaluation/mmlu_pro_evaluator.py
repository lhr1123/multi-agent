"""MMLU-Pro evaluation entrypoints."""

import contextlib
import io
import threading
import time
from typing import Any, Dict, List, Optional

from llm.llm_config import SUB_AGENT_MODEL
from pipelines.multi_agent_pipeline import run_multi_agent_flow
from semantic_matcher import SemanticMatcher
from services.single_llm_solver import solve_with_single_llm
from utils.answer_utils import (
    extract_option_letter,
    extract_option_prediction_from_multi_result,
    is_correct_option_prediction,
    random_option_fallback,
)
from utils.dataset_utils import load_mmlu_pro_dataset, load_mmlu_pro_splits
from utils.eval_visualization import generate_eval_charts

from evaluation.runtime import (
    append_checkpoint_sample,
    build_summary,
    dedupe_samples,
    load_checkpoint_samples,
    normalize_run_kind,
    print_summary,
    run_items,
    sample_key,
    should_run_multi,
    should_run_single,
    sort_samples,
    write_summary_json,
)


OPTION_LABELS = "ABCDEFGHIJ"


def _format_options(options: List[str]) -> str:
    lines = ["Options:"]
    for idx, option in enumerate(options):
        if idx >= len(OPTION_LABELS):
            break
        lines.append(f"({OPTION_LABELS[idx]}) {option}")
    return "\n".join(lines)


def _build_category_prefix(validation_samples: List[Dict[str, Any]], shots_per_category: int) -> Dict[str, str]:
    grouped: Dict[str, List[str]] = {}
    for sample in validation_samples:
        category = str(sample.get("category", "unknown"))
        grouped.setdefault(category, [])
        if len(grouped[category]) >= shots_per_category:
            continue
        cot_content = str(sample.get("cot_content", "") or "").strip()
        if not cot_content:
            continue
        grouped[category].append(
            "\n".join(
                [
                    f"Q: {sample.get('question', '')}",
                    _format_options(list(sample.get("options", []) or [])),
                    cot_content,
                ]
            )
        )
    return {category: "\n\n".join(examples) for category, examples in grouped.items()}


def _build_mmlu_task_description(
    sample: Dict[str, Any],
    category_prefix: str = "",
) -> str:
    instructions = (
        "You are solving an MMLU-Pro multiple-choice question.\n"
        "Reason carefully over the options.\n"
        "Your final answer must be a single option letter from A to J.\n"
        "Return the final answer in the exact form: The answer is (X)\n"
    )
    prompt_parts = [instructions]
    if category_prefix:
        prompt_parts.append("Here are category-specific solved examples:")
        prompt_parts.append(category_prefix)
    prompt_parts.append(f"Question Category: {sample.get('category', 'unknown')}")
    prompt_parts.append(f"Question: {sample.get('question', '')}")
    prompt_parts.append(_format_options(list(sample.get("options", []) or [])))
    return "\n\n".join(prompt_parts)


def _solve_mmlu_multiple_choice_with_single_llm(llm, sample: Dict[str, Any], category_prefix: str = "") -> Dict[str, Any]:
    task_description = _build_mmlu_task_description(sample, category_prefix=category_prefix)
    return solve_with_single_llm(
        llm,
        task_title=f"MMLU_Pro_{sample.get('question_id', 'unknown')}",
        task_description=task_description,
        system_prompt="You are an expert multiple-choice problem solver.",
        output_schema=(
            '{\n'
            '  "answer": "The answer is (X)",\n'
            '  "reasoning": "brief reasoning"\n'
            '}'
        ),
    )


def _empty_multi_entry() -> Dict[str, Any]:
    return {
        "prediction": None,
        "is_correct": False,
        "tokens": 0,
        "elapsed_seconds": 0.0,
        "error": "",
        "final_result": "",
    }


def _empty_single_entry() -> Dict[str, Any]:
    return {
        "prediction": None,
        "is_correct": False,
        "tokens": 0,
        "elapsed_seconds": 0.0,
        "error": "",
        "answer_text": "",
    }


def _build_per_category(samples: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, float]]:
    raw: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        entry = sample.get(field)
        if not entry:
            continue
        category = str(sample.get("category", "unknown"))
        raw.setdefault(category, {"correct": 0, "total": 0})
        raw[category]["total"] += 1
        if entry.get("is_correct"):
            raw[category]["correct"] += 1

    result: Dict[str, Dict[str, float]] = {}
    for category, stats in raw.items():
        total_count = max(1, stats["total"])
        result[category] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": stats["correct"] / total_count,
        }
    return result


def evaluate_multi_agent_on_mmlu_pro(
    orchestrator_llm,
    dataset_path: str,
    sub_agent_model_name: str = SUB_AGENT_MODEL,
    limit: Optional[int] = None,
    offset: int = 0,
    save_path: str = "result/mmlu_pro_eval_output.json",
    compare_single_baseline: bool = False,
    single_llm=None,
    quiet_per_sample: bool = True,
    shots_per_category: int = 5,
    run_kind: str = "multi",
    workers: int = 1,
    checkpoint_path: Optional[str] = None,
    resume: bool = False,
    charts_during_run: bool = True,
) -> Dict[str, Any]:
    run_kind = normalize_run_kind(run_kind, compare_single_baseline=compare_single_baseline)
    include_multi = should_run_multi(run_kind)
    include_single = should_run_single(run_kind)
    if include_single and single_llm is None:
        raise ValueError("single_llm is required when run_kind includes single")

    splits = load_mmlu_pro_splits(dataset_path)
    test_samples = load_mmlu_pro_dataset(dataset_path, split="test", limit=limit, offset=offset)
    validation_samples = splits.get("validation", [])
    prompts_by_category = _build_category_prefix(validation_samples, shots_per_category) if validation_samples else {}

    if not test_samples:
        raise ValueError(f"No MMLU-Pro test samples loaded from dataset: {dataset_path}")

    checkpoint_samples, loaded_keys = load_checkpoint_samples(checkpoint_path, "mmlu-pro") if resume else ([], set())
    completed_keys = {
        sample_key(sample, "mmlu-pro")
        for sample in checkpoint_samples
        if ((not include_multi or sample.get("multi_agent")) and (not include_single or sample.get("single_baseline")))
    }
    pending_samples = [item for item in test_samples if sample_key({"question_id": item.get("question_id")}, "mmlu-pro") not in completed_keys]
    total = len(test_samples)
    checkpoint_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_this_run = 0
    semantic_matcher = SemanticMatcher() if include_multi else None
    suppress_stdout = quiet_per_sample and workers <= 1

    print("\n" + "=" * 60)
    print("MMLU-Pro Dataset Evaluation")
    print("=" * 60)
    print(f"dataset_path: {dataset_path}")
    print(f"samples: {total}, offset: {max(0, offset)}, limit: {limit if limit is not None else 'all'}")
    print(f"run_kind: {run_kind}, workers: {max(1, workers)}")
    print(f"orchestrator_model: {orchestrator_llm.model_name}")
    print(f"sub_agent_model: {sub_agent_model_name}")
    print(f"single_baseline_model: {single_llm.model_name if include_single else None}")
    print(f"checkpoint_path: {checkpoint_path or 'off'}, resume: {resume}, loaded: {len(loaded_keys)}, skipped: {len(completed_keys)}")
    print(f"few-shot validation examples per category: {shots_per_category if validation_samples else 0}")

    def evaluate_one(item: Dict[str, Any]) -> Dict[str, Any]:
        question_id = item.get("question_id")
        category = str(item.get("category", "unknown"))
        target = str(item.get("answer", "") or "").strip().upper()
        category_prefix = prompts_by_category.get(category, "")
        task_description = _build_mmlu_task_description(item, category_prefix=category_prefix)

        multi_entry = None
        if include_multi:
            multi_entry = _empty_multi_entry()
            start = time.time()
            try:
                kwargs = {
                    "task_title": f"MMLU_Pro_{question_id}",
                    "task_description": task_description,
                    "sub_agent_model_name": sub_agent_model_name,
                    "final_summary_instruction": (
                        "You are responsible for final answer extraction. "
                        "This is a multiple-choice benchmark question. "
                        "Return strict JSON with status/final_answer/result_text/key_facts/confidence. "
                        "final_answer must be exactly one option letter from A to J, with no extra text."
                    ),
                    "semantic_matcher": semantic_matcher,
                }
                if suppress_stdout:
                    with contextlib.redirect_stdout(io.StringIO()):
                        multi_result = run_multi_agent_flow(orchestrator_llm, **kwargs)
                else:
                    multi_result = run_multi_agent_flow(orchestrator_llm, **kwargs)
                multi_entry["elapsed_seconds"] = time.time() - start
                multi_entry["final_result"] = str(multi_result.get("final_result", "") or "")[:1200]
                multi_entry["prediction"] = extract_option_prediction_from_multi_result(multi_result)
                multi_entry["tokens"] = int(multi_result.get("total_tokens", 0) or 0)
                multi_entry["is_correct"] = is_correct_option_prediction(multi_entry["prediction"], target)
            except Exception as exc:
                multi_entry["elapsed_seconds"] = time.time() - start
                multi_entry["error"] = str(exc)

        single_entry = None
        if include_single:
            single_entry = _empty_single_entry()
            single_start = time.time()
            try:
                single_result = _solve_mmlu_multiple_choice_with_single_llm(
                    single_llm,
                    item,
                    category_prefix=category_prefix,
                )
                single_entry["elapsed_seconds"] = time.time() - single_start
                single_entry["answer_text"] = str(single_result.get("answer", "") or "")[:1000]
                single_entry["prediction"] = extract_option_letter(single_entry["answer_text"])
                if single_entry["prediction"] is None:
                    single_entry["prediction"] = random_option_fallback(len(list(item.get("options", []) or [])))
                single_entry["tokens"] = int(
                    (single_result.get("tokens_used", {}) or {}).get("total_tokens", 0) or 0
                )
                single_entry["is_correct"] = is_correct_option_prediction(single_entry["prediction"], target)
            except Exception as exc:
                single_entry["elapsed_seconds"] = time.time() - single_start
                single_entry["error"] = str(exc)

        return {
            "question_id": question_id,
            "category": category,
            "target": target,
            "question": item.get("question", ""),
            "options": list(item.get("options", []) or []),
            "multi_agent": multi_entry,
            "single_baseline": single_entry,
        }

    def on_complete(sample: Dict[str, Any]) -> None:
        nonlocal completed_this_run
        append_checkpoint_sample(checkpoint_path, sample, checkpoint_lock)
        with progress_lock:
            completed_this_run += 1
            multi = sample.get("multi_agent") or {}
            single = sample.get("single_baseline") or {}
            active = multi if include_multi else single
            done = len(completed_keys) + completed_this_run
            print(
                f"[{done}/{total}] qid={sample.get('question_id')} category={sample.get('category')} "
                f"target={sample.get('target')} pred={active.get('prediction')} "
                f"correct={int(bool(active.get('is_correct')))} token={active.get('tokens', 0)} "
                f"time={float(active.get('elapsed_seconds', 0.0) or 0.0):.2f}s"
            )

    new_samples = run_items(pending_samples, evaluate_one, max(1, workers), on_complete)
    sample_results = dedupe_samples([*checkpoint_samples, *new_samples], "mmlu-pro")
    sample_results = sort_samples(sample_results, "mmlu-pro")

    summary = build_summary(
        benchmark="mmlu-pro",
        dataset_path=dataset_path,
        samples=sample_results,
        config={
            "offset": max(0, offset),
            "limit": limit,
            "run_kind": run_kind,
            "workers": max(1, workers),
            "checkpoint_path": checkpoint_path,
            "resume": resume,
            "loaded_from_checkpoint": len(loaded_keys),
            "skipped_from_checkpoint": len(completed_keys),
            "orchestrator_model": orchestrator_llm.model_name,
            "sub_agent_model": sub_agent_model_name,
            "single_baseline_model": single_llm.model_name if include_single else None,
            "quiet_per_sample": quiet_per_sample,
            "shots_per_category": shots_per_category,
            "validation_available": bool(validation_samples),
        },
        include_multi=include_multi,
        include_single=include_single,
        extra_fields={
            "multi_agent_per_category": _build_per_category(sample_results, "multi_agent") if include_multi else None,
            "single_baseline_per_category": _build_per_category(sample_results, "single_baseline") if include_single else None,
        },
    )

    visualization = generate_eval_charts(summary, save_path=save_path) if charts_during_run else {"enabled": False, "error": "disabled during run"}
    summary["visualization"] = visualization
    write_summary_json(summary, save_path)
    print_summary(summary, visualization, save_path)
    return summary
