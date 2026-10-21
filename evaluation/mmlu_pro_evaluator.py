"""MMLU-Pro evaluation entrypoints."""

import contextlib
import io
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional

from llm.llm_config import SUB_AGENT_MODEL
from pipelines.multi_agent_pipeline import run_multi_agent_flow
from semantic_matcher import SemanticMatcher
from models import SchedulerConfig
from services.single_llm_solver import solve_with_single_llm
from services.single_tool_solver import solve_with_single_tool_llm
from utils.answer_utils import (
    is_correct_option_prediction,
)
from utils.dataset_utils import load_mmlu_pro_dataset, load_mmlu_pro_splits
from utils.eval_visualization import generate_eval_charts

from evaluation.runtime import (
    experiment_metadata, prepare_checkpoint, attach_multi_trace, attach_single_usage,
    append_checkpoint_sample,
    build_summary,
    dedupe_samples,
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


def stratified_mmlu_samples(samples, limit=None, offset=0, seed=0):
    """Build a reproducible category-balanced sequence before slicing."""
    groups = {}
    for sample in samples:
        groups.setdefault(str(sample.get("category", "unknown")), []).append(sample)
    rng = random.Random(seed)
    categories = sorted(groups)
    rng.shuffle(categories)
    for rows in groups.values():
        rng.shuffle(rows)
    ordered = []
    while categories:
        next_categories = []
        for category in categories:
            rows = groups[category]
            if rows:
                ordered.append(rows.pop())
            if rows:
                next_categories.append(category)
        categories = next_categories
    start = max(0, offset)
    return ordered[start:] if limit is None else ordered[start:start + max(0, limit)]


def extract_choice_prediction(text, option_count):
    """Require an explicit answer, never a letter mentioned only in reasoning."""
    text = str(text or "").strip()
    direct = re.fullmatch(r"\(?([A-J])\)?[.!]?", text, re.IGNORECASE)
    if direct:
        letter = direct.group(1).upper()
    else:
        matches = re.findall(
            r"(?:\b(?:the\s+)?answer\s+is|\bfinal\s+answer\s*[:：]|答案\s*[:：])"
            r"\s*\(?([A-J])\)?(?=\s*(?:[.!。]|$))", text, re.IGNORECASE)
        if not matches or len({m.upper() for m in matches}) != 1:
            return None
        letter = matches[-1].upper()
    return letter if letter in OPTION_LABELS[:option_count] else None


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


def _solve_mmlu_multiple_choice_with_single_llm(
    llm, sample: Dict[str, Any], category_prefix: str = "", scheduler_config=None,
    single_baseline_mode: str = "plain",
) -> Dict[str, Any]:
    task_description = _build_mmlu_task_description(sample, category_prefix=category_prefix)
    solver = solve_with_single_tool_llm if single_baseline_mode == "python" else solve_with_single_llm
    return solver(
        llm,
        task_title=f"MMLU_Pro_{sample.get('question_id', 'unknown')}",
        task_description=task_description,
        scheduler_config=scheduler_config,
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
    scheduler_config=None,
    sample_strategy: str = "sequential",
    single_baseline_mode: str = "plain",
) -> Dict[str, Any]:
    run_kind = normalize_run_kind(run_kind, compare_single_baseline=compare_single_baseline)
    scheduler_config = scheduler_config or SchedulerConfig()
    if single_baseline_mode not in {"plain", "python"}:
        raise ValueError("single_baseline_mode must be plain or python")
    include_multi = should_run_multi(run_kind)
    include_single = should_run_single(run_kind)
    if include_single and single_llm is None:
        raise ValueError("single_llm is required when run_kind includes single")

    splits = load_mmlu_pro_splits(dataset_path)
    if sample_strategy not in {"sequential", "stratified"}:
        raise ValueError("sample_strategy must be sequential or stratified")
    if sample_strategy == "stratified":
        test_samples = stratified_mmlu_samples(
            splits.get("test", []), limit=limit, offset=offset, seed=scheduler_config.seed)
    else:
        test_samples = load_mmlu_pro_dataset(dataset_path, split="test", limit=limit, offset=offset)
    validation_samples = splits.get("validation", [])
    prompts_by_category = _build_category_prefix(validation_samples, shots_per_category) if validation_samples else {}

    if not test_samples:
        raise ValueError(f"No MMLU-Pro test samples loaded from dataset: {dataset_path}")

    metadata = experiment_metadata(scheduler_config, orchestrator_llm, sub_agent_model_name, single_llm,
                                   benchmark="mmlu-pro", dataset_path=dataset_path, run_kind=run_kind,
                                   shots_per_category=shots_per_category,
                                   sample_strategy=sample_strategy,
                                   single_baseline_mode=single_baseline_mode,
                                   category_prompts=prompts_by_category)
    checkpoint_samples, loaded_keys = prepare_checkpoint(checkpoint_path, "mmlu-pro", resume, metadata)
    selected_keys = {str(item.get("question_id")) for item in test_samples}
    checkpoint_samples = [s for s in checkpoint_samples if sample_key(s, "mmlu-pro") in selected_keys]
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
    print(f"samples: {total}, offset: {max(0, offset)}, limit: {limit if limit is not None else 'all'}, strategy: {sample_strategy}")
    print(f"run_kind: {run_kind}, workers: {max(1, workers)}")
    print(f"orchestrator_model: {orchestrator_llm.model_name}")
    print(f"sub_agent_model: {sub_agent_model_name}")
    print(f"single_baseline_model: {single_llm.model_name if include_single else None}")
    print(f"single_baseline_mode: {single_baseline_mode if include_single else None}")
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
                    "scheduler_config": scheduler_config,
                }
                if suppress_stdout:
                    with contextlib.redirect_stdout(io.StringIO()):
                        multi_result = run_multi_agent_flow(orchestrator_llm, **kwargs)
                else:
                    multi_result = run_multi_agent_flow(orchestrator_llm, **kwargs)
                multi_entry["elapsed_seconds"] = time.time() - start
                multi_entry["final_result"] = str(multi_result.get("final_result", "") or "")[:1200]
                multi_entry["prediction"] = extract_choice_prediction(
                    multi_result.get("final_result", ""), len(item.get("options", [])))
                multi_entry["tokens"] = int(multi_result.get("total_tokens", 0) or 0)
                attach_multi_trace(multi_entry, multi_result)
                multi_entry["parse_failed"] = multi_entry["prediction"] is None
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
                    scheduler_config=scheduler_config,
                    single_baseline_mode=single_baseline_mode,
                )
                single_entry["elapsed_seconds"] = time.time() - single_start
                single_entry["answer_text"] = str(single_result.get("answer", "") or "")
                single_entry["prediction"] = extract_choice_prediction(
                    single_entry["answer_text"], len(item.get("options", [])))
                single_entry["parse_failed"] = single_entry["prediction"] is None
                single_entry["tokens"] = int(
                    (single_result.get("tokens_used", {}) or {}).get("total_tokens", 0) or 0
                )
                single_entry["is_correct"] = is_correct_option_prediction(single_entry["prediction"], target)
                single_entry["baseline_mode"] = single_baseline_mode
                attach_single_usage(single_entry, single_result)
            except Exception as exc:
                single_entry["elapsed_seconds"] = time.time() - single_start
                single_entry["error"] = str(exc)
                single_entry["baseline_mode"] = single_baseline_mode
                attach_single_usage(single_entry, error=exc)

        return {
            "experiment": metadata,
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
            "experiment": metadata,
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
            "single_baseline_mode": single_baseline_mode if include_single else None,
            "quiet_per_sample": quiet_per_sample,
            "shots_per_category": shots_per_category,
            "sample_strategy": sample_strategy,
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
