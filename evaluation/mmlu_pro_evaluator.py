"""MMLU-Pro evaluation entrypoints."""

import contextlib
from datetime import datetime
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

from llm.llm_config import SUB_AGENT_MODEL
from pipelines.multi_agent_pipeline import run_multi_agent_flow
from services.single_llm_solver import solve_with_single_llm
from utils.answer_utils import (
    extract_option_letter,
    extract_option_prediction_from_multi_result,
    is_correct_option_prediction,
    random_option_fallback,
)
from utils.dataset_utils import load_mmlu_pro_dataset, load_mmlu_pro_splits
from utils.eval_visualization import generate_eval_charts


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
) -> Dict[str, Any]:
    splits = load_mmlu_pro_splits(dataset_path)
    test_samples = load_mmlu_pro_dataset(dataset_path, split="test", limit=limit, offset=offset)
    validation_samples = splits.get("validation", [])
    prompts_by_category = _build_category_prefix(validation_samples, shots_per_category) if validation_samples else {}

    if not test_samples:
        raise ValueError(f"No MMLU-Pro test samples loaded from dataset: {dataset_path}")

    total = len(test_samples)
    multi_stats = {
        "total": total,
        "correct": 0,
        "valid_predictions": 0,
        "failed_runs": 0,
        "total_tokens": 0,
        "total_elapsed_seconds": 0.0,
    }
    single_stats = {
        "total": total,
        "correct": 0,
        "valid_predictions": 0,
        "failed_runs": 0,
        "total_tokens": 0,
        "total_elapsed_seconds": 0.0,
    }
    per_category_multi: Dict[str, Dict[str, int]] = {}
    per_category_single: Dict[str, Dict[str, int]] = {}

    sample_results: List[Dict[str, Any]] = []
    print("\n" + "=" * 60)
    print("MMLU-Pro Dataset Evaluation (Multi-agent)")
    print("=" * 60)
    print(f"dataset_path: {dataset_path}")
    print(f"samples: {total}, offset: {max(0, offset)}, limit: {limit if limit is not None else 'all'}")
    print(f"orchestrator_model: {orchestrator_llm.model_name}")
    print(f"sub_agent_model: {sub_agent_model_name}")
    print(f"single_baseline: {'on' if compare_single_baseline else 'off'}")
    print(f"few-shot validation examples per category: {shots_per_category if validation_samples else 0}")

    for i, item in enumerate(test_samples, 1):
        question_id = item.get("question_id", i - 1)
        category = str(item.get("category", "unknown"))
        target = str(item.get("answer", "") or "").strip().upper()
        category_prefix = prompts_by_category.get(category, "")
        task_description = _build_mmlu_task_description(item, category_prefix=category_prefix)

        per_category_multi.setdefault(category, {"correct": 0, "total": 0})
        per_category_multi[category]["total"] += 1

        multi_error = ""
        multi_result_text = ""
        multi_pred = None
        multi_tokens = 0
        multi_elapsed = 0.0
        multi_correct = False

        start = time.time()
        try:
            if quiet_per_sample:
                with contextlib.redirect_stdout(io.StringIO()):
                    multi_result = run_multi_agent_flow(
                        orchestrator_llm,
                        task_title=f"MMLU_Pro_{question_id}",
                        task_description=task_description,
                        sub_agent_model_name=sub_agent_model_name,
                        final_summary_instruction=(
                            "You are responsible for final answer extraction. "
                            "This is a multiple-choice benchmark question. "
                            "Return strict JSON with status/final_answer/result_text/key_facts/confidence. "
                            "final_answer must be exactly one option letter from A to J, with no extra text."
                        ),
                    )
            else:
                multi_result = run_multi_agent_flow(
                    orchestrator_llm,
                    task_title=f"MMLU_Pro_{question_id}",
                    task_description=task_description,
                    sub_agent_model_name=sub_agent_model_name,
                    final_summary_instruction=(
                        "You are responsible for final answer extraction. "
                        "This is a multiple-choice benchmark question. "
                        "Return strict JSON with status/final_answer/result_text/key_facts/confidence. "
                        "final_answer must be exactly one option letter from A to J, with no extra text."
                    ),
                )
            multi_elapsed = time.time() - start
            multi_result_text = str(multi_result.get("final_result", "") or "")
            multi_pred = extract_option_prediction_from_multi_result(multi_result)
            multi_tokens = int(multi_result.get("total_tokens", 0) or 0)
            multi_correct = is_correct_option_prediction(multi_pred, target)

            multi_stats["total_tokens"] += multi_tokens
            if multi_pred is not None:
                multi_stats["valid_predictions"] += 1
            if multi_correct:
                multi_stats["correct"] += 1
                per_category_multi[category]["correct"] += 1
        except Exception as exc:
            multi_elapsed = time.time() - start
            multi_error = str(exc)
            multi_stats["failed_runs"] += 1
        multi_stats["total_elapsed_seconds"] += multi_elapsed

        single_entry = None
        if compare_single_baseline:
            if single_llm is None:
                raise ValueError("single_llm is required when compare_single_baseline=True")
            per_category_single.setdefault(category, {"correct": 0, "total": 0})
            per_category_single[category]["total"] += 1

            single_error = ""
            single_pred = None
            single_tokens = 0
            single_elapsed = 0.0
            single_answer_text = ""
            single_correct = False
            single_start = time.time()
            try:
                single_result = _solve_mmlu_multiple_choice_with_single_llm(
                    single_llm,
                    item,
                    category_prefix=category_prefix,
                )
                single_elapsed = time.time() - single_start
                single_answer_text = str(single_result.get("answer", "") or "")
                single_pred = extract_option_letter(single_answer_text)
                if single_pred is None:
                    single_pred = random_option_fallback(len(list(item.get("options", []) or [])))
                single_tokens = int(
                    (single_result.get("tokens_used", {}) or {}).get("total_tokens", 0) or 0
                )
                single_correct = is_correct_option_prediction(single_pred, target)
                single_stats["total_tokens"] += single_tokens
                if single_pred is not None:
                    single_stats["valid_predictions"] += 1
                if single_correct:
                    single_stats["correct"] += 1
                    per_category_single[category]["correct"] += 1
            except Exception as exc:
                single_elapsed = time.time() - single_start
                single_error = str(exc)
                single_stats["failed_runs"] += 1
            single_stats["total_elapsed_seconds"] += single_elapsed
            single_entry = {
                "prediction": single_pred,
                "is_correct": single_correct,
                "tokens": single_tokens,
                "elapsed_seconds": single_elapsed,
                "error": single_error,
                "answer_text": single_answer_text[:1000],
            }

        sample_results.append(
            {
                "question_id": question_id,
                "category": category,
                "target": target,
                "question": item.get("question", ""),
                "options": list(item.get("options", []) or []),
                "multi_agent": {
                    "prediction": multi_pred,
                    "is_correct": multi_correct,
                    "tokens": multi_tokens,
                    "elapsed_seconds": multi_elapsed,
                    "error": multi_error,
                    "final_result": multi_result_text[:1200],
                },
                "single_baseline": single_entry,
            }
        )

        accuracy = multi_stats["correct"] / i
        print(
            f"[{i}/{total}] qid={question_id} category={category} "
            f"target={target} pred={multi_pred} correct={int(multi_correct)} "
            f"acc={accuracy:.4f} token={multi_tokens} time={multi_elapsed:.2f}s"
        )

    def finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        n = max(1, stats["total"])
        return {
            **stats,
            "accuracy": stats["correct"] / n,
            "extraction_rate": stats["valid_predictions"] / n,
            "avg_tokens": stats["total_tokens"] / n,
            "avg_elapsed_seconds": stats["total_elapsed_seconds"] / n,
        }

    def finalize_per_category(raw: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for category, stats in raw.items():
            total_count = max(1, stats["total"])
            result[category] = {
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / total_count,
            }
        return result

    summary: Dict[str, Any] = {
        "run_at": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "benchmark": "mmlu-pro",
        "config": {
            "offset": max(0, offset),
            "limit": limit,
            "orchestrator_model": orchestrator_llm.model_name,
            "sub_agent_model": sub_agent_model_name,
            "single_baseline_model": single_llm.model_name if single_llm else None,
            "quiet_per_sample": quiet_per_sample,
            "shots_per_category": shots_per_category,
            "validation_available": bool(validation_samples),
        },
        "multi_agent_metrics": finalize_stats(multi_stats),
        "single_baseline_metrics": finalize_stats(single_stats) if compare_single_baseline else None,
        "multi_agent_per_category": finalize_per_category(per_category_multi),
        "single_baseline_per_category": finalize_per_category(per_category_single) if compare_single_baseline else None,
        "samples": sample_results,
    }

    visualization = generate_eval_charts(summary, save_path=save_path)
    summary["visualization"] = visualization

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(
        "multi-agent: "
        f"acc={summary['multi_agent_metrics']['accuracy']:.4f}, "
        f"extract_rate={summary['multi_agent_metrics']['extraction_rate']:.4f}, "
        f"avg_tokens={summary['multi_agent_metrics']['avg_tokens']:.2f}, "
        f"avg_time={summary['multi_agent_metrics']['avg_elapsed_seconds']:.2f}s"
    )
    if compare_single_baseline and summary.get("single_baseline_metrics"):
        single_metrics = summary["single_baseline_metrics"]
        print(
            "single-baseline: "
            f"acc={single_metrics['accuracy']:.4f}, "
            f"extract_rate={single_metrics['extraction_rate']:.4f}, "
            f"avg_tokens={single_metrics['avg_tokens']:.2f}, "
            f"avg_time={single_metrics['avg_elapsed_seconds']:.2f}s"
        )
    if visualization.get("enabled"):
        print(
            f"charts saved to: {visualization.get('chart_dir')} "
            f"(backend={visualization.get('backend')}, files={len(visualization.get('files', []))})"
        )
    else:
        print(f"charts skipped: {visualization.get('error')}")
    print(f"saved to: {save_path}")
    return summary
