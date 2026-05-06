"""GSM-hard evaluation entrypoints."""

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
    extract_best_number_from_sources,
    extract_prediction_from_multi_result,
    is_correct_prediction,
)
from utils.dataset_utils import load_gsmhard_dataset
from utils.eval_visualization import generate_eval_charts


def evaluate_multi_agent_on_gsmhard(
    orchestrator_llm,
    dataset_path: str,
    sub_agent_model_name: str = SUB_AGENT_MODEL,
    limit: Optional[int] = None,
    offset: int = 0,
    save_path: str = "result/gsmhard_eval_output.json",
    compare_single_baseline: bool = False,
    single_llm=None,
    quiet_per_sample: bool = True,
) -> Dict[str, Any]:
    tasks = load_gsmhard_dataset(dataset_path, limit=limit, offset=offset)
    if not tasks:
        raise ValueError(f"No samples loaded from dataset: {dataset_path}")

    total = len(tasks)
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

    sample_results: List[Dict[str, Any]] = []
    print("\n" + "=" * 60)
    print("GSM-hard Dataset Evaluation (Multi-agent)")
    print("=" * 60)
    print(f"dataset_path: {dataset_path}")
    print(f"samples: {total}, offset: {max(0, offset)}, limit: {limit if limit is not None else 'all'}")
    print(f"orchestrator_model: {orchestrator_llm.model_name}")
    print(f"sub_agent_model: {sub_agent_model_name}")
    print(f"single_baseline: {'on' if compare_single_baseline else 'off'}")

    for i, item in enumerate(tasks, 1):
        idx = item.get("index", i - 1)
        question = item.get("input", "")
        target = item.get("target")

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
                        task_title=f"GSM_hard_{idx}_",
                        task_description=question,
                        sub_agent_model_name=sub_agent_model_name,
                    )
            else:
                multi_result = run_multi_agent_flow(
                    orchestrator_llm,
                    task_title=f"GSM_hard_{idx}_",
                    task_description=question,
                    sub_agent_model_name=sub_agent_model_name,
                )
            multi_elapsed = time.time() - start
            multi_result_text = str(multi_result.get("final_result", "") or "")
            multi_pred = extract_prediction_from_multi_result(multi_result)
            multi_tokens = int(multi_result.get("total_tokens", 0) or 0)
            multi_correct = is_correct_prediction(multi_pred, target)

            multi_stats["total_tokens"] += multi_tokens
            if multi_pred is not None:
                multi_stats["valid_predictions"] += 1
            if multi_correct:
                multi_stats["correct"] += 1
        except Exception as exc:
            multi_elapsed = time.time() - start
            multi_error = str(exc)
            multi_stats["failed_runs"] += 1
        multi_stats["total_elapsed_seconds"] += multi_elapsed

        single_entry = None
        if compare_single_baseline:
            if single_llm is None:
                raise ValueError("single_llm is required when compare_single_baseline=True")
            single_error = ""
            single_pred = None
            single_tokens = 0
            single_elapsed = 0.0
            single_answer_text = ""
            single_correct = False
            single_start = time.time()
            try:
                single_result = solve_with_single_llm(single_llm, f"GSM_hard_{idx}_", question)
                single_elapsed = time.time() - single_start
                single_answer_text = str(single_result.get("answer", "") or "")
                single_pred = extract_best_number_from_sources(
                    [("single_answer", single_answer_text, 2.0)]
                )
                single_tokens = int(
                    (single_result.get("tokens_used", {}) or {}).get("total_tokens", 0) or 0
                )
                single_correct = is_correct_prediction(single_pred, target)
                single_stats["total_tokens"] += single_tokens
                if single_pred is not None:
                    single_stats["valid_predictions"] += 1
                if single_correct:
                    single_stats["correct"] += 1
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
                "dataset_index": idx,
                "target": target,
                "question": question,
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
            f"[{i}/{total}] idx={idx} "
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

    summary: Dict[str, Any] = {
        "run_at": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "config": {
            "offset": max(0, offset),
            "limit": limit,
            "orchestrator_model": orchestrator_llm.model_name,
            "sub_agent_model": sub_agent_model_name,
            "single_baseline_model": single_llm.model_name if single_llm else None,
            "quiet_per_sample": quiet_per_sample,
        },
        "multi_agent_metrics": finalize_stats(multi_stats),
        "single_baseline_metrics": finalize_stats(single_stats) if compare_single_baseline else None,
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
