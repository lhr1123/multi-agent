"""GSM-hard evaluation entrypoints."""

import contextlib
import io
import threading
import time
from typing import Any, Dict, Optional

from llm.llm_config import SUB_AGENT_MODEL
from pipelines.multi_agent_pipeline import run_multi_agent_flow
from semantic_matcher import SemanticMatcher
from services.single_llm_solver import solve_with_single_llm
from utils.answer_utils import (
    extract_best_number_from_sources,
    extract_prediction_from_multi_result,
    is_correct_prediction,
)
from utils.dataset_utils import load_gsmhard_dataset
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

    tasks = load_gsmhard_dataset(dataset_path, limit=limit, offset=offset)
    if not tasks:
        raise ValueError(f"No samples loaded from dataset: {dataset_path}")

    checkpoint_samples, loaded_keys = load_checkpoint_samples(checkpoint_path, "gsm-hard") if resume else ([], set())
    completed_keys = {
        sample_key(sample, "gsm-hard")
        for sample in checkpoint_samples
        if ((not include_multi or sample.get("multi_agent")) and (not include_single or sample.get("single_baseline")))
    }
    pending_tasks = [item for item in tasks if sample_key({"dataset_index": item.get("index")}, "gsm-hard") not in completed_keys]
    total = len(tasks)
    checkpoint_lock = threading.Lock()
    progress_lock = threading.Lock()
    completed_this_run = 0
    semantic_matcher = SemanticMatcher() if include_multi else None
    suppress_stdout = quiet_per_sample and workers <= 1

    print("\n" + "=" * 60)
    print("GSM-hard Dataset Evaluation")
    print("=" * 60)
    print(f"dataset_path: {dataset_path}")
    print(f"samples: {total}, offset: {max(0, offset)}, limit: {limit if limit is not None else 'all'}")
    print(f"run_kind: {run_kind}, workers: {max(1, workers)}")
    print(f"orchestrator_model: {orchestrator_llm.model_name}")
    print(f"sub_agent_model: {sub_agent_model_name}")
    print(f"single_baseline_model: {single_llm.model_name if include_single else None}")
    print(f"checkpoint_path: {checkpoint_path or 'off'}, resume: {resume}, loaded: {len(loaded_keys)}, skipped: {len(completed_keys)}")

    def evaluate_one(item: Dict[str, Any]) -> Dict[str, Any]:
        idx = item.get("index")
        question = item.get("input", "")
        target = item.get("target")

        multi_entry = None
        if include_multi:
            multi_entry = _empty_multi_entry()
            start = time.time()
            try:
                kwargs = {
                    "task_title": f"GSM_hard_{idx}_",
                    "task_description": question,
                    "sub_agent_model_name": sub_agent_model_name,
                    "semantic_matcher": semantic_matcher,
                }
                if suppress_stdout:
                    with contextlib.redirect_stdout(io.StringIO()):
                        multi_result = run_multi_agent_flow(orchestrator_llm, **kwargs)
                else:
                    multi_result = run_multi_agent_flow(orchestrator_llm, **kwargs)
                multi_entry["elapsed_seconds"] = time.time() - start
                multi_entry["final_result"] = str(multi_result.get("final_result", "") or "")[:1200]
                multi_entry["prediction"] = extract_prediction_from_multi_result(multi_result)
                multi_entry["tokens"] = int(multi_result.get("total_tokens", 0) or 0)
                multi_entry["is_correct"] = is_correct_prediction(multi_entry["prediction"], target)
            except Exception as exc:
                multi_entry["elapsed_seconds"] = time.time() - start
                multi_entry["error"] = str(exc)

        single_entry = None
        if include_single:
            single_entry = _empty_single_entry()
            single_start = time.time()
            try:
                single_result = solve_with_single_llm(single_llm, f"GSM_hard_{idx}_", question)
                single_entry["elapsed_seconds"] = time.time() - single_start
                single_entry["answer_text"] = str(single_result.get("answer", "") or "")[:1000]
                single_entry["prediction"] = extract_best_number_from_sources(
                    [("single_answer", single_entry["answer_text"], 2.0)]
                )
                single_entry["tokens"] = int(
                    (single_result.get("tokens_used", {}) or {}).get("total_tokens", 0) or 0
                )
                single_entry["is_correct"] = is_correct_prediction(single_entry["prediction"], target)
            except Exception as exc:
                single_entry["elapsed_seconds"] = time.time() - single_start
                single_entry["error"] = str(exc)

        return {
            "dataset_index": idx,
            "target": target,
            "question": question,
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
                f"[{done}/{total}] idx={sample.get('dataset_index')} "
                f"target={sample.get('target')} pred={active.get('prediction')} "
                f"correct={int(bool(active.get('is_correct')))} token={active.get('tokens', 0)} "
                f"time={float(active.get('elapsed_seconds', 0.0) or 0.0):.2f}s"
            )

    new_samples = run_items(pending_tasks, evaluate_one, max(1, workers), on_complete)
    sample_results = dedupe_samples([*checkpoint_samples, *new_samples], "gsm-hard")
    sample_results = sort_samples(sample_results, "gsm-hard")

    summary = build_summary(
        benchmark="gsm-hard",
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
        },
        include_multi=include_multi,
        include_single=include_single,
    )

    visualization = generate_eval_charts(summary, save_path=save_path) if charts_during_run else {"enabled": False, "error": "disabled during run"}
    summary["visualization"] = visualization
    write_summary_json(summary, save_path)
    print_summary(summary, visualization, save_path)
    return summary
