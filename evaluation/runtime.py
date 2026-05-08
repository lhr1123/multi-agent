"""Shared helpers for resumable and parallel dataset evaluation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


RUN_KINDS = {"multi", "single", "both"}


def normalize_run_kind(run_kind: str, compare_single_baseline: bool = False) -> str:
    """Keep old --compare-single behavior while allowing explicit run-kind."""
    value = (run_kind or "multi").strip().lower()
    if value not in RUN_KINDS:
        raise ValueError(f"run_kind must be one of {sorted(RUN_KINDS)}, got: {run_kind}")
    if compare_single_baseline and value == "multi":
        return "both"
    return value


def should_run_multi(run_kind: str) -> bool:
    return run_kind in {"multi", "both"}


def should_run_single(run_kind: str) -> bool:
    return run_kind in {"single", "both"}


def empty_stats() -> Dict[str, Any]:
    return {
        "total": 0,
        "correct": 0,
        "valid_predictions": 0,
        "failed_runs": 0,
        "total_tokens": 0,
        "total_elapsed_seconds": 0.0,
    }


def update_stats(stats: Dict[str, Any], entry: Optional[Dict[str, Any]]) -> None:
    if not entry:
        return
    stats["total"] += 1
    if entry.get("is_correct"):
        stats["correct"] += 1
    if entry.get("prediction") is not None:
        stats["valid_predictions"] += 1
    if entry.get("error"):
        stats["failed_runs"] += 1
    stats["total_tokens"] += int(entry.get("tokens", 0) or 0)
    stats["total_elapsed_seconds"] += float(entry.get("elapsed_seconds", 0.0) or 0.0)


def finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    n = max(1, int(stats.get("total", 0) or 0))
    return {
        **stats,
        "accuracy": stats.get("correct", 0) / n,
        "extraction_rate": stats.get("valid_predictions", 0) / n,
        "avg_tokens": stats.get("total_tokens", 0) / n,
        "avg_elapsed_seconds": stats.get("total_elapsed_seconds", 0.0) / n,
    }


def sample_key(sample: Dict[str, Any], benchmark: str) -> str:
    if benchmark == "mmlu-pro":
        return str(sample.get("question_id"))
    if benchmark == "gsm-hard":
        return str(sample.get("dataset_index"))
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def load_checkpoint_samples(
    checkpoint_path: Optional[str],
    benchmark: str,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return [], set()

    by_key: Dict[str, Dict[str, Any]] = {}
    with open(checkpoint_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = sample_key(sample, benchmark)
            if key and key != "None":
                by_key[key] = sample
    return list(by_key.values()), set(by_key.keys())


def append_checkpoint_sample(
    checkpoint_path: Optional[str],
    sample: Dict[str, Any],
    lock: Optional[threading.Lock] = None,
) -> None:
    if not checkpoint_path:
        return

    def write_line() -> None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        with open(checkpoint_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
            file.flush()

    if lock is None:
        write_line()
        return
    with lock:
        write_line()


def sort_samples(samples: Sequence[Dict[str, Any]], benchmark: str) -> List[Dict[str, Any]]:
    key_name = "question_id" if benchmark == "mmlu-pro" else "dataset_index"

    def sort_key(sample: Dict[str, Any]) -> Tuple[int, str]:
        value = sample.get(key_name)
        try:
            return int(value), str(value)
        except (TypeError, ValueError):
            return 10**12, str(value)

    return sorted(samples, key=sort_key)


def dedupe_samples(samples: Iterable[Dict[str, Any]], benchmark: str) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        key = sample_key(sample, benchmark)
        if not key or key == "None":
            continue
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = sample
        else:
            by_key[key] = _merge_sample(previous, sample)
    return sort_samples(by_key.values(), benchmark)


def _merge_sample(previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(previous)
    for key, value in current.items():
        if key in {"multi_agent", "single_baseline"}:
            continue
        if value not in (None, "", []):
            merged[key] = value

    for field in ("multi_agent", "single_baseline"):
        old_entry = previous.get(field)
        new_entry = current.get(field)
        if old_entry and new_entry:
            merged[field] = new_entry if _entry_quality(new_entry) >= _entry_quality(old_entry) else old_entry
        elif new_entry:
            merged[field] = new_entry
        else:
            merged[field] = old_entry
    return merged


def _entry_quality(entry: Dict[str, Any]) -> Tuple[int, int, float]:
    has_no_error = not entry.get("error")
    has_prediction = entry.get("prediction") is not None
    elapsed = float(entry.get("elapsed_seconds", 0.0) or 0.0)
    return int(has_no_error), int(has_prediction), -elapsed


def _sample_quality(sample: Dict[str, Any]) -> Tuple[int, int, float]:
    entries = [sample.get("multi_agent"), sample.get("single_baseline")]
    has_no_error = any(isinstance(e, dict) and not e.get("error") for e in entries)
    has_prediction = any(isinstance(e, dict) and e.get("prediction") is not None for e in entries)
    elapsed = sum(
        float(e.get("elapsed_seconds", 0.0) or 0.0)
        for e in entries
        if isinstance(e, dict)
    )
    return int(has_no_error), int(has_prediction), -elapsed


def run_items(
    items: Sequence[Dict[str, Any]],
    worker: Callable[[Dict[str, Any]], Dict[str, Any]],
    workers: int,
    on_complete: Callable[[Dict[str, Any]], None],
) -> List[Dict[str, Any]]:
    if workers <= 1:
        results = []
        for item in items:
            result = worker(item)
            on_complete(result)
            results.append(result)
        return results

    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_item = {executor.submit(worker, item): item for item in items}
        for future in as_completed(future_to_item):
            result = future.result()
            on_complete(result)
            results.append(result)
    return results


def build_summary(
    *,
    benchmark: str,
    dataset_path: str,
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    include_multi: bool,
    include_single: bool,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    multi_stats = empty_stats()
    single_stats = empty_stats()
    for sample in samples:
        update_stats(multi_stats, sample.get("multi_agent"))
        update_stats(single_stats, sample.get("single_baseline"))

    summary = {
        "run_at": datetime.now().isoformat(),
        "dataset_path": dataset_path,
        "benchmark": benchmark,
        "config": config,
        "multi_agent_metrics": finalize_stats(multi_stats) if include_multi else None,
        "single_baseline_metrics": finalize_stats(single_stats) if include_single else None,
        "samples": samples,
    }
    if extra_fields:
        summary.update(extra_fields)
    return summary


def write_summary_json(summary: Dict[str, Any], save_path: str) -> None:
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def print_summary(summary: Dict[str, Any], visualization: Dict[str, Any], save_path: str) -> None:
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    multi = summary.get("multi_agent_metrics")
    if multi:
        print(
            "multi-agent: "
            f"acc={multi['accuracy']:.4f}, "
            f"extract_rate={multi['extraction_rate']:.4f}, "
            f"avg_tokens={multi['avg_tokens']:.2f}, "
            f"avg_time={multi['avg_elapsed_seconds']:.2f}s"
        )
    single = summary.get("single_baseline_metrics")
    if single:
        print(
            "single-baseline: "
            f"acc={single['accuracy']:.4f}, "
            f"extract_rate={single['extraction_rate']:.4f}, "
            f"avg_tokens={single['avg_tokens']:.2f}, "
            f"avg_time={single['avg_elapsed_seconds']:.2f}s"
        )
    if visualization.get("enabled"):
        print(
            f"charts saved to: {visualization.get('chart_dir')} "
            f"(backend={visualization.get('backend')}, files={len(visualization.get('files', []))})"
        )
    else:
        print(f"charts skipped: {visualization.get('error')}")
    print(f"saved to: {save_path}")
