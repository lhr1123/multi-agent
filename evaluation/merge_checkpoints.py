"""Merge evaluation checkpoint JSONL files into one summary JSON."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List

from evaluation.runtime import build_summary, dedupe_samples, write_summary_json
from utils.eval_visualization import generate_eval_charts


def _read_samples(path: str) -> Iterable[Dict[str, Any]]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        for sample in payload.get("samples", []) or []:
            if isinstance(sample, dict):
                yield sample
        return

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(sample, dict):
                yield sample


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


def merge_checkpoint_files(
    paths: List[str],
    benchmark: str,
    dataset_path: str,
    save_path: str,
    generate_charts: bool = True,
) -> Dict[str, Any]:
    samples: List[Dict[str, Any]] = []
    for path in paths:
        samples.extend(_read_samples(path))

    merged_samples = dedupe_samples(samples, benchmark)
    include_multi = any(sample.get("multi_agent") for sample in merged_samples)
    include_single = any(sample.get("single_baseline") for sample in merged_samples)
    extra_fields = {}
    if benchmark == "mmlu-pro":
        extra_fields = {
            "multi_agent_per_category": _build_per_category(merged_samples, "multi_agent") if include_multi else None,
            "single_baseline_per_category": _build_per_category(merged_samples, "single_baseline") if include_single else None,
        }

    summary = build_summary(
        benchmark=benchmark,
        dataset_path=dataset_path,
        samples=merged_samples,
        config={
            "merge_inputs": paths,
            "merged_sample_count": len(merged_samples),
        },
        include_multi=include_multi,
        include_single=include_single,
        extra_fields=extra_fields,
    )
    visualization = generate_eval_charts(summary, save_path=save_path) if generate_charts else {"enabled": False, "error": "disabled"}
    summary["visualization"] = visualization
    write_summary_json(summary, save_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge evaluation checkpoint JSONL files.")
    parser.add_argument("--benchmark", choices=["gsm-hard", "mmlu-pro"], required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("inputs", nargs="+", help="Checkpoint JSONL files or summary JSON files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    missing = [path for path in args.inputs if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    summary = merge_checkpoint_files(
        paths=args.inputs,
        benchmark=args.benchmark,
        dataset_path=args.dataset_path,
        save_path=args.save_path,
        generate_charts=not args.no_charts,
    )
    multi = summary.get("multi_agent_metrics")
    single = summary.get("single_baseline_metrics")
    print(f"merged samples: {len(summary.get('samples', []) or [])}")
    if multi:
        print(f"multi-agent acc={multi['accuracy']:.4f}, total={multi['total']}")
    if single:
        print(f"single-baseline acc={single['accuracy']:.4f}, total={single['total']}")
    print(f"saved to: {args.save_path}")


if __name__ == "__main__":
    main()
