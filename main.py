"""
CLI entrypoint for the multi-agent task allocation system.

The heavy lifting now lives in pipeline / evaluation / service modules so
this file can stay focused on argument parsing and high-level dispatch.
"""

import argparse
import json
import os
import sys

from models import SchedulerConfig


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


DEFAULT_TASK_TITLE = "GSM_hard_1_"
DEFAULT_TASK_DESCRIPTION = (
    "Grandma Jones baked 5 apple pies for the fireman's luncheon. "
    "She cut each pie into 4225558 pieces and set the five pies out on the buffet table "
    "for the guests to serve themselves. At the end of the evening, after the guests had "
    "taken and eaten their pieces of pie, there were 14 pieces of pie remaining. "
    "How many pieces were taken by the guests?"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多智能体任务分配系统")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "single", "multi", "dataset"],
        help="运行模式: both=对比实验, single=仅单LLM, multi=仅多智能体",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/gsm-hard/gsmhardv2.jsonl",
        help="GSM-hard JSONL, MMLU-Pro directory/parquet, or TaskBench data.json JSONL.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gsm-hard",
        choices=["gsm-hard", "mmlu-pro", "taskbench"],
        help="Benchmark adapter used in dataset mode.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=0,
        help="Number of samples to evaluate; <=0 means all samples.",
    )
    parser.add_argument(
        "--dataset-offset",
        type=int,
        default=0,
        help="Start index offset in dataset.",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "stratified"],
        default="sequential",
        help="MMLU-Pro sample selection strategy.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where to save evaluation summary JSON. If omitted, a benchmark-specific default is used.",
    )
    parser.add_argument(
        "--compare-single",
        action="store_true",
        help="Also run single-LLM baseline on the same benchmark samples.",
    )
    parser.add_argument(
        "--run-kind",
        type=str,
        default="multi",
        choices=["multi", "single", "both"],
        help="Dataset run side: multi=multi-agent only, single=baseline only, both=run both. Old --compare-single behavior is preserved.",
    )
    parser.add_argument(
        "--single-baseline",
        dest="single_baseline_mode",
        choices=["plain", "python"],
        default="plain",
        help="Single-agent baseline: plain model call or one agent with the shared Python executor.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of dataset samples to evaluate concurrently. Default 1 preserves old serial behavior.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Append one JSONL record per completed dataset sample for crash-safe resume.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Read --checkpoint-path and skip already completed sample ids.",
    )
    parser.add_argument(
        "--no-charts-during-run",
        action="store_true",
        help="Skip chart generation for shard/checkpoint runs; generate charts during merge instead.",
    )
    parser.add_argument(
        "--verbose-eval",
        action="store_true",
        help="Print full per-sample execution logs during dataset evaluation.",
    )
    parser.add_argument("--scheduler", choices=["static", "rolling-hungarian"])
    parser.add_argument("--scheduler-config", help="JSON file with SchedulerConfig fields")
    parser.add_argument("--assignment-policy", choices=["hungarian", "greedy", "random"])
    parser.add_argument("--max-parallel", type=int, help="Concurrent operations per sample")
    parser.add_argument("--service-concurrency", type=int, help="Concurrent calls per endpoint across samples")
    parser.add_argument("--agent-capacity", action="append", default=[], metavar="AGENT=N")
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-tokens", type=int, help="Per-sample token budget; 0 is unlimited")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--final-reserve-tokens", type=int)
    parser.add_argument("--deadline-seconds", type=float)
    parser.add_argument("--seed", type=int, help="Assignment random seed; not an API generation seed")
    parser.add_argument("--max-plan-tasks", type=int, help="Maximum planned work items per sample")
    parser.add_argument("--plan-cache-path", help="JSONL plan cache shared by scheduler variants")
    parser.add_argument("--evidence-max-chars", type=int, help="Maximum compact evidence characters")
    parser.add_argument("--optional-utility-threshold", type=float,
                        help="Minimum utility required to execute optional work")
    parser.add_argument("--reliability-priors", help="JSON file of frozen development-set agent priors")
    parser.add_argument("--no-direct-plan", action="store_true")
    parser.add_argument("--no-early-stop-optional", action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--taskbench-tool-map",
                        help="Reviewed tool-to-eligible-agent JSON; omit for inventory only")
    parser.add_argument("--taskbench-tools",
                        help="TaskBench tool_desc.json; defaults to dataset sibling when present")
    return parser


def scheduler_config_from_args(args):
    values = {}
    if args.scheduler_config:
        with open(args.scheduler_config, encoding="utf-8") as file:
            values = json.load(file)
        if not isinstance(values, dict):
            raise ValueError("Scheduler config must be a JSON object")
    for arg, key in (
        ("scheduler", "mode"), ("assignment_policy", "policy"),
        ("max_parallel", "max_parallel"), ("service_concurrency", "service_concurrency"),
        ("max_attempts", "max_attempts"), ("max_tokens", "max_tokens"),
        ("max_output_tokens", "max_output_tokens"), ("final_reserve_tokens", "final_reserve_tokens"),
        ("deadline_seconds", "deadline_seconds"), ("seed", "seed"),
        ("max_plan_tasks", "max_plan_tasks"), ("plan_cache_path", "plan_cache_path"),
        ("evidence_max_chars", "evidence_max_chars"),
        ("optional_utility_threshold", "optional_utility_threshold"),
    ):
        if getattr(args, arg) is not None:
            values[key] = getattr(args, arg)
    if args.agent_capacity:
        capacities = dict(values.get("capacities", {}))
        for item in args.agent_capacity:
            agent, count = item.rsplit("=", 1)
            capacities[agent] = int(count)
        values["capacities"] = capacities
    if args.reliability_priors:
        with open(args.reliability_priors, encoding="utf-8") as file:
            priors = json.load(file)
        if isinstance(priors, dict) and "reliability_priors" in priors:
            priors = priors["reliability_priors"]
        if not isinstance(priors, dict):
            raise ValueError("Reliability priors must be a JSON object keyed by agent ID")
        values["reliability_priors"] = priors
    if args.no_feedback:
        values["feedback"] = False
    if args.no_direct_plan:
        values["allow_direct_plan"] = False
    if args.no_early_stop_optional:
        values["early_stop_optional"] = False
    return SchedulerConfig(**values)


def main() -> None:
    args = build_parser().parse_args()
    scheduler_config = scheduler_config_from_args(args)

    if args.mode == "dataset" and args.benchmark == "taskbench":
        from evaluation.taskbench_evaluator import evaluate_taskbench_offline

        if args.run_kind != "multi" or args.compare_single:
            raise ValueError("TaskBench first version supports offline assignment only")
        if args.assignment_policy:
            raise ValueError("TaskBench offline compares all three assignment policies")

        data_path = ("dataset/taskbench/data.json" if args.dataset_path ==
                     "dataset/gsm-hard/gsmhardv2.jsonl" else args.dataset_path)
        report = evaluate_taskbench_offline(
            data_path=data_path,
            save_path=args.save_path or "result/taskbench_offline.json",
            config=scheduler_config,
            tool_map_path=args.taskbench_tool_map,
            tool_desc_path=args.taskbench_tools,
            offset=args.dataset_offset,
            limit=None if args.dataset_limit <= 0 else args.dataset_limit,
            checkpoint_path=args.checkpoint_path,
            resume=args.resume,
        )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return

    # Keep the offline TaskBench path independent of API credentials and clients.
    from evaluation.gsmhard_evaluator import evaluate_multi_agent_on_gsmhard
    from evaluation.mmlu_pro_evaluator import evaluate_multi_agent_on_mmlu_pro
    from llm.llm_config import (
        SINGLE_BASELINE_MODEL, SUB_AGENT_MODEL, TASK_ORCHESTRATOR_MODEL,
        single_baseline_llm_model, task_orchestrator_llm_model,
    )
    from llm.llm_interface import SiliconFlowLLMInterface
    from pipelines.multi_agent_pipeline import run_multi_agent_flow
    from services.single_llm_solver import solve_with_single_llm
    from services.single_tool_solver import solve_with_single_tool_llm

    orchestrator_llm = SiliconFlowLLMInterface(
        client=task_orchestrator_llm_model,
        model_name=TASK_ORCHESTRATOR_MODEL,
    )
    single_llm = SiliconFlowLLMInterface(
        client=single_baseline_llm_model,
        model_name=SINGLE_BASELINE_MODEL,
    )

    if args.mode == "dataset":
        eval_limit = None if args.dataset_limit is None or args.dataset_limit <= 0 else args.dataset_limit
        if args.save_path:
            save_path = args.save_path
        elif args.benchmark == "mmlu-pro":
            save_path = os.path.join("result", "mmlu_pro_eval_output.json")
        else:
            save_path = os.path.join("result", "gsmhard_eval_output.json")

        if args.benchmark == "mmlu-pro":
            evaluate_multi_agent_on_mmlu_pro(
                orchestrator_llm=orchestrator_llm,
                dataset_path=args.dataset_path,
                sub_agent_model_name=SUB_AGENT_MODEL,
                limit=eval_limit,
                offset=args.dataset_offset,
                save_path=save_path,
                compare_single_baseline=args.compare_single,
                single_llm=single_llm if args.compare_single or args.run_kind in {"single", "both"} else None,
                quiet_per_sample=not args.verbose_eval,
                run_kind=args.run_kind,
                workers=max(1, args.workers),
                checkpoint_path=args.checkpoint_path,
                resume=args.resume,
                charts_during_run=not args.no_charts_during_run,
                scheduler_config=scheduler_config,
                sample_strategy=args.sample_strategy,
                single_baseline_mode=args.single_baseline_mode,
            )
        else:
            evaluate_multi_agent_on_gsmhard(
                orchestrator_llm=orchestrator_llm,
                dataset_path=args.dataset_path,
                sub_agent_model_name=SUB_AGENT_MODEL,
                limit=eval_limit,
                offset=args.dataset_offset,
                save_path=save_path,
                compare_single_baseline=args.compare_single,
                single_llm=single_llm if args.compare_single or args.run_kind in {"single", "both"} else None,
                quiet_per_sample=not args.verbose_eval,
                run_kind=args.run_kind,
                workers=max(1, args.workers),
                checkpoint_path=args.checkpoint_path,
                resume=args.resume,
                charts_during_run=not args.no_charts_during_run,
                scheduler_config=scheduler_config,
                single_baseline_mode=args.single_baseline_mode,
            )
        return

    single_solver = (solve_with_single_tool_llm
                     if args.single_baseline_mode == "python" else solve_with_single_llm)

    if args.mode == "both":
        print("\n" + "=" * 60)
        print("对比实验：单一大模型 vs 多智能体系统")
        print("=" * 60)

        print("\n【实验】单一大模型")
        print("-" * 60)
        single_result = single_solver(single_llm, DEFAULT_TASK_TITLE, DEFAULT_TASK_DESCRIPTION,
                                      scheduler_config=scheduler_config)
        print(f"答案: {single_result.get('answer', 'N/A')}")
        print(f"Token消耗: {single_result['tokens_used']['total_tokens']}")

        print("\n" + "=" * 60)
        print("【实验】多智能体系统")
        print("-" * 60)
        multi_result = run_multi_agent_flow(
            orchestrator_llm,
            DEFAULT_TASK_TITLE,
            DEFAULT_TASK_DESCRIPTION,
            sub_agent_model_name=SUB_AGENT_MODEL,
            scheduler_config=scheduler_config,
        )
        print(json.dumps(multi_result, ensure_ascii=False, indent=2))
        return

    if args.mode == "single":
        result = single_solver(single_llm, DEFAULT_TASK_TITLE, DEFAULT_TASK_DESCRIPTION,
                               scheduler_config=scheduler_config)
        print(f"答案: {result.get('answer', 'N/A')}")
        return

    multi_result = run_multi_agent_flow(
        orchestrator_llm,
        DEFAULT_TASK_TITLE,
        DEFAULT_TASK_DESCRIPTION,
        sub_agent_model_name=SUB_AGENT_MODEL,
        scheduler_config=scheduler_config,
    )
    print(json.dumps(multi_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
