"""
CLI entrypoint for the multi-agent task allocation system.

The heavy lifting now lives in pipeline / evaluation / service modules so
this file can stay focused on argument parsing and high-level dispatch.
"""

import argparse
import os
import sys

from evaluation.gsmhard_evaluator import evaluate_multi_agent_on_gsmhard
from evaluation.mmlu_pro_evaluator import evaluate_multi_agent_on_mmlu_pro
from llm.llm_config import (
    SINGLE_BASELINE_MODEL,
    SUB_AGENT_MODEL,
    TASK_ORCHESTRATOR_MODEL,
    single_baseline_llm_model,
    task_orchestrator_llm_model,
)
from llm.llm_interface import SiliconFlowLLMInterface
from pipelines.multi_agent_pipeline import run_multi_agent_flow
from services.single_llm_solver import solve_with_single_llm


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
        help="Dataset path. For GSM-hard use a jsonl file; for MMLU-Pro use the dataset directory or parquet file.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gsm-hard",
        choices=["gsm-hard", "mmlu-pro"],
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
        "--verbose-eval",
        action="store_true",
        help="Print full per-sample execution logs during dataset evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
                single_llm=single_llm if args.compare_single else None,
                quiet_per_sample=not args.verbose_eval,
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
                single_llm=single_llm if args.compare_single else None,
                quiet_per_sample=not args.verbose_eval,
            )
        return

    if args.mode == "both":
        print("\n" + "=" * 60)
        print("对比实验：单一大模型 vs 多智能体系统")
        print("=" * 60)

        print("\n【实验】单一大模型")
        print("-" * 60)
        single_result = solve_with_single_llm(single_llm, DEFAULT_TASK_TITLE, DEFAULT_TASK_DESCRIPTION)
        print(f"答案: {single_result.get('answer', 'N/A')}")
        print(f"Token消耗: {single_result['tokens_used']['total_tokens']}")

        print("\n" + "=" * 60)
        print("【实验】多智能体系统")
        print("-" * 60)
        run_multi_agent_flow(
            orchestrator_llm,
            DEFAULT_TASK_TITLE,
            DEFAULT_TASK_DESCRIPTION,
            sub_agent_model_name=SUB_AGENT_MODEL,
        )
        return

    if args.mode == "single":
        result = solve_with_single_llm(single_llm, DEFAULT_TASK_TITLE, DEFAULT_TASK_DESCRIPTION)
        print(f"答案: {result.get('answer', 'N/A')}")
        return

    run_multi_agent_flow(
        orchestrator_llm,
        DEFAULT_TASK_TITLE,
        DEFAULT_TASK_DESCRIPTION,
        sub_agent_model_name=SUB_AGENT_MODEL,
    )


if __name__ == "__main__":
    main()
