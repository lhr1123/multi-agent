"""Console-print helpers for CLI runs."""

import re


def print_decomposition_result(result) -> None:
    print("\n" + "=" * 60)
    print("任务分解结果")
    print("=" * 60)
    print(f"\n原始任务: {result.original_task.title}")
    print(f"分解策略: {result.decomposition_strategy}")

    print(f"\n子任务列表 (共 {len(result.sub_tasks)} 个):")
    for i, sub_task in enumerate(result.sub_tasks, 1):
        print(f"\n  子任务{i}: {sub_task.title}")
        print(f"    描述: {sub_task.description}")
        print(f"    所需技能: {sub_task.required_skills}")
        print(f"    依赖: {sub_task.dependencies}")

    print("\n任务依赖关系:")
    graph = result.get_task_graph()
    for task_id, deps in graph.items():
        print(f"  {task_id} -> {deps if deps else '无依赖'}")


def print_final_result(final_result: str) -> None:
    print("\n" + "=" * 60)
    print("最终答案")
    print("=" * 60)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", final_result or "")
    if numbers:
        print(f"答案: {numbers[-1]}")
    else:
        print(f"答案: {final_result[:300]}...")
    print(f"\n完整结果:\n{(final_result or '')[:800]}...")
