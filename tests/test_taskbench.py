import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.taskbench_evaluator import evaluate_taskbench_offline, load_samples, make_fixed_tasks
from models import SchedulerConfig


class Matcher:
    scores = {("tool_a", "bingagent"): 10, ("tool_a", "websiteagent"): 9,
              ("tool_b", "bingagent"): 9, ("tool_b", "websiteagent"): 1}

    def get_task_match_score(self, task, agent):
        return self.scores.get((task.title, agent.agent_id), 0)


class TaskBenchTests(unittest.TestCase):
    def test_exact_duplicate_source_row_is_skipped_but_conflict_is_rejected(self):
        sample = {"id": "one", "user_request": "test"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.json"
            path.write_text(json.dumps(sample) + "\n" + json.dumps(sample) + "\n", encoding="utf-8")
            rows, duplicates = load_samples(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(duplicates, 1)
            path.write_text(json.dumps(sample) + "\n" +
                            json.dumps({"id": "one", "user_request": "different"}) + "\n",
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Conflicting TaskBench id"):
                load_samples(path)

    def test_inventory_cli_does_not_need_api_credentials(self):
        sample = {"id": "inventory-1", "type": "node", "user_request": "Look up a topic",
                  "task_nodes": [{"task": "get_news_for_topic"}], "task_links": []}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, summary = root / "data.json", root / "summary.json"
            data.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            environment = os.environ.copy()
            for key in ("SILICONFLOW_API_KEY", "TASK_ORCHESTRATOR_API_KEY",
                        "SUB_AGENT_API_KEY", "SINGLE_BASELINE_API_KEY"):
                environment.pop(key, None)
            process = subprocess.run(
                [sys.executable, "-B", "main.py", "--mode", "dataset", "--benchmark", "taskbench",
                 "--dataset-path", str(data), "--save-path", str(summary)],
                cwd=Path(__file__).resolve().parents[1], env=environment,
                capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(summary.read_text(encoding="utf-8"))["summary"]
                             ["tool_inventory"]["get_news_for_topic"], 1)

    def test_gold_graph_preserves_dependencies_and_rejects_ambiguous_links(self):
        mapping = {"tool_a": {"allowed_agent_ids": ["bingagent"]},
                   "tool_b": {"allowed_agent_ids": ["websiteagent"]}}
        sample = {"id": "x", "task_nodes": [{"task": "tool_a"}, {"task": "tool_b"}],
                  "task_links": [{"source": "tool_a", "target": "tool_b"}]}
        tasks = make_fixed_tasks(sample, mapping, {}, 3)
        self.assertEqual(tasks[1].dependencies, ["x:node:0"])
        sample["task_nodes"].append({"task": "tool_a"})
        with self.assertRaisesRegex(ValueError, "ambiguous_or_unknown_link"):
            make_fixed_tasks(sample, mapping, {}, 3)

    def test_offline_policies_and_resume_use_identical_scores(self):
        sample = {"id": "42", "type": "dag", "user_request": "Complete both tasks",
                  "task_nodes": [{"task": "tool_a"}, {"task": "tool_b"}], "task_links": []}
        mapping = {"version": 1, "tools": {
            "tool_a": {"allowed_agent_ids": ["bingagent", "websiteagent"], "reviewed": True},
            "tool_b": {"allowed_agent_ids": ["bingagent", "websiteagent"], "reviewed": True}}}
        config = SchedulerConfig(max_parallel=2, capacities={"bingagent": 1, "websiteagent": 1})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, tool_map = root / "data.json", root / "map.json"
            summary, checkpoint = root / "summary.json", root / "checkpoint.jsonl"
            data.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            tool_map.write_text(json.dumps(mapping), encoding="utf-8")
            kwargs = dict(data_path=data, save_path=summary, config=config,
                          tool_map_path=tool_map, checkpoint_path=checkpoint, matcher=Matcher())
            report = evaluate_taskbench_offline(**kwargs)
            runs = report["samples"][0]["policies"]
            self.assertEqual(runs["hungarian"]["assignment_score_sum"], 18)
            self.assertEqual(runs["greedy"]["assignment_score_sum"], 11)
            self.assertEqual(runs["hungarian"]["capacity_conflict_rounds"], 1)
            self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), 1)
            resumed = evaluate_taskbench_offline(**kwargs, resume=True)
            self.assertEqual(resumed["summary"]["policy_totals"], report["summary"]["policy_totals"])
            self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), 1)
            with self.assertRaisesRegex(ValueError, "checkpoint path|Output already exists"):
                evaluate_taskbench_offline(**kwargs)


if __name__ == "__main__":
    unittest.main()
