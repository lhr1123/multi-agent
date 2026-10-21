import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Imports create SDK clients, but these tests never use a real transport.
os.environ.setdefault("SILICONFLOW_API_KEY", "test-key")

from models import SchedulerConfig
from llm.llm_interface import SiliconFlowLLMInterface
from pipelines.multi_agent_pipeline import run_multi_agent_flow
from services.single_llm_solver import solve_with_single_llm
from services.single_tool_solver import solve_with_single_tool_llm
from services.usage_tracker import UsageTracker, usage_scope
from services.failure_classification import classify_failure
from services.evidence_store import EvidenceStore
from multi_agent_pool.workflow import WorkflowManager
from evaluation import mmlu_pro_evaluator as mmlu
from evaluation import gsmhard_evaluator as gsm
from evaluation.runtime import validate_checkpoint


def reply(content):
    return SimpleNamespace(
        usage=SimpleNamespace(total_tokens=10, prompt_tokens=6, completion_tokens=4),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


class FakeClient:
    base_url = "mock://local"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        with self.lock:
            self.requests.append(kwargs)
            content = next(self.responses)
        if isinstance(content, Exception):
            raise content
        return reply(content)


class Matcher:
    def get_task_match_score(self, task, agent):
        return 10 if agent.agent_id in task.metadata.get("allowed_agent_ids", []) else 0


def output(answer="42", status="ok"):
    return json.dumps({"status": status, "final_answer": answer, "result_text": "Calculation: " + answer,
                       "key_facts": [], "confidence": 0.9})


def plan(agent="reasoningagent", count=1):
    return json.dumps({"sub_tasks": [
        {"title": f"task{i}", "description": "calculate the requested value", "dependencies": [] if i == 0 else [i-1],
         "metadata": {"allowed_agent_ids": [agent]}, "required_skills": []}
        for i in range(count)
    ]})


class PipelineTests(unittest.TestCase):
    def run_flow(self, orchestration, agent_responses, config=None, task_description="calculate"):
        orchestrator = SiliconFlowLLMInterface(client=FakeClient(orchestration), model_name="mock")
        agent = FakeClient(agent_responses)
        with patch("multi_agent_pool.workflow.sub_agent_llm_model", agent), contextlib.redirect_stdout(io.StringIO()):
            result = run_multi_agent_flow(orchestrator, "test", task_description, semantic_matcher=Matcher(),
                                          scheduler_config=config or SchedulerConfig())
        return result, agent

    def test_full_pipeline_counts_decomposition_reasoning_final(self):
        for mode in ("static", "rolling-hungarian"):
            result, _ = self.run_flow([plan(count=2)], [output(), output(), output()],
                                      SchedulerConfig(mode=mode))
            self.assertEqual(result["stop_reason"], "completed", result)
            self.assertEqual(result["final_result"], "42")
            self.assertEqual(result["total_tokens"], 40)
            self.assertEqual([c["stage"] for c in result["usage"]["calls"]],
                             ["decomposition", "reasoning", "reasoning", "final"])
            self.assertTrue(all(c["enable_thinking"] is False for c in result["usage"]["calls"]))
            json.dumps(result)

    def test_reasoning_agent_receives_complete_original_task(self):
        original = "Question: choose the correct option. Options: A. 10 B. 42"
        result, api = self.run_flow([plan()], [output(), output()], task_description=original)
        self.assertEqual(result["final_result"], "42", result)
        reasoning_prompt = api.requests[0]["messages"][1]["content"]
        self.assertIn(original, reasoning_prompt)
        self.assertIn("remaining_budget", reasoning_prompt)

    def test_oversized_plan_collapses_to_direct(self):
        result, _ = self.run_flow([plan(count=4)], [output(), output()],
                                  SchedulerConfig(max_plan_tasks=3), task_description="full original")
        tasks = result["decomposition"]["sub_tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["description"], "full original")
        self.assertEqual(tasks[0]["metadata"]["plan_mode"], "DIRECT")

    def test_plan_cache_replays_without_second_planner_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = SchedulerConfig(plan_cache_path=str(Path(tmp) / "plans.jsonl"))
            first, _ = self.run_flow([plan()], [output(), output()], config,
                                     task_description="cache me")
            second, _ = self.run_flow([], [output(), output()], config,
                                      task_description="cache me")
        self.assertFalse(first["plan_cache"]["hit"])
        self.assertTrue(second["plan_cache"]["hit"])
        self.assertEqual(second["final_result"], "42")

    def test_failure_classifier_distinguishes_missing_context_and_format(self):
        context_step = {"response": output("", status="failed"),
                        "structured_response": {"status": "failed", "result_text": "Options were not provided"}}
        self.assertEqual(classify_failure(context_step, {"failure_kind": "capability"}), "context")
        format_step = {"response": "not json", "structured_response": {"status": "partial"}}
        self.assertEqual(classify_failure(format_step, {"failure_kind": "capability"}), "format")

    def test_evidence_store_bounds_repeated_output(self):
        store = EvidenceStore(max_chars=100)
        raw = {"success": True, "structured_response": {
            "status": "ok", "final_answer": "42", "result_text": "x" * 1000,
            "key_facts": [{"value": "y" * 1000}], "confidence": 0.9,
        }}
        store.add("a", raw)
        store.add("b", raw)
        self.assertLessEqual(sum(len(item["result_text"]) for item in store.all()), 100)
        self.assertLessEqual(len(store.all()[0]["key_facts"][0]["value"]), 300)

    def test_tool_path_counts_code_decision_execution_summary(self):
        decision = json.dumps({"use_tool": True, "chosen_action": "execute_python",
                               "tool_input": {"code": "result = 6 * 7\nprint(result)"}})
        result, _ = self.run_flow([plan("pythonagent"), "result = 6 * 7\nprint(result)"],
                                  [decision, output(), output()])
        self.assertEqual(result["final_result"], "42", result)
        self.assertEqual(result["total_tokens"], 50)
        self.assertEqual([c["stage"] for c in result["usage"]["calls"]],
                         ["decomposition", "code_generation", "tool_decision", "tool_summary", "final"])
        self.assertTrue(all(c["enable_thinking"] is False for c in result["usage"]["calls"]))

    def test_failed_step_never_unlocks_dependent(self):
        result, api = self.run_flow([plan(count=2)], [output(status="failed")],
                                    SchedulerConfig(max_attempts=1))
        self.assertEqual(result["stop_reason"], "attempts_exhausted", result)
        self.assertEqual(len(api.requests), 1)
        self.assertIn("blocked", [t["status"] for t in result["execution"]["tasks"].values()])

    def test_malformed_decomposition_keeps_usage(self):
        result, _ = self.run_flow(["not json"], [])
        self.assertEqual(result["stop_reason"], "execution_error")
        self.assertEqual(result["total_tokens"], 10)

    def test_cyclic_plan_rejected_without_agent_calls(self):
        payload = json.loads(plan())
        payload["sub_tasks"][0]["dependencies"] = [0]
        result, api = self.run_flow([json.dumps(payload)], [])
        self.assertEqual(result["stop_reason"], "execution_error")
        self.assertEqual(api.requests, [])

    def test_missing_usage_and_failure_kept_in_pipeline(self):
        result, _ = self.run_flow([plan()], [TimeoutError("failed")], SchedulerConfig(max_attempts=1))
        self.assertEqual(result["total_tokens"], 10)
        self.assertEqual(result["usage"]["unknown_usage_calls"], 1)

    def test_model_failed_status_not_treated_as_completed(self):
        manager = WorkflowManager(client=FakeClient([output(status="failed")]))
        manager.create_workflow("w", "")
        manager.add_step("reasoningagent", "logical_reasoning", {"problem": "x", "context": {}})
        with usage_scope(UsageTracker()), contextlib.redirect_stdout(io.StringIO()):
            result = manager.execute_workflow()
        self.assertFalse(result["success"])
        self.assertEqual(manager.current_workflow.status.value, "failed")

    def test_single_bad_json_retains_text_instead_of_guessing_number(self):
        llm = SimpleNamespace(model_name="mock", client=FakeClient(["Unable to choose; considered 3 possibilities."]))
        result = solve_with_single_llm(llm, "test", "question")
        self.assertEqual(result["answer"], "Unable to choose; considered 3 possibilities.")
        self.assertEqual(result["usage"]["known_total_tokens"], 10)

    def test_single_python_baseline_executes_tool_without_orchestrator(self):
        decision = json.dumps({
            "use_python": True,
            "code": "result = 6 * 7\nprint(result)",
            "direct_answer": "",
            "reasoning": "exact arithmetic",
        })
        llm = SimpleNamespace(model_name="mock", client=FakeClient([
            decision,
            json.dumps({"answer": "42", "reasoning": "Python returned 42"}),
        ]))
        result = solve_with_single_tool_llm(llm, "test", "What is 6 * 7?")
        self.assertEqual(result["answer"], "42")
        self.assertTrue(result["tool_trace"]["used"])
        self.assertEqual(result["tool_trace"]["attempts"][0]["return_value"], "42")
        self.assertEqual([call["stage"] for call in result["usage"]["calls"]],
                         ["single_tool_decision", "single_tool_final"])
        self.assertEqual(result["tokens_used"]["total_tokens"], 20)

    def test_single_python_baseline_direct_path_uses_one_call(self):
        decision = json.dumps({
            "use_python": False,
            "code": "",
            "direct_answer": "The answer is (B)",
            "reasoning": "No calculation needed",
        })
        llm = SimpleNamespace(model_name="mock", client=FakeClient([decision]))
        result = solve_with_single_tool_llm(llm, "test", "Choose an option")
        self.assertEqual(result["answer"], "The answer is (B)")
        self.assertFalse(result["tool_trace"]["used"])
        self.assertEqual(len(result["usage"]["calls"]), 1)

    def test_single_python_baseline_repairs_failed_code(self):
        decision = json.dumps({
            "use_python": True,
            "code": "raise ValueError('bad')",
            "direct_answer": "",
            "reasoning": "calculate",
        })
        repair = json.dumps({"code": "result = -5\nprint(result)", "reasoning": "fixed"})
        final = json.dumps({"answer": "-5", "reasoning": "literal result"})
        llm = SimpleNamespace(model_name="mock", client=FakeClient([decision, repair, final]))
        result = solve_with_single_tool_llm(
            llm, "test", "Return a negative value", scheduler_config=SchedulerConfig(max_attempts=2))
        self.assertEqual(result["answer"], "-5")
        self.assertEqual(len(result["tool_trace"]["attempts"]), 2)
        self.assertEqual([call["stage"] for call in result["usage"]["calls"]],
                         ["single_tool_decision", "single_tool_repair", "single_tool_final"])

    def test_mmlu_parse_failure_and_resume(self):
        item = {"question_id": 1, "question": "q", "options": ["x", "y"], "answer": "A", "category": "test"}
        llm = SimpleNamespace(model_name="mock", client=FakeClient(['{"answer":"unknown"}']))
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(mmlu, "load_mmlu_pro_splits", return_value={"validation": []}), \
             patch.object(mmlu, "load_mmlu_pro_dataset", return_value=[item]), \
             contextlib.redirect_stdout(io.StringIO()):
            kwargs = dict(orchestrator_llm=llm, dataset_path="mock", single_llm=llm,
                          run_kind="single", charts_during_run=False,
                          save_path=str(Path(tmp) / "summary.json"),
                          checkpoint_path=str(Path(tmp) / "checkpoint.jsonl"))
            summary = mmlu.evaluate_multi_agent_on_mmlu_pro(**kwargs)
            self.assertIsNone(summary["samples"][0]["single_baseline"]["prediction"])
            self.assertEqual(summary["single_baseline_metrics"]["extraction_rate"], 0)
            self.assertEqual(summary["single_baseline_metrics"]["model_calls"], 1)
            resumed = mmlu.evaluate_multi_agent_on_mmlu_pro(**kwargs, resume=True)
            self.assertEqual(len(llm.client.requests), 1)
            self.assertEqual(resumed["single_baseline_metrics"]["accuracy"], 0)
            with self.assertRaises(ValueError):
                mmlu.evaluate_multi_agent_on_mmlu_pro(**kwargs, resume=True,
                    scheduler_config=SchedulerConfig(mode="static"))

    def test_gsm_entry_saves_pipeline_trace(self):
        item = {"index": 0, "input": "q", "target": 42}
        llm = SimpleNamespace(model_name="mock")
        result = {"final_result": "42", "total_tokens": 20, "stop_reason": "completed", "error": "",
                  "usage": {"known_total_tokens": 20, "calls": []}, "execution": {"rounds": []}}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(gsm, "load_gsmhard_dataset", return_value=[item]), \
             patch.object(gsm, "SemanticMatcher", return_value=Matcher()), \
             patch.object(gsm, "run_multi_agent_flow", return_value=result), \
             contextlib.redirect_stdout(io.StringIO()):
            summary = gsm.evaluate_multi_agent_on_gsmhard(
                llm, "mock", charts_during_run=False, save_path=str(Path(tmp) / "summary.json"))
            entry = summary["samples"][0]["multi_agent"]
            self.assertTrue(entry["is_correct"])
            self.assertEqual(entry["trace"], result)

    def test_gsm_python_baseline_is_recorded_in_metadata_and_trace(self):
        item = {"index": 0, "input": "What is 6 * 7?", "target": 42}
        llm = SimpleNamespace(model_name="mock", client=FakeClient([]))
        tool_result = {
            "answer": "42",
            "baseline_mode": "python",
            "tool_trace": {"used": True, "attempts": [{"success": True, "return_value": "42"}]},
            "usage": {"known_total_tokens": 20, "calls": []},
            "tokens_used": {"total_tokens": 20},
            "raw_response": '{"answer":"42"}',
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(gsm, "load_gsmhard_dataset", return_value=[item]), \
             patch.object(gsm, "solve_with_single_tool_llm", return_value=tool_result), \
             contextlib.redirect_stdout(io.StringIO()):
            summary = gsm.evaluate_multi_agent_on_gsmhard(
                llm, "mock", run_kind="single", single_llm=llm,
                single_baseline_mode="python", charts_during_run=False,
                save_path=str(Path(tmp) / "summary.json"),
                checkpoint_path=str(Path(tmp) / "checkpoint.jsonl"),
            )
        entry = summary["samples"][0]["single_baseline"]
        self.assertTrue(entry["is_correct"])
        self.assertEqual(entry["baseline_mode"], "python")
        self.assertTrue(entry["tool_trace"]["used"])
        self.assertEqual(summary["config"]["single_baseline_mode"], "python")
        self.assertEqual(summary["config"]["experiment"]["single_baseline_mode"], "python")

    def test_old_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            validate_checkpoint([{"question_id": 1}], {"fingerprint": "new"})

    def test_choice_parser_rejects_incidental_and_conflicting_letters(self):
        self.assertIsNone(mmlu.extract_choice_prediction("I considered A and B but cannot decide.", 10))
        self.assertIsNone(mmlu.extract_choice_prediction("The answer is (A). The answer is (B).", 10))
        self.assertIsNone(mmlu.extract_choice_prediction("The answer is (J)", 4))
        self.assertEqual(mmlu.extract_choice_prediction("The answer is (B)", 4), "B")
        self.assertEqual(mmlu.extract_choice_prediction("B", 4), "B")

    def test_parallel_workflows_keep_their_own_inputs(self):
        payload = json.loads(plan(count=2))
        for i, task in enumerate(payload["sub_tasks"]):
            task["dependencies"] = []
            task["description"] = f"payload-{i}"
        barrier = threading.Barrier(2)

        def respond(**kwargs):
            if "logical reasoning specialist" in kwargs["messages"][0]["content"]:
                marker = "payload-0" if "payload-0" in kwargs["messages"][1]["content"] else "payload-1"
                barrier.wait(timeout=2)
                return reply(output(marker))
            return reply(output())

        api = SimpleNamespace(base_url="mock://parallel",
                              chat=SimpleNamespace(completions=SimpleNamespace(create=respond)))
        llm = SiliconFlowLLMInterface(client=FakeClient([json.dumps(payload)]), model_name="mock")
        with patch("multi_agent_pool.workflow.sub_agent_llm_model", api), contextlib.redirect_stdout(io.StringIO()):
            result = run_multi_agent_flow(llm, "test", "q", semantic_matcher=Matcher(),
                scheduler_config=SchedulerConfig(max_parallel=2, service_concurrency=2))
        self.assertEqual(result["stop_reason"], "completed", result)
        results = list(result["step_structured"].values())
        self.assertEqual([r["final_answer"] for r in results], ["payload-0", "payload-1"])

    def test_embedding_cache_without_downloading_model(self):
        import torch
        from semantic_matcher import SemanticMatcher
        from unittest.mock import Mock
        encoder = Mock()
        encoder.encode.side_effect = lambda texts, **kw: torch.tensor([[1.0, 0.0] for _ in texts])
        with patch("semantic_matcher.SentenceTransformer", return_value=encoder):
            matcher = SemanticMatcher()
            self.assertEqual(matcher.get_match_score(["math"], ["math"]), 10)
            self.assertEqual(matcher.get_match_score(["math"], ["math"]), 10)
            self.assertEqual(encoder.encode.call_count, 1)

    def test_stratified_sampling_spreads_categories(self):
        rows = [
            {"question_id": f"{category}-{i}", "category": category}
            for category in ("business", "math", "law") for i in range(4)
        ]
        selected = mmlu.stratified_mmlu_samples(rows, limit=3, seed=7)
        self.assertEqual(len({row["category"] for row in selected}), 3)
