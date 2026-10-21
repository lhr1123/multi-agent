import itertools
import random
import threading
import unittest
from types import SimpleNamespace

from models import SchedulerConfig, SubTask
from services.assignment_service import assign_with_capacities
from services.execution_controller import ExecutionController
from services.usage_tracker import UsageTracker, BudgetExceeded


class AssignmentTests(unittest.TestCase):
    def test_defer_is_part_of_optimization(self):
        self.assertEqual(assign_with_capacities([[10, 1], [1, -100]], [1, 1]), [0, -1])

    def test_mask_and_zero_capacity(self):
        self.assertEqual(assign_with_capacities([[100, 1]], [0, 1]), [1])
        self.assertEqual(assign_with_capacities([[100, 1]], [1, 1], [[False, False]]), [-1])

    def test_matching_agrees_with_exhaustive_search(self):
        rng = random.Random(10)
        for _ in range(30):
            matrix = [[rng.uniform(-3, 4) for _ in range(3)] for _ in range(3)]
            mask = [[rng.random() > 0.2 for _ in range(3)] for _ in range(3)]
            capacities = [1, 0, 2]
            result = assign_with_capacities(matrix, capacities, mask)
            actual = sum(matrix[i][j] for i, j in enumerate(result) if j >= 0)
            optimum = max(sum(matrix[i][j] for i, j in enumerate(candidate) if j >= 0)
                          for candidate in itertools.product(range(-1, 3), repeat=3)
                          if all(candidate.count(j) <= capacities[j] for j in range(3))
                          and all(j < 0 or mask[i][j] for i, j in enumerate(candidate)))
            self.assertAlmostEqual(actual, optimum)


class ControllerTests(unittest.TestCase):
    def make(self, tasks, scores, **config):
        cfg = SchedulerConfig(**config)
        agents = [SimpleNamespace(agent_id="a"), SimpleNamespace(agent_id="b")]
        return ExecutionController(tasks, agents, scores, cfg, UsageTracker(cfg))

    def test_failure_reassigned_and_dependencies_wait(self):
        tasks = [SubTask("t", "first", ""), SubTask("u", "second", "", dependencies=["t"])]
        controller = self.make(tasks, [[10, 9], [10, 9]])
        calls = []

        def execute(task, agent, deps):
            calls.append((task.id, agent.agent_id))
            if task.id == "t" and agent.agent_id == "a":
                return {"success": False, "failure_kind": "capability"}
            if task.id == "u":
                self.assertTrue(deps["t"]["success"])
            return {"success": True}

        state = controller.run(execute)
        self.assertEqual(calls, [("t", "a"), ("t", "b"), ("u", "a")])
        self.assertEqual(state.stop_reason, "completed")

    def test_failure_blocks_dependents(self):
        tasks = [SubTask("t", "", ""), SubTask("u", "", "", dependencies=["t"])]
        controller = self.make(tasks, [[10, 9], [10, 9]], max_attempts=1)
        seen = []
        state = controller.run(lambda t, a, d: seen.append(t.id) or {"success": False, "failure_kind": "capability"})
        self.assertEqual(seen, ["t"])
        self.assertEqual(state.stop_reason, "attempts_exhausted")
        self.assertEqual(state.tasks["u"].status, "blocked")

    def test_capacity_released_and_deferred_tasks_eventually_execute(self):
        tasks = [SubTask(str(i), "", "") for i in range(3)]
        controller = self.make(tasks, [[10, 0]] * 3, max_parallel=3, capacities={"a": 1, "b": 0})
        state = controller.run(lambda *args: {"success": True})
        self.assertEqual(state.stop_reason, "completed")
        self.assertEqual(len(state.rounds), 3)
        self.assertEqual(len(state.rounds[0]["deferred_ids"]), 2)

    def test_transport_failure_does_not_penalize_capability(self):
        controller = self.make([SubTask("t", "", "")], [[10, 9]])
        seen = []

        def execute(t, a, d):
            seen.append(a.agent_id)
            return {"success": len(seen) == 2, "failure_kind": "transport"}

        controller.run(execute)
        self.assertEqual(seen, ["a", "a"])

    def test_context_repair_stays_with_same_agent(self):
        controller = self.make([SubTask("t", "", "")], [[10, 10]])
        seen = []

        def execute(t, a, d):
            seen.append(a.agent_id)
            return {"success": len(seen) == 2, "failure_kind": "context"}

        controller.run(execute)
        self.assertEqual(seen, ["a", "a"])

    def test_static_keeps_initial_agent(self):
        controller = self.make([SubTask("t", "", "")], [[10, 9]], mode="static")
        seen = []
        controller.run(lambda t, a, d: seen.append(a.agent_id) or {"success": False, "failure_kind": "capability"})
        self.assertEqual(seen, ["a", "a"])

    def test_budget_stops_without_success(self):
        controller = self.make([SubTask("t", "", "")], [[10, 9]])

        def execute(*args):
            raise BudgetExceeded("no budget")

        state = controller.run(execute)
        self.assertEqual(state.stop_reason, "budget_exhausted")
        self.assertEqual(state.tasks["t"].status, "failed")

    def test_parallel_wave_runs_independent_tasks(self):
        controller = self.make([SubTask("t", "", ""), SubTask("u", "", "")],
                               [[10, 9], [9, 10]], max_parallel=2)
        barrier = threading.Barrier(2)

        def execute(*args):
            barrier.wait(timeout=2)
            return {"success": True}

        self.assertEqual(controller.run(execute).stop_reason, "completed")

    def test_negative_required_scores_still_make_progress(self):
        controller = self.make([SubTask("t", "", "")], [[0, 0]], cost_weight=100)
        self.assertEqual(controller.run(lambda *args: {"success": True}).stop_reason, "completed")

    def test_cycle_is_blocked(self):
        tasks = [SubTask("t", "", "", dependencies=["u"]), SubTask("u", "", "", dependencies=["t"])]
        controller = self.make(tasks, [[1, 1], [1, 1]])
        self.assertEqual(controller.run(lambda *a: self.fail("must not execute")).stop_reason, "blocked")

    def test_nonretryable_failure_is_not_repeated(self):
        controller = self.make([SubTask("t", "", "")], [[10, 9]])
        seen = []
        state = controller.run(lambda *args: seen.append(1) or {
            "success": False, "retryable": False, "failure_kind": "capability"})
        self.assertEqual(len(seen), 1)
        self.assertEqual(state.stop_reason, "blocked")

    def test_optional_work_is_skipped_after_required_evidence(self):
        tasks = [
            SubTask("solve", "", "", metadata={"required": True}),
            SubTask("critic", "", "", dependencies=["solve"], metadata={"required": False}),
        ]
        controller = self.make(tasks, [[10, 9], [10, 9]], early_stop_optional=True,
                               optional_utility_threshold=2.0)
        seen = []
        state = controller.run(lambda task, *_: seen.append(task.id) or {"success": True})
        self.assertEqual(seen, ["solve"])
        self.assertEqual(state.stop_reason, "completed_early")
        self.assertEqual(state.tasks["critic"].status, "skipped")
        self.assertEqual(state.termination["decision"], "stop")

    def test_high_utility_optional_work_continues(self):
        tasks = [
            SubTask("solve", "", "", metadata={"required": True}),
            SubTask("critic", "", "", dependencies=["solve"],
                    metadata={"required": False, "information_gain": 1.0}),
        ]
        controller = self.make(tasks, [[10, 9], [10, 9]], early_stop_optional=True,
                               optional_utility_threshold=0.0)
        seen = []
        state = controller.run(lambda task, *_: seen.append(task.id) or {"success": True})
        self.assertEqual(seen, ["solve", "critic"])
        self.assertEqual(state.stop_reason, "completed")
