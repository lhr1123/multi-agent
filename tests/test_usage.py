import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from models import SchedulerConfig
from services.usage_tracker import UsageTracker, usage_scope, tracked_completion, BudgetExceeded, DeadlineExceeded
from services.code_execution import execute_python


def response(total=10):
    usage = None if total is None else SimpleNamespace(total_tokens=total, prompt_tokens=3, completion_tokens=total-3)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


def client(callback):
    return SimpleNamespace(base_url="test", chat=SimpleNamespace(completions=SimpleNamespace(create=callback)))


class UsageTests(unittest.TestCase):
    def test_missing_usage_is_unknown_and_charged(self):
        tracker = UsageTracker()
        with usage_scope(tracker):
            tracked_completion(client(lambda **kw: response(None)), stage="reasoning", messages=[])
        report = tracker.snapshot()
        self.assertEqual(report["unknown_usage_calls"], 1)
        self.assertEqual(report["known_total_tokens"], 0)
        self.assertGreater(report["budget_charged_tokens"], 0)
        self.assertEqual(report["reserved_tokens"], 0)

    def test_exception_retains_unknown_charge(self):
        tracker = UsageTracker()

        def fail(**kwargs):
            raise TimeoutError()

        with usage_scope(tracker), self.assertRaises(TimeoutError):
            tracked_completion(client(fail), stage="reasoning", messages=[])
        self.assertEqual(tracker.snapshot()["unknown_usage_calls"], 1)

    def test_final_reserve_and_output_cap(self):
        config = SchedulerConfig(max_tokens=1000, max_output_tokens=200, final_reserve_tokens=800)
        tracker = UsageTracker(config)
        requests = []
        api = client(lambda **kw: requests.append(kw) or response())
        with usage_scope(tracker):
            with self.assertRaises(BudgetExceeded):
                tracked_completion(api, stage="reasoning", messages=[])
            tracked_completion(api, stage="final", messages=[])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["max_tokens"], 200)
        self.assertEqual(tracker.snapshot()["known_total_tokens"], 10)

    def test_parallel_reservations_are_atomic(self):
        tracker = UsageTracker(SchedulerConfig(max_tokens=600, max_output_tokens=200, final_reserve_tokens=0))
        entered, release = threading.Event(), threading.Event()

        def api(**kwargs):
            entered.set()
            release.wait(timeout=2)
            return response()

        def first():
            with usage_scope(tracker):
                tracked_completion(client(api), stage="reasoning", messages=[])

        with ThreadPoolExecutor() as pool:
            future = pool.submit(first)
            self.assertTrue(entered.wait(timeout=2))
            try:
                with usage_scope(tracker), self.assertRaises(BudgetExceeded):
                    tracked_completion(client(api), stage="reasoning", messages=[])
            finally:
                release.set()
            future.result()
        self.assertEqual(tracker.snapshot()["reserved_tokens"], 0)

    def test_trackers_are_isolated_but_endpoint_limit_is_shared(self):
        lock = threading.Lock()
        active = peak = 0
        trackers = [UsageTracker(), UsageTracker()]

        def api(**kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
            time.sleep(0.02)
            with lock:
                active -= 1
            return response()

        def run(tracker):
            with usage_scope(tracker):
                tracked_completion(client(api), stage="reasoning", messages=[])

        with ThreadPoolExecutor() as pool:
            list(pool.map(run, trackers))
        self.assertEqual(peak, 1)
        self.assertEqual([t.snapshot()["known_total_tokens"] for t in trackers], [10, 10])

    def test_deadline_before_call(self):
        tracker = UsageTracker()
        tracker.started -= 1000
        with usage_scope(tracker), self.assertRaises(DeadlineExceeded):
            tracked_completion(client(lambda **kw: self.fail()), stage="reasoning", messages=[])

    def test_python_process_returns_value_and_times_out(self):
        result = execute_python("result = sum(i for i in range(5))\nprint(result)")
        self.assertEqual(result["return_value"], "10")
        self.assertEqual(result["output"].strip(), "10")
        result = execute_python("while True: pass", timeout=0.2)
        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])
