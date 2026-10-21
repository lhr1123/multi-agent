"""Per-run accounting, reservations and shared endpoint concurrency limits."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import json
import threading
import time

from models import SchedulerConfig


class BudgetExceeded(RuntimeError):
    pass


class DeadlineExceeded(RuntimeError):
    pass


_tracker = ContextVar("usage_tracker", default=None)
_operation = ContextVar("usage_operation", default="")
_gate = threading.Condition()
_active = {}


@contextmanager
def usage_scope(tracker, operation=None):
    token = _tracker.set(tracker)
    op_token = _operation.set(operation) if operation is not None else None
    try:
        yield tracker
    finally:
        if op_token is not None:
            _operation.reset(op_token)
        _tracker.reset(token)


def current_tracker():
    return _tracker.get()


class UsageTracker:
    def __init__(self, config=None):
        self.config = config or SchedulerConfig()
        self.started = time.monotonic()
        self.calls = []
        self.charged = 0
        self.reserved = 0
        self.lock = threading.RLock()

    def remaining_seconds(self):
        return self.config.deadline_seconds - (time.monotonic() - self.started)

    def check(self):
        if self.remaining_seconds() <= 0:
            raise DeadlineExceeded("Run deadline exceeded")

    def reserve(self, amount, final=False):
        with self.lock:
            self.check()
            hold = 0 if final else self.config.final_reserve_tokens
            if self.config.max_tokens and self.charged + self.reserved + amount + hold > self.config.max_tokens:
                raise BudgetExceeded("Insufficient token budget for the next call")
            self.reserved += amount

    def snapshot(self):
        with self.lock:
            calls = [dict(c) for c in self.calls]
            return {
                "known_total_tokens": sum(c.get("total_tokens") or 0 for c in calls),
                "unknown_usage_calls": sum(c["total_tokens"] is None for c in calls),
                "budget_charged_tokens": self.charged,
                "reserved_tokens": self.reserved,
                "calls": calls,
                "config": asdict(self.config),
                "input_estimator": "utf8_bytes_plus_message_overhead; not a tokenizer guarantee",
            }


def tracked_completion(client, *, stage, **kwargs):
    tracker = current_tracker()
    if tracker is None:
        return client.chat.completions.create(**kwargs)
    tracker.check()
    # Byte count is deliberately conservative for text, but provider tokenizers vary.
    input_estimate = len(json.dumps(kwargs.get("messages", []), ensure_ascii=False).encode("utf-8")) + 128
    output_limit = min(kwargs.get("max_tokens", tracker.config.max_output_tokens), tracker.config.max_output_tokens)
    kwargs["max_tokens"] = output_limit
    amount = input_estimate + output_limit
    tracker.reserve(amount, final=stage == "final" or stage.endswith("_final"))
    started = time.monotonic()
    record = {"stage": stage, "operation": _operation.get(), "model": kwargs.get("model"),
              "enable_thinking": (kwargs.get("extra_body") or {}).get("enable_thinking"),
              "estimated_tokens": amount, "total_tokens": None, "prompt_tokens": None,
              "completion_tokens": None, "error": "", "request_sent": False}
    endpoint = str(getattr(client, "base_url", "default"))
    acquired = False
    try:
        with _gate:
            while _active.get(endpoint, 0) >= tracker.config.service_concurrency:
                tracker.check()
                _gate.wait(timeout=min(0.25, tracker.remaining_seconds()))
            tracker.check()
            _active[endpoint] = _active.get(endpoint, 0) + 1
            acquired = True
        # Avoid hidden SDK retries that cannot be accounted individually.
        actual_client = client.with_options(max_retries=0) if hasattr(client, "with_options") else client
        kwargs["timeout"] = max(0.001, tracker.remaining_seconds())
        record["request_sent"] = True
        response = actual_client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                record[key] = getattr(usage, key, None)
        return response
    except Exception as exc:
        record["error"] = type(exc).__name__
        raise
    finally:
        if acquired:
            with _gate:
                _active[endpoint] -= 1
                _gate.notify_all()
        record["elapsed_seconds"] = time.monotonic() - started
        with tracker.lock:
            tracker.reserved -= amount
            if record["request_sent"]:
                tracker.charged += record["total_tokens"] if record["total_tokens"] is not None else amount
                tracker.calls.append(record)
