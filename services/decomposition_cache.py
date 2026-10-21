"""Stable JSONL cache for replaying the same decomposition across schedulers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading


PLANNER_VERSION = "minimal-plan-v2"
_LOCK = threading.RLock()


def plan_cache_key(title, description, model_name, max_tasks, allow_direct):
    payload = {
        "title": title,
        "description": description,
        "model": model_name,
        "max_tasks": max_tasks,
        "allow_direct": allow_direct,
        "planner_version": PLANNER_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


class DecompositionCache:
    def __init__(self, path=""):
        self.path = Path(path) if path else None

    def get(self, key):
        if not self.path or not self.path.exists():
            return None
        with _LOCK:
            found = None
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("key") == key and isinstance(record.get("response"), dict):
                        found = record["response"]
            return found

    def put(self, key, response):
        if not self.path:
            return
        with _LOCK:
            if self.get(key) is not None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
