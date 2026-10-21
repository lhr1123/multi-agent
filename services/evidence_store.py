"""Compact, structured evidence passed between agents and to the finalizer."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List
import threading


def _trim(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= 15:
        return text[:limit]
    return text[: limit - 15] + "...[truncated]"


class EvidenceStore:
    def __init__(self, max_chars: int = 6000):
        self.max_chars = max_chars
        self._items: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def compact_result(self, task_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        structured = result.get("structured_response") or {}
        facts = structured.get("key_facts") if isinstance(structured.get("key_facts"), list) else []
        compact_facts = []
        for fact in facts[:8]:
            if isinstance(fact, dict):
                compact_facts.append({str(key): _trim(value, 300) for key, value in fact.items()})
            else:
                compact_facts.append(_trim(fact, 300))
        item = {
            "task_id": task_id,
            "agent_id": result.get("agent_id", ""),
            "action_id": result.get("action_id", ""),
            "status": structured.get("status", "ok" if result.get("success") else "failed"),
            "final_answer": _trim(structured.get("final_answer", ""), 500),
            "result_text": _trim(structured.get("result_text") or result.get("response", ""), 1600),
            "key_facts": compact_facts,
            "confidence": structured.get("confidence", 0.5),
        }
        return item

    def add(self, task_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        item = self.compact_result(task_id, result)
        with self._lock:
            self._items[task_id] = item
        return item

    def dependencies(self, task_ids: Iterable[str], raw_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        items = []
        with self._lock:
            for task_id in task_ids:
                if task_id in self._items:
                    items.append(self._items[task_id])
                elif task_id in raw_results:
                    items.append(self.compact_result(task_id, raw_results[task_id]))
        return self._fit(items)

    def all(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._fit(list(self._items.values()))

    def candidates(self) -> List[Dict[str, Any]]:
        return [
            {"task_id": item["task_id"], "answer": item["final_answer"],
             "confidence": item["confidence"], "agent_id": item["agent_id"]}
            for item in self.all() if item.get("final_answer")
        ]

    def _fit(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        remaining = self.max_chars
        fitted = []
        for item in items:
            copy = dict(item)
            text = str(copy.get("result_text", ""))
            allowance = max(0, min(len(text), remaining))
            copy["result_text"] = _trim(text, allowance) if allowance else ""
            remaining -= len(copy["result_text"])
            fitted.append(copy)
        return fitted
