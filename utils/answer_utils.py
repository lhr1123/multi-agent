"""Answer extraction and evaluation helpers."""

import random
import re
from typing import Any, Dict, List, Optional, Tuple


def extract_last_number(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text)
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def extract_first_number(text: str) -> Optional[float]:
    if text is None:
        return None
    s = str(text)
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
    if not match:
        return None
    raw = match.group(0).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _coerce_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return extract_first_number(str(value))


def extract_number_from_final_answer_field(text: str) -> Optional[float]:
    """Extract a number explicitly attached to a final-answer label."""
    if text is None:
        return None
    s = str(text)
    number = r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?"
    patterns = [
        rf'"final_answer"\s*:\s*"?\s*({number})',
        rf"'final_answer'\s*:\s*'?\s*({number})",
        rf"\bfinal[_\s-]*answer\b\s*[:：=]\s*[\(\[`'\"]*\s*({number})",
        rf"\banswer\b\s*[:：=]\s*[\(\[`'\"]*\s*({number})",
        rf"答案\s*[:：=]\s*[\(\[`'\"]*\s*({number})",
    ]
    for pattern in patterns:
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).replace(",", "")
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def is_correct_prediction(pred: Optional[float], target: Optional[float], tol: float = 1e-6) -> bool:
    if pred is None or target is None:
        return False
    try:
        p = float(pred)
        t = float(target)
    except (TypeError, ValueError):
        return False
    return abs(p - t) <= tol * max(1.0, abs(t))


def extract_best_number_from_sources(sources: List[Tuple[str, str, float]]) -> Optional[float]:
    """
    Extract one robust numeric answer from multiple text sources.
    Source tuple: (name, text, base_weight)
    """
    number_re = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?")
    positive_hints = [
        "final answer",
        "answer",
        "result",
        "therefore",
        "hence",
        "total",
        "=",
    ]
    negative_hints = [
        "cannot",
        "unable",
        "failed",
        "not enough",
        "unknown",
        "irrelevant",
        "not consistent",
        "conflicting",
    ]
    intermediate_hints = [
        "overtime",
        "regular pay",
        "subtotal",
        "per hour",
        "per dozen",
        "each",
        "times",
    ]

    scored: Dict[float, float] = {}
    freq: Dict[float, int] = {}

    for _, text, base_weight in sources:
        if not text:
            continue
        s = str(text)
        lower = s.lower()
        for m in number_re.finditer(s):
            raw = m.group(0).replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue

            left = lower[max(0, m.start() - 48) : m.start()]
            right = lower[m.end() : min(len(lower), m.end() + 48)]
            context = f"{left} {right}"

            score = float(base_weight)
            if any(h in context for h in positive_hints):
                score += 1.4
            if any(h in context for h in negative_hints):
                score -= 2.0
            if any(h in context for h in intermediate_hints):
                score -= 0.6
            if "http" in context or "www." in context:
                score -= 1.5

            score += 0.05 * (m.end() / max(1, len(s)))

            key = round(value, 10)
            scored[key] = scored.get(key, 0.0) + score
            freq[key] = freq.get(key, 0) + 1

    if not scored:
        return None

    best_value = None
    best_score = -10**9
    for value, sc in scored.items():
        total_score = sc + 0.15 * max(0, freq.get(value, 0) - 1)
        if total_score > best_score:
            best_score = total_score
            best_value = value

    if best_value is None:
        return None
    if best_score < 0.3:
        return None
    return float(best_value)


def extract_prediction_from_multi_result(run_result: Dict[str, Any]) -> Optional[float]:
    final_result = str(run_result.get("final_result", "") or "").strip()

    sources: List[Tuple[str, str, float]] = []

    terminate_step = run_result.get("terminate_step", {}) or {}
    if isinstance(terminate_step, dict):
        terminate_structured = terminate_step.get("structured_response", {}) or {}
        if isinstance(terminate_structured, dict):
            fa_raw = terminate_structured.get("final_answer", "")
            direct = _coerce_number(fa_raw)
            if direct is not None:
                return direct
            fa = str(fa_raw or "")
            rt = str(terminate_structured.get("result_text", "") or "")
            conf = terminate_structured.get("confidence", 0.7)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.7
            conf = max(0.0, min(1.0, conf))
            base = 3.5 + 1.2 * conf
            if fa:
                sources.append(("terminate:final_answer", fa, base))
            if rt and rt != fa:
                sources.append(("terminate:result_text", rt, base * 0.8))
        resp = str(terminate_step.get("response", "") or "")
        if resp:
            direct = extract_number_from_final_answer_field(resp)
            if direct is not None:
                return direct
            sources.append(("terminate:response", resp, 2.6))

    if final_result:
        direct = extract_number_from_final_answer_field(final_result)
        if direct is not None:
            return direct
        sources.append(("final_result", final_result, 4.8))

    pred = extract_best_number_from_sources(sources)
    if pred is not None:
        return pred
    return extract_last_number(final_result)


def extract_option_letter(text: str) -> Optional[str]:
    """Extract a multiple-choice answer option letter from free-form text."""
    if text is None:
        return None

    s = str(text).strip()
    patterns = [
        r"answer\s+is\s*\(?\s*([A-J])\s*\)?",
        r"final\s+answer\s*[:：]?\s*\(?\s*([A-J])\s*\)?",
        r"option\s*\(?\s*([A-J])\s*\)?",
        r"答案\s*[:：]?\s*\(?\s*([A-J])\s*\)?",
        r"选择\s*\(?\s*([A-J])\s*\)?",
        r"^\(?\s*([A-J])\s*\)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    standalone = re.findall(r"\b([A-J])\b", s.upper())
    if standalone:
        return standalone[-1]
    return None


def extract_option_prediction_from_multi_result(run_result: Dict[str, Any]) -> Optional[str]:
    """Extract a multiple-choice answer letter from multi-agent outputs."""
    candidate_texts: List[str] = []

    terminate_step = run_result.get("terminate_step", {}) or {}
    if isinstance(terminate_step, dict):
        terminate_structured = terminate_step.get("structured_response", {}) or {}
        if isinstance(terminate_structured, dict):
            final_answer = str(terminate_structured.get("final_answer", "") or "")
            result_text = str(terminate_structured.get("result_text", "") or "")
            if final_answer:
                candidate_texts.append(final_answer)
            if result_text and result_text != final_answer:
                candidate_texts.append(result_text)
        response = str(terminate_step.get("response", "") or "")
        if response:
            candidate_texts.append(response)

    final_result = str(run_result.get("final_result", "") or "")
    if final_result:
        candidate_texts.append(final_result)

    for text in candidate_texts:
        letter = extract_option_letter(text)
        if letter:
            return letter
    return None


def is_correct_option_prediction(pred: Optional[str], target: Optional[str]) -> bool:
    if pred is None or target is None:
        return False
    return str(pred).strip().upper() == str(target).strip().upper()


def random_option_fallback(option_count: int) -> str:
    labels = "ABCDEFGHIJ"
    upper = max(1, min(len(labels), option_count))
    return random.choice(list(labels[:upper]))
