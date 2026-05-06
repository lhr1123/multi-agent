"""Evaluation chart generation helpers."""

import os
from typing import Any, Dict, List, Optional, Tuple


def _svg_escape(text: Any) -> str:
    s = str(text)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_grouped_bar_svg(
    path: str,
    title: str,
    labels: List[str],
    series: List[Tuple[str, List[float], str]],
    y_max: Optional[float] = None,
) -> None:
    width, height = 980, 560
    m_left, m_right, m_top, m_bottom = 90, 40, 80, 90
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom

    n_groups = max(1, len(labels))
    n_series = max(1, len(series))
    if y_max is None:
        y_max = 0.0
        for _, vals, _ in series:
            for v in vals:
                y_max = max(y_max, float(v))
    y_max = max(1e-9, y_max)

    group_w = plot_w / n_groups
    bar_total_w = group_w * 0.7
    bar_w = bar_total_w / n_series
    base_y = m_top + plot_h

    parts: List[str] = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    parts.append("<rect width='100%' height='100%' fill='white'/>")
    parts.append(f"<text x='{width/2}' y='38' text-anchor='middle' font-size='24' fill='#222'>{_svg_escape(title)}</text>")

    for i in range(6):
        y = m_top + plot_h * (i / 5.0)
        val = y_max * (1 - i / 5.0)
        parts.append(f"<line x1='{m_left}' y1='{y:.2f}' x2='{m_left+plot_w}' y2='{y:.2f}' stroke='#e5e7eb'/>")
        parts.append(
            f"<text x='{m_left-12}' y='{y+4:.2f}' text-anchor='end' font-size='12' fill='#666'>{val:.2f}</text>"
        )

    parts.append(f"<line x1='{m_left}' y1='{base_y}' x2='{m_left+plot_w}' y2='{base_y}' stroke='#111'/>")
    parts.append(f"<line x1='{m_left}' y1='{m_top}' x2='{m_left}' y2='{base_y}' stroke='#111'/>")

    for g, label in enumerate(labels):
        gx = m_left + group_w * g + (group_w - bar_total_w) / 2.0
        for s_idx, (_, vals, color) in enumerate(series):
            v = float(vals[g]) if g < len(vals) else 0.0
            h = 0.0 if y_max <= 0 else (v / y_max) * plot_h
            x = gx + s_idx * bar_w
            y = base_y - h
            parts.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_w*0.85:.2f}' height='{h:.2f}' fill='{color}'/>")
        cx = m_left + group_w * (g + 0.5)
        parts.append(f"<text x='{cx:.2f}' y='{base_y+24}' text-anchor='middle' font-size='13' fill='#333'>{_svg_escape(label)}</text>")

    lx = m_left
    ly = height - 38
    for idx, (name, _, color) in enumerate(series):
        x = lx + idx * 220
        parts.append(f"<rect x='{x}' y='{ly-11}' width='16' height='16' fill='{color}'/>")
        parts.append(f"<text x='{x+24}' y='{ly+2}' font-size='13' fill='#333'>{_svg_escape(name)}</text>")

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def _write_line_svg(
    path: str,
    title: str,
    series: List[Tuple[str, List[float], str]],
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> None:
    width, height = 980, 560
    m_left, m_right, m_top, m_bottom = 90, 40, 80, 90
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom
    base_y = m_top + plot_h

    max_len = max((len(vals) for _, vals, _ in series), default=1)
    max_len = max(1, max_len)
    y_max = max(y_min + 1e-9, y_max)

    parts: List[str] = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>")
    parts.append("<rect width='100%' height='100%' fill='white'/>")
    parts.append(f"<text x='{width/2}' y='38' text-anchor='middle' font-size='24' fill='#222'>{_svg_escape(title)}</text>")

    for i in range(6):
        y = m_top + plot_h * (i / 5.0)
        val = y_max - (y_max - y_min) * (i / 5.0)
        parts.append(f"<line x1='{m_left}' y1='{y:.2f}' x2='{m_left+plot_w}' y2='{y:.2f}' stroke='#e5e7eb'/>")
        parts.append(
            f"<text x='{m_left-12}' y='{y+4:.2f}' text-anchor='end' font-size='12' fill='#666'>{val:.2f}</text>"
        )

    parts.append(f"<line x1='{m_left}' y1='{base_y}' x2='{m_left+plot_w}' y2='{base_y}' stroke='#111'/>")
    parts.append(f"<line x1='{m_left}' y1='{m_top}' x2='{m_left}' y2='{base_y}' stroke='#111'/>")

    for _, vals, color in series:
        if not vals:
            continue
        pts = []
        for i, v in enumerate(vals):
            x = m_left + (i / max(1, max_len - 1)) * plot_w
            ratio = (float(v) - y_min) / (y_max - y_min)
            ratio = max(0.0, min(1.0, ratio))
            y = base_y - ratio * plot_h
            pts.append(f"{x:.2f},{y:.2f}")
        parts.append(f"<polyline fill='none' stroke='{color}' stroke-width='2.5' points='{' '.join(pts)}'/>")

    parts.append(f"<text x='{m_left}' y='{base_y+24}' text-anchor='start' font-size='13' fill='#333'>1</text>")
    parts.append(
        f"<text x='{m_left+plot_w}' y='{base_y+24}' text-anchor='end' font-size='13' fill='#333'>{max_len}</text>"
    )

    lx = m_left
    ly = height - 38
    for idx, (name, _, color) in enumerate(series):
        x = lx + idx * 220
        parts.append(f"<rect x='{x}' y='{ly-11}' width='16' height='16' fill='{color}'/>")
        parts.append(f"<text x='{x+24}' y='{ly+2}' font-size='13' fill='#333'>{_svg_escape(name)}</text>")

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def generate_eval_charts_svg_fallback(summary: Dict[str, Any], save_path: str) -> Dict[str, Any]:
    save_dir = os.path.dirname(save_path) or "."
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    chart_dir = os.path.join(save_dir, "charts", base_name)
    os.makedirs(chart_dir, exist_ok=True)

    samples = summary.get("samples", []) or []
    multi = summary.get("multi_agent_metrics", {}) or {}
    single = summary.get("single_baseline_metrics")
    has_single = isinstance(single, dict) and bool(single)

    files: List[str] = []

    labels_q = ["accuracy", "extraction_rate"]
    series_q: List[Tuple[str, List[float], str]] = [
        (
            "multi-agent",
            [float(multi.get("accuracy", 0.0)), float(multi.get("extraction_rate", 0.0))],
            "#1f77b4",
        )
    ]
    if has_single:
        series_q.append(
            (
                "single-baseline",
                [float(single.get("accuracy", 0.0)), float(single.get("extraction_rate", 0.0))],
                "#ff7f0e",
            )
        )
    p1 = os.path.join(chart_dir, "quality_metrics.svg")
    _write_grouped_bar_svg(p1, "Quality Metrics", labels_q, series_q, y_max=1.0)
    files.append(p1)

    labels_e = ["avg_tokens", "avg_elapsed_seconds"]
    series_e: List[Tuple[str, List[float], str]] = [
        (
            "multi-agent",
            [float(multi.get("avg_tokens", 0.0)), float(multi.get("avg_elapsed_seconds", 0.0))],
            "#1f77b4",
        )
    ]
    if has_single:
        series_e.append(
            (
                "single-baseline",
                [float(single.get("avg_tokens", 0.0)), float(single.get("avg_elapsed_seconds", 0.0))],
                "#ff7f0e",
            )
        )
    p2 = os.path.join(chart_dir, "efficiency_metrics.svg")
    _write_grouped_bar_svg(p2, "Efficiency Metrics", labels_e, series_e)
    files.append(p2)

    multi_flags = [1 if (s.get("multi_agent", {}) or {}).get("is_correct") else 0 for s in samples]
    multi_cum = []
    run = 0
    for i, v in enumerate(multi_flags, 1):
        run += v
        multi_cum.append(run / i)
    series_l: List[Tuple[str, List[float], str]] = [("multi-agent", multi_cum, "#1f77b4")]
    if has_single:
        single_flags = [1 if (s.get("single_baseline", {}) or {}).get("is_correct") else 0 for s in samples]
        single_cum = []
        run2 = 0
        for i, v in enumerate(single_flags, 1):
            run2 += v
            single_cum.append(run2 / i)
        series_l.append(("single-baseline", single_cum, "#ff7f0e"))
    p3 = os.path.join(chart_dir, "cumulative_accuracy.svg")
    _write_line_svg(p3, "Cumulative Accuracy Curve", series_l, y_min=0.0, y_max=1.0)
    files.append(p3)

    total = int(multi.get("total", len(samples)) or len(samples))
    correct = int(multi.get("correct", 0))
    failed = int(multi.get("failed_runs", 0))
    wrong = max(0, total - correct - failed)
    p4 = os.path.join(chart_dir, "result_breakdown.svg")
    _write_grouped_bar_svg(
        p4,
        "Multi-agent Result Breakdown",
        ["correct", "wrong", "failed"],
        [("multi-agent", [float(correct), float(wrong), float(failed)], "#2ca02c")],
    )
    files.append(p4)

    return {
        "enabled": True,
        "backend": "svg-fallback",
        "error": None,
        "chart_dir": chart_dir,
        "files": files,
    }


def generate_eval_charts(summary: Dict[str, Any], save_path: str) -> Dict[str, Any]:
    """
    Generate evaluation charts (PNG) next to the JSON output.
    If matplotlib is unavailable, return gracefully without raising.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        fallback = generate_eval_charts_svg_fallback(summary, save_path=save_path)
        fallback["error"] = f"matplotlib unavailable: {e}"
        return fallback

    save_dir = os.path.dirname(save_path) or "."
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    chart_dir = os.path.join(save_dir, "charts", base_name)
    os.makedirs(chart_dir, exist_ok=True)

    created_files: List[str] = []
    samples = summary.get("samples", []) or []
    multi = summary.get("multi_agent_metrics", {}) or {}
    single = summary.get("single_baseline_metrics")
    has_single = isinstance(single, dict) and bool(single)

    labels = ["accuracy", "extraction_rate"]
    multi_vals = [float(multi.get("accuracy", 0.0)), float(multi.get("extraction_rate", 0.0))]
    single_vals = (
        [float(single.get("accuracy", 0.0)), float(single.get("extraction_rate", 0.0))] if has_single else []
    )
    x = list(range(len(labels)))
    plt.figure(figsize=(8, 5))
    if has_single:
        width = 0.35
        plt.bar([xi - width / 2 for xi in x], multi_vals, width=width, label="multi-agent")
        plt.bar([xi + width / 2 for xi in x], single_vals, width=width, label="single-baseline")
        plt.legend()
    else:
        plt.bar(x, multi_vals, width=0.5, label="multi-agent")
    plt.ylim(0.0, 1.0)
    plt.xticks(x, labels)
    plt.ylabel("score")
    plt.title("Quality Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    p1 = os.path.join(chart_dir, "quality_metrics.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()
    created_files.append(p1)

    labels_eff = ["avg_tokens", "avg_elapsed_seconds"]
    multi_eff = [float(multi.get("avg_tokens", 0.0)), float(multi.get("avg_elapsed_seconds", 0.0))]
    single_eff = (
        [float(single.get("avg_tokens", 0.0)), float(single.get("avg_elapsed_seconds", 0.0))]
        if has_single
        else []
    )
    x2 = list(range(len(labels_eff)))
    plt.figure(figsize=(8, 5))
    if has_single:
        width = 0.35
        plt.bar([xi - width / 2 for xi in x2], multi_eff, width=width, label="multi-agent")
        plt.bar([xi + width / 2 for xi in x2], single_eff, width=width, label="single-baseline")
        plt.legend()
    else:
        plt.bar(x2, multi_eff, width=0.5, label="multi-agent")
    plt.xticks(x2, labels_eff)
    plt.ylabel("value")
    plt.title("Efficiency Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    p2 = os.path.join(chart_dir, "efficiency_metrics.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()
    created_files.append(p2)

    multi_flags = [1 if (s.get("multi_agent", {}) or {}).get("is_correct") else 0 for s in samples]
    multi_cum = []
    running = 0
    for i, v in enumerate(multi_flags, 1):
        running += v
        multi_cum.append(running / i)

    plt.figure(figsize=(10, 5))
    if multi_cum:
        plt.plot(range(1, len(multi_cum) + 1), multi_cum, label="multi-agent", linewidth=2)

    if has_single:
        single_flags = [1 if (s.get("single_baseline", {}) or {}).get("is_correct") else 0 for s in samples]
        single_cum = []
        running2 = 0
        for i, v in enumerate(single_flags, 1):
            running2 += v
            single_cum.append(running2 / i)
        if single_cum:
            plt.plot(range(1, len(single_cum) + 1), single_cum, label="single-baseline", linewidth=2)
        plt.legend()

    plt.ylim(0.0, 1.0)
    plt.xlabel("sample index")
    plt.ylabel("cumulative accuracy")
    plt.title("Cumulative Accuracy Curve")
    plt.grid(linestyle="--", alpha=0.3)
    p3 = os.path.join(chart_dir, "cumulative_accuracy.png")
    plt.tight_layout()
    plt.savefig(p3, dpi=160)
    plt.close()
    created_files.append(p3)

    total = int(multi.get("total", len(samples)) or len(samples))
    correct = int(multi.get("correct", 0))
    failed = int(multi.get("failed_runs", 0))
    wrong = max(0, total - correct - failed)
    vals = [correct, wrong, failed]
    labels_break = ["correct", "wrong", "failed"]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]
    plt.figure(figsize=(6, 6))
    plt.pie(vals, labels=labels_break, autopct="%1.1f%%", colors=colors, startangle=120)
    plt.title("Multi-agent Result Breakdown")
    p4 = os.path.join(chart_dir, "result_breakdown.png")
    plt.tight_layout()
    plt.savefig(p4, dpi=160)
    plt.close()
    created_files.append(p4)

    return {
        "enabled": True,
        "backend": "matplotlib",
        "error": None,
        "chart_dir": chart_dir,
        "files": created_files,
    }
