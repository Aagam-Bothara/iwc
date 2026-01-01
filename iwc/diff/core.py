from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from iwc.analyze.summary import WorkloadSummary


def _is_nan(x: float) -> bool:
    return x != x


def _fmt_num(x: float, nd: int = 2) -> str:
    if _is_nan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _fmt_int(x: float) -> str:
    if _is_nan(x):
        return "n/a"
    return f"{int(round(x))}"


def _fmt_delta(a: float, b: float, nd: int = 2, signed: bool = True) -> str:
    if _is_nan(a) or _is_nan(b):
        return "n/a"
    d = b - a
    if signed:
        return f"{d:+.{nd}f}"
    return f"{d:.{nd}f}"


def _delta_is_zero(delta: str) -> bool:
    """
    Treat these as "no change" for text output filtering:
      "", "+0", "+0.0", "+0.00", "+0.000", "0", "0.00", etc.
    """
    if not delta:
        return True
    d = delta.strip()
    if d == "n/a":
        return False
    if d.startswith("+"):
        d = d[1:]
    if d.startswith("-"):
        d = d[1:]
    try:
        return float(d) == 0.0
    except ValueError:
        return False


@dataclass(frozen=True)
class FieldDiff:
    label: str
    a: str
    b: str
    delta: str


@dataclass(frozen=True)
class SummaryDiff:
    a: WorkloadSummary
    b: WorkloadSummary
    rows: List[FieldDiff]


def diff_summaries(a: WorkloadSummary, b: WorkloadSummary) -> SummaryDiff:
    rows: List[FieldDiff] = []

    rows.append(FieldDiff("Tokenizer", a.tokenizer_used, b.tokenizer_used, ""))
    rows.append(FieldDiff("Requests", str(a.requests), str(b.requests), _fmt_delta(float(a.requests), float(b.requests), 0)))

    rows.append(FieldDiff("Prompt tokens P50", _fmt_int(a.prompt_tokens.p50), _fmt_int(b.prompt_tokens.p50), _fmt_delta(a.prompt_tokens.p50, b.prompt_tokens.p50, 0)))
    rows.append(FieldDiff("Prompt tokens P90", _fmt_int(a.prompt_tokens.p90), _fmt_int(b.prompt_tokens.p90), _fmt_delta(a.prompt_tokens.p90, b.prompt_tokens.p90, 0)))
    rows.append(FieldDiff("Prompt tokens P99", _fmt_int(a.prompt_tokens.p99), _fmt_int(b.prompt_tokens.p99), _fmt_delta(a.prompt_tokens.p99, b.prompt_tokens.p99, 0)))

    rows.append(FieldDiff("Max output cap P90", _fmt_int(a.max_output_tokens.p90), _fmt_int(b.max_output_tokens.p90), _fmt_delta(a.max_output_tokens.p90, b.max_output_tokens.p90, 0)))

    rows.append(FieldDiff("Prefill dominance P50", _fmt_num(a.prefill_dominance.p50, 3), _fmt_num(b.prefill_dominance.p50, 3), _fmt_delta(a.prefill_dominance.p50, b.prefill_dominance.p50, 3)))
    rows.append(FieldDiff("Prefill dominance P90", _fmt_num(a.prefill_dominance.p90, 3), _fmt_num(b.prefill_dominance.p90, 3), _fmt_delta(a.prefill_dominance.p90, b.prefill_dominance.p90, 3)))

    rows.append(FieldDiff("Duration (s)", _fmt_num(a.arrivals.duration_s, 2), _fmt_num(b.arrivals.duration_s, 2), _fmt_delta(a.arrivals.duration_s, b.arrivals.duration_s, 2)))
    rows.append(FieldDiff("Mean RPS", _fmt_num(a.arrivals.mean_rps, 2), _fmt_num(b.arrivals.mean_rps, 2), _fmt_delta(a.arrivals.mean_rps, b.arrivals.mean_rps, 2)))
    rows.append(FieldDiff("Peak reqs (1s bin)", _fmt_int(a.arrivals.peak_rps_1s), _fmt_int(b.arrivals.peak_rps_1s), _fmt_delta(a.arrivals.peak_rps_1s, b.arrivals.peak_rps_1s, 0)))
    rows.append(FieldDiff("Inter-arrival ms P50", _fmt_int(a.arrivals.interarrival_ms.p50), _fmt_int(b.arrivals.interarrival_ms.p50), _fmt_delta(a.arrivals.interarrival_ms.p50, b.arrivals.interarrival_ms.p50, 0)))
    rows.append(FieldDiff("Inter-arrival ms P90", _fmt_int(a.arrivals.interarrival_ms.p90), _fmt_int(b.arrivals.interarrival_ms.p90), _fmt_delta(a.arrivals.interarrival_ms.p90, b.arrivals.interarrival_ms.p90, 0)))
    rows.append(FieldDiff("Burstiness (CV)", _fmt_num(a.arrivals.burstiness_cv, 2), _fmt_num(b.arrivals.burstiness_cv, 2), _fmt_delta(a.arrivals.burstiness_cv, b.arrivals.burstiness_cv, 2)))

    rows.append(FieldDiff("Sessions detected", str(a.sessions.sessions_detected), str(b.sessions.sessions_detected), _fmt_delta(float(a.sessions.sessions_detected), float(b.sessions.sessions_detected), 0)))
    rows.append(FieldDiff("Turns/session P90", _fmt_int(a.sessions.turns_per_session.p90), _fmt_int(b.sessions.turns_per_session.p90), _fmt_delta(a.sessions.turns_per_session.p90, b.sessions.turns_per_session.p90, 0)))
    rows.append(FieldDiff("Prompt reuse (tokens)", _fmt_num(a.sessions.prompt_reuse_ratio_tokens, 3), _fmt_num(b.sessions.prompt_reuse_ratio_tokens, 3), _fmt_delta(a.sessions.prompt_reuse_ratio_tokens, b.sessions.prompt_reuse_ratio_tokens, 3)))
    rows.append(FieldDiff("Prompt tokens/turn P50", _fmt_int(a.sessions.prompt_tokens_by_turn.p50), _fmt_int(b.sessions.prompt_tokens_by_turn.p50), _fmt_delta(a.sessions.prompt_tokens_by_turn.p50, b.sessions.prompt_tokens_by_turn.p50, 0)))
    rows.append(FieldDiff("Prompt tokens/turn P90", _fmt_int(a.sessions.prompt_tokens_by_turn.p90), _fmt_int(b.sessions.prompt_tokens_by_turn.p90), _fmt_delta(a.sessions.prompt_tokens_by_turn.p90, b.sessions.prompt_tokens_by_turn.p90, 0)))
    rows.append(FieldDiff("Δtokens/turn P50", _fmt_int(a.sessions.prompt_token_growth.p50), _fmt_int(b.sessions.prompt_token_growth.p50), _fmt_delta(a.sessions.prompt_token_growth.p50, b.sessions.prompt_token_growth.p50, 0)))
    rows.append(FieldDiff("Δtokens/turn P90", _fmt_int(a.sessions.prompt_token_growth.p90), _fmt_int(b.sessions.prompt_token_growth.p90), _fmt_delta(a.sessions.prompt_token_growth.p90, b.sessions.prompt_token_growth.p90, 0)))

    return SummaryDiff(a=a, b=b, rows=rows)


def _infer_primary(s: WorkloadSummary) -> str:
    if s.sessions.sessions_detected and not _is_nan(s.sessions.prompt_reuse_ratio_tokens) and s.sessions.prompt_reuse_ratio_tokens > 0.5:
        if not _is_nan(s.prefill_dominance.p50) and s.prefill_dominance.p50 > 0.65:
            return "interactive-chat (prefill-heavy)"
        return "interactive-chat"
    if not _is_nan(s.arrivals.burstiness_cv) and s.arrivals.burstiness_cv > 1.5:
        return "bursty-api"
    return "batch/offline"


def _direction_hint(a: WorkloadSummary, b: WorkloadSummary) -> str:
    hints: List[str] = []

    if not _is_nan(a.arrivals.burstiness_cv) and not _is_nan(b.arrivals.burstiness_cv):
        if b.arrivals.burstiness_cv - a.arrivals.burstiness_cv > 0.5:
            hints.append("more bursty")
        elif a.arrivals.burstiness_cv - b.arrivals.burstiness_cv > 0.5:
            hints.append("less bursty")

    if not _is_nan(a.prefill_dominance.p50) and not _is_nan(b.prefill_dominance.p50):
        if b.prefill_dominance.p50 - a.prefill_dominance.p50 > 0.05:
            hints.append("more prefill-heavy")
        elif a.prefill_dominance.p50 - b.prefill_dominance.p50 > 0.05:
            hints.append("less prefill-heavy")

    if not _is_nan(a.sessions.prompt_reuse_ratio_tokens) and not _is_nan(b.sessions.prompt_reuse_ratio_tokens):
        if b.sessions.prompt_reuse_ratio_tokens - a.sessions.prompt_reuse_ratio_tokens > 0.05:
            hints.append("higher reuse")
        elif a.sessions.prompt_reuse_ratio_tokens - b.sessions.prompt_reuse_ratio_tokens > 0.05:
            hints.append("lower reuse")

    return ", ".join(hints) if hints else "no major shift detected"


def render_diff(d: SummaryDiff, a_label: str = "A", b_label: str = "B", only_changed: bool = False) -> str:
    lines: List[str] = []
    lines.append("WORKLOAD DIFF")
    lines.append("-------------")
    lines.append(f"A (baseline) : {a_label}")
    lines.append(f"B (candidate): {b_label}")
    lines.append("")
    lines.append(f"Primary class A : {_infer_primary(d.a)}")
    lines.append(f"Primary class B : {_infer_primary(d.b)}")
    lines.append(f"Shift           : {_direction_hint(d.a, d.b)}")
    lines.append("")

    rows = d.rows
    if only_changed:
        rows = [r for r in rows if not _delta_is_zero(r.delta)]

    col1 = max(len(r.label) for r in rows) if rows else 10
    col2 = max(len(r.a) for r in rows) if rows else 10
    col3 = max(len(r.b) for r in rows) if rows else 10

    header = f"{'Metric'.ljust(col1)}  {'A'.ljust(col2)}  {'B'.ljust(col3)}  Δ(B-A)"
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        lines.append(f"{r.label.ljust(col1)}  {r.a.ljust(col2)}  {r.b.ljust(col3)}  {r.delta}")

    return "\n".join(lines)


def diff_to_dict(d: SummaryDiff, a_label: str = "A", b_label: str = "B") -> Dict[str, Any]:
    return {
        "a_label": a_label,
        "b_label": b_label,
        "primary_class_a": _infer_primary(d.a),
        "primary_class_b": _infer_primary(d.b),
        "shift": _direction_hint(d.a, d.b),
        "metrics": [
            {"metric": r.label, "a": r.a, "b": r.b, "delta": r.delta}
            for r in d.rows
        ],
    }


def check_regressions(
    d: SummaryDiff,
    burstiness_delta: float | None = None,
    prefill_p50_delta: float | None = None,
    reuse_delta: float | None = None,
    prompt_p50_delta: float | None = None,
    prompt_p90_delta: float | None = None,
) -> List[str]:
    """
    Returns a list of human-readable regression strings.
    If empty -> OK.
    All checks are absolute deltas |B-A|.
    """
    msgs: List[str] = []

    a = d.a
    b = d.b

    def abs_delta(xa: float, xb: float) -> float | None:
        if _is_nan(xa) or _is_nan(xb):
            return None
        return abs(xb - xa)

    # Burstiness CV
    if burstiness_delta is not None:
        v = abs_delta(a.arrivals.burstiness_cv, b.arrivals.burstiness_cv)
        if v is not None and v > burstiness_delta:
            msgs.append(f"Burstiness CV changed by {v:.3f} (> {burstiness_delta:.3f})")

    # Prefill dominance P50
    if prefill_p50_delta is not None:
        v = abs_delta(a.prefill_dominance.p50, b.prefill_dominance.p50)
        if v is not None and v > prefill_p50_delta:
            msgs.append(f"Prefill dominance P50 changed by {v:.3f} (> {prefill_p50_delta:.3f})")

    # Reuse (tokens)
    if reuse_delta is not None:
        v = abs_delta(a.sessions.prompt_reuse_ratio_tokens, b.sessions.prompt_reuse_ratio_tokens)
        if v is not None and v > reuse_delta:
            msgs.append(f"Prompt reuse (tokens) changed by {v:.3f} (> {reuse_delta:.3f})")

    # Prompt tokens P50/P90
    if prompt_p50_delta is not None:
        v = abs_delta(a.prompt_tokens.p50, b.prompt_tokens.p50)
        if v is not None and v > prompt_p50_delta:
            msgs.append(f"Prompt tokens P50 changed by {v:.1f} (> {prompt_p50_delta:.1f})")

    if prompt_p90_delta is not None:
        v = abs_delta(a.prompt_tokens.p90, b.prompt_tokens.p90)
        if v is not None and v > prompt_p90_delta:
            msgs.append(f"Prompt tokens P90 changed by {v:.1f} (> {prompt_p90_delta:.1f})")

    return msgs


# --------------------------
# Diff Lite (CORE DIFF)
# --------------------------

CORE_THRESHOLDS = {
    "prefill_p90_abs": 0.05,      # |Δ| > 0.05
    "prompt_p90_rel": 0.10,       # |Δ|/max(A,B) > 10%
    "burstiness_abs": 0.50,       # |Δ| > 0.5
    "mean_rps_rel_a": 0.10,       # |Δ|/A > 10%
    "reuse_abs": 0.05,            # |Δ| > 0.05 (only when both have sessions)
}


def _abs_delta(a: float, b: float) -> float | None:
    if _is_nan(a) or _is_nan(b):
        return None
    return abs(b - a)


def _rel_delta_over_max(a: float, b: float) -> float | None:
    if _is_nan(a) or _is_nan(b):
        return None
    denom = max(abs(a), abs(b), 1e-9)
    return abs(b - a) / denom


def _rel_delta_over_a(a: float, b: float) -> float | None:
    if _is_nan(a) or _is_nan(b):
        return None
    denom = max(abs(a), 1e-9)
    return abs(b - a) / denom


@dataclass(frozen=True)
class CoreRow:
    metric: str
    a: str
    b: str
    delta: str
    status: str  # "OK" or "FLAG"
    reason: str  # threshold explanation (for JSON / debugging)


def build_core_diff(d: SummaryDiff) -> Tuple[List[CoreRow], List[str], bool]:
    """
    Core metrics (3–5) + flags.

    Rules:
    - Never output NaN as a metric row (omit it).
    - If sessions exist on one side but not the other, emit a structural FLAG line.
    """
    a = d.a
    b = d.b

    rows: List[CoreRow] = []
    structural: List[str] = []
    any_flag = False

    def add_row(metric: str, a_val: float, b_val: float, nd: int, flag: bool, reason: str, as_int: bool = False) -> None:
        nonlocal any_flag
        if _is_nan(a_val) or _is_nan(b_val):
            return

        a_s = _fmt_int(a_val) if as_int else _fmt_num(a_val, nd)
        b_s = _fmt_int(b_val) if as_int else _fmt_num(b_val, nd)
        delta_s = _fmt_delta(a_val, b_val, 0 if as_int else nd)

        rows.append(
            CoreRow(
                metric=metric,
                a=a_s,
                b=b_s,
                delta=delta_s,
                status="FLAG" if flag else "OK",
                reason=reason,
            )
        )
        any_flag = any_flag or flag

    # 1) Prefill dominance P90 (abs delta)
    ap = a.prefill_dominance.p90
    bp = b.prefill_dominance.p90
    v = _abs_delta(ap, bp)
    if v is not None:
        thr = CORE_THRESHOLDS["prefill_p90_abs"]
        add_row(
            "Prefill dominance P90",
            ap,
            bp,
            nd=3,
            flag=(v > thr),
            reason=f"|Δ|={v:.3f} > {thr:.3f}",
            as_int=False,
        )

    # 2) Prompt tokens P90 (relative over max)
    at = a.prompt_tokens.p90
    bt = b.prompt_tokens.p90
    rel = _rel_delta_over_max(at, bt)
    if rel is not None:
        thr = CORE_THRESHOLDS["prompt_p90_rel"]
        add_row(
            "Prompt tokens P90",
            at,
            bt,
            nd=0,
            flag=(rel > thr),
            reason=f"|Δ|/max={rel:.3f} > {thr:.3f}",
            as_int=True,
        )

    # 3) Burstiness CV (abs delta)
    acv = a.arrivals.burstiness_cv
    bcv = b.arrivals.burstiness_cv
    v = _abs_delta(acv, bcv)
    if v is not None:
        thr = CORE_THRESHOLDS["burstiness_abs"]
        add_row(
            "Burstiness (CV)",
            acv,
            bcv,
            nd=2,
            flag=(v > thr),
            reason=f"|Δ|={v:.3f} > {thr:.3f}",
            as_int=False,
        )

    # 4) Mean RPS (relative over A)
    arps = a.arrivals.mean_rps
    brps = b.arrivals.mean_rps
    rel = _rel_delta_over_a(arps, brps)
    if rel is not None:
        thr = CORE_THRESHOLDS["mean_rps_rel_a"]
        add_row(
            "Mean RPS",
            arps,
            brps,
            nd=2,
            flag=(rel > thr),
            reason=f"|Δ|/A={rel:.3f} > {thr:.3f}",
            as_int=False,
        )

    # 5) Prompt reuse ratio (tokens): only meaningful if both have sessions.
    a_has = bool(a.sessions.sessions_detected)
    b_has = bool(b.sessions.sessions_detected)

    if a_has != b_has:
        structural.append(
            f"Sessions mismatch: A={'present' if a_has else 'none'}, B={'present' if b_has else 'none'} (FLAG)"
        )
        any_flag = True
    elif a_has and b_has:
        aru = a.sessions.prompt_reuse_ratio_tokens
        bru = b.sessions.prompt_reuse_ratio_tokens
        v = _abs_delta(aru, bru)
        if v is not None:
            thr = CORE_THRESHOLDS["reuse_abs"]
            add_row(
                "Prompt reuse ratio (tokens)",
                aru,
                bru,
                nd=3,
                flag=(v > thr),
                reason=f"|Δ|={v:.3f} > {thr:.3f}",
                as_int=False,
            )

    return rows, structural, any_flag


def core_diff_to_dict(d: SummaryDiff, a_label: str, b_label: str) -> Dict[str, Any]:
    rows, structural, any_flag = build_core_diff(d)
    return {
        "a_label": a_label,
        "b_label": b_label,
        "any_flag": any_flag,
        "thresholds": CORE_THRESHOLDS,
        "structural_flags": structural,
        "metrics": [
            {
                "metric": r.metric,
                "a": r.a,
                "b": r.b,
                "delta": r.delta,
                "status": r.status,
                "reason": r.reason,
            }
            for r in rows
        ],
    }


def render_core_diff(d: SummaryDiff, a_label: str = "A", b_label: str = "B") -> Tuple[str, bool]:
    rows, structural, any_flag = build_core_diff(d)

    lines: List[str] = []
    lines.append("CORE DIFF")
    lines.append("---------")
    lines.append(f"A (baseline) : {a_label}")
    lines.append(f"B (candidate): {b_label}")
    lines.append("")

    for s in structural:
        lines.append(s)
    if structural:
        lines.append("")

    if not rows:
        lines.append("(no core metrics applicable)")
        return "\n".join(lines), any_flag

    col1 = max(len(r.metric) for r in rows)
    col2 = max(len(r.a) for r in rows)
    col3 = max(len(r.b) for r in rows)

    header = f"{'Metric'.ljust(col1)}  {'A'.ljust(col2)}  {'B'.ljust(col3)}  Δ(B-A)   Status"
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        lines.append(f"{r.metric.ljust(col1)}  {r.a.ljust(col2)}  {r.b.ljust(col3)}  {r.delta.ljust(7)}  {r.status}")

    return "\n".join(lines), any_flag
