from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            yield obj


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def _percentile(sorted_vals: list[int], p: float) -> Optional[int]:
    """Nearest-rank percentile. p in [0,100]."""
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = int((p / 100.0) * (len(sorted_vals) - 1))
    return sorted_vals[k]


@dataclass
class WorkloadReport:
    num_requests: int
    min_arrival_ms: Optional[int]
    max_arrival_ms: Optional[int]
    arrival_span_ms: int

    mot_min: Optional[int]
    mot_avg: Optional[float]
    mot_p50: Optional[int]
    mot_p90: Optional[int]
    mot_p99: Optional[int]
    mot_max: Optional[int]

    prompt_chars_min: Optional[int]
    prompt_chars_avg: Optional[float]
    prompt_chars_p50: Optional[int]
    prompt_chars_p90: Optional[int]
    prompt_chars_p99: Optional[int]
    prompt_chars_max: Optional[int]

    task_counts: Counter[str]
    tag_counts: Counter[str]


def build_report(path: Path) -> WorkloadReport:
    arrivals: list[int] = []
    max_out: list[int] = []
    prompt_chars: list[int] = []

    task_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()

    n = 0
    for req in read_jsonl(path):
        n += 1

        # arrival_time_ms (required by schema)
        at = req.get("arrival_time_ms")
        if isinstance(at, int):
            arrivals.append(at)

        # max_output_tokens (required by schema)
        mot = req.get("max_output_tokens")
        if isinstance(mot, int):
            max_out.append(mot)

        # prompt (required by schema)
        prompt = req.get("prompt")
        if isinstance(prompt, str):
            prompt_chars.append(len(prompt))

        sem = req.get("semantic")
        if isinstance(sem, dict):
            task = sem.get("task")
            if isinstance(task, str) and task.strip():
                task_counts[task] += 1
            tags = sem.get("tags")
            if isinstance(tags, list):
                for t in tags:
                    if isinstance(t, str) and t.strip():
                        tag_counts[t] += 1

    min_at = min(arrivals) if arrivals else None
    max_at = max(arrivals) if arrivals else None
    span = int(max_at - min_at) if (min_at is not None and max_at is not None) else 0

    def stats(vals: list[int]) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if not vals:
            return (None, None, None, None, None, None)
        s = sorted(vals)
        avg = sum(s) / len(s)
        return (
            s[0],
            avg,
            _percentile(s, 50),
            _percentile(s, 90),
            _percentile(s, 99),
            s[-1],
        )

    mot_min, mot_avg, mot_p50, mot_p90, mot_p99, mot_max = stats(max_out)
    pc_min, pc_avg, pc_p50, pc_p90, pc_p99, pc_max = stats(prompt_chars)

    return WorkloadReport(
        num_requests=n,
        min_arrival_ms=min_at,
        max_arrival_ms=max_at,
        arrival_span_ms=span,
        mot_min=mot_min,
        mot_avg=mot_avg,
        mot_p50=mot_p50,
        mot_p90=mot_p90,
        mot_p99=mot_p99,
        mot_max=mot_max,
        prompt_chars_min=pc_min,
        prompt_chars_avg=pc_avg,
        prompt_chars_p50=pc_p50,
        prompt_chars_p90=pc_p90,
        prompt_chars_p99=pc_p99,
        prompt_chars_max=pc_max,
        task_counts=task_counts,
        tag_counts=tag_counts,
    )


def format_report(r: WorkloadReport, *, top_k_tags: int = 10) -> str:
    lines: list[str] = []

    lines.append("IWC Workload Report")
    lines.append("-" * 60)
    lines.append(f"requests: {r.num_requests}")
    lines.append(f"arrival_time_ms: min={r.min_arrival_ms}  max={r.max_arrival_ms}  span={r.arrival_span_ms}")

    lines.append("")
    lines.append("max_output_tokens:")
    lines.append(f"  min={r.mot_min}  avg={None if r.mot_avg is None else round(r.mot_avg, 2)}  p50={r.mot_p50}  p90={r.mot_p90}  p99={r.mot_p99}  max={r.mot_max}")

    lines.append("")
    lines.append("prompt length (chars):")
    lines.append(f"  min={r.prompt_chars_min}  avg={None if r.prompt_chars_avg is None else round(r.prompt_chars_avg, 2)}  p50={r.prompt_chars_p50}  p90={r.prompt_chars_p90}  p99={r.prompt_chars_p99}  max={r.prompt_chars_max}")

    lines.append("")
    lines.append("semantic.task distribution:")
    if r.task_counts:
        total_labeled = sum(r.task_counts.values())
        for task, cnt in r.task_counts.most_common():
            lines.append(f"  - {task}: {cnt} ({round(_pct(cnt, r.num_requests), 2)}%)")
        if total_labeled < r.num_requests:
            unlabeled = r.num_requests - total_labeled
            lines.append(f"  - (missing): {unlabeled} ({round(_pct(unlabeled, r.num_requests), 2)}%)")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"top tags (top {top_k_tags}):")
    if r.tag_counts:
        for tag, cnt in r.tag_counts.most_common(top_k_tags):
            lines.append(f"  - {tag}: {cnt}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)
