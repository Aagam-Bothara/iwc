from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .stats import DistSummary, coeff_var


@dataclass(frozen=True)
class ArrivalStats:
    n: int
    duration_s: float
    mean_rps: float
    peak_rps_1s: float
    interarrival_ms: DistSummary
    burstiness_cv: float
    pattern: str


def analyze_arrivals(arrival_ms: List[int]) -> ArrivalStats:
    if not arrival_ms:
        return ArrivalStats(
            n=0,
            duration_s=0.0,
            mean_rps=float("nan"),
            peak_rps_1s=float("nan"),
            interarrival_ms=DistSummary.from_list([]),
            burstiness_cv=float("nan"),
            pattern="unknown",
        )

    ts = sorted(arrival_ms)
    n = len(ts)
    duration_ms = max(0, ts[-1] - ts[0])
    duration_s = max(1e-9, duration_ms / 1000.0)

    mean_rps = n / duration_s

    # Inter-arrival deltas
    deltas = [ts[i] - ts[i - 1] for i in range(1, n)]
    deltas_f = [float(x) for x in deltas] if deltas else []
    inter = DistSummary.from_list(deltas_f)
    cv = coeff_var(deltas_f) if deltas_f else float("nan")

    # Peak RPS in 1-second buckets
    # bucket = floor((t - t0)/1000)
    t0 = ts[0]
    counts = {}
    for t in ts:
        b = (t - t0) // 1000
        counts[b] = counts.get(b, 0) + 1
    peak = float(max(counts.values())) if counts else float("nan")

    # Simple interpretation
    if cv != cv:  # NaN
        pattern = "unknown"
    elif cv < 0.8:
        pattern = "smooth / batch-like"
    elif cv < 1.5:
        pattern = "mixed / poisson-ish"
    else:
        pattern = "bursty / interactive-like"

    return ArrivalStats(
        n=n,
        duration_s=duration_s,
        mean_rps=mean_rps,
        peak_rps_1s=peak,
        interarrival_ms=inter,
        burstiness_cv=cv,
        pattern=pattern,
    )
