from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stddev(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def coeff_var(xs: List[float]) -> float:
    m = mean(xs)
    if not xs or m == 0 or math.isnan(m):
        return float("nan")
    return stddev(xs) / m


def percentile(xs: List[float], p: float) -> float:
    """
    p in [0, 100]
    """
    if not xs:
        return float("nan")
    if p <= 0:
        return float(min(xs))
    if p >= 100:
        return float(max(xs))
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    d0 = ys[f] * (c - k)
    d1 = ys[c] * (k - f)
    return float(d0 + d1)


def histogram(xs: List[float], bins: List[Tuple[float, float]]) -> Dict[str, int]:
    """
    bins: list of (lo_inclusive, hi_inclusive) ranges.
    Returns dict labels -> counts
    """
    out: Dict[str, int] = {}
    for lo, hi in bins:
        label = f"{int(lo)}-{int(hi)}"
        out[label] = 0

    for x in xs:
        for lo, hi in bins:
            if lo <= x <= hi:
                out[f"{int(lo)}-{int(hi)}"] += 1
                break
    return out


@dataclass(frozen=True)
class DistSummary:
    n: int
    mean: float
    p50: float
    p90: float
    p99: float
    min: float
    max: float

    @staticmethod
    def from_list(xs: List[float]) -> "DistSummary":
        if not xs:
            return DistSummary(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        return DistSummary(
            n=len(xs),
            mean=mean(xs),
            p50=percentile(xs, 50),
            p90=percentile(xs, 90),
            p99=percentile(xs, 99),
            min=float(min(xs)),
            max=float(max(xs)),
        )
