from __future__ import annotations

import math
import random
from typing import Optional


def arrival_fixed_step(n: int, step_ms: int) -> list[int]:
    if n < 0:
        raise ValueError("n must be >= 0")
    if step_ms < 0:
        raise ValueError("step_ms must be >= 0")
    return [i * step_ms for i in range(n)]


def arrival_poisson(n: int, rate_rps: float, seed: Optional[int] = None) -> list[int]:
    """
    Poisson arrivals: inter-arrival times ~ Exp(rate_rps).
    Returns integer arrival_time_ms offsets, starting at 0ms.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if rate_rps <= 0:
        raise ValueError("rate_rps must be > 0")

    rng = random.Random(seed)
    arrivals: list[int] = []
    t_s = 0.0

    for i in range(n):
        if i == 0:
            arrivals.append(0)
            continue
        # exponential inter-arrival with rate lambda
        u = rng.random()
        dt = -math.log(1.0 - u) / rate_rps
        t_s += dt
        arrivals.append(int(round(t_s * 1000.0)))

    return arrivals
