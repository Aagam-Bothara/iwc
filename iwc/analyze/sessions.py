from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .read_jsonl import Request
from .stats import DistSummary
from .tokenizer import get_tokenizer


def _common_prefix_len_tokens(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass(frozen=True)
class SessionStats:
    sessions_detected: int
    avg_turns_per_session: float
    turns_per_session: DistSummary

    # Token-based metrics (defensible)
    prompt_reuse_ratio_tokens: float  # mean LCP(token_ids)/len(curr_tokens)
    prompt_tokens_by_turn: DistSummary
    prompt_token_growth: DistSummary


def analyze_sessions(
    reqs: List[Request],
    tokenizer_prefer: str = "tiktoken",
    tokenizer_model: str = "gpt-4o-mini",
) -> SessionStats:
    tok = get_tokenizer(prefer=tokenizer_prefer, model=tokenizer_model)

    by: Dict[str, List[Request]] = {}
    for r in reqs:
        if not r.session_id:
            continue
        by.setdefault(r.session_id, []).append(r)

    if not by:
        return SessionStats(
            sessions_detected=0,
            avg_turns_per_session=float("nan"),
            turns_per_session=DistSummary.from_list([]),
            prompt_reuse_ratio_tokens=float("nan"),
            prompt_tokens_by_turn=DistSummary.from_list([]),
            prompt_token_growth=DistSummary.from_list([]),
        )

    turns: List[float] = []
    reuse_samples: List[float] = []
    prompt_lens: List[float] = []
    growth: List[float] = []

    for sid, items in by.items():
        items_sorted = sorted(items, key=lambda x: x.arrival_time_ms)
        turns.append(float(len(items_sorted)))

        prev_tokens: Optional[List[int]] = None
        prev_len: Optional[int] = None

        for r in items_sorted:
            cur_text = r.prompt or ""
            cur_tokens = tok.encode(cur_text)
            cur_len = len(cur_tokens)

            prompt_lens.append(float(cur_len))

            if prev_tokens is not None and cur_len > 0:
                cpl = _common_prefix_len_tokens(prev_tokens, cur_tokens)
                reuse_samples.append(cpl / float(cur_len))

            if prev_len is not None:
                growth.append(float(cur_len - prev_len))

            prev_tokens = cur_tokens
            prev_len = cur_len

    turns_sum = DistSummary.from_list(turns)
    reuse_mean = sum(reuse_samples) / len(reuse_samples) if reuse_samples else float("nan")

    return SessionStats(
        sessions_detected=len(by),
        avg_turns_per_session=turns_sum.mean,
        turns_per_session=turns_sum,
        prompt_reuse_ratio_tokens=reuse_mean,
        prompt_tokens_by_turn=DistSummary.from_list(prompt_lens),
        prompt_token_growth=DistSummary.from_list(growth),
    )
