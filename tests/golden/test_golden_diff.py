from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    # tests/golden/test_golden_diff.py -> repo root
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _import_diff_fns():
    # Be resilient to refactors of diff module layout.
    try:
        from iwc.diff.diff import diff_summaries, diff_to_dict  # type: ignore
        return diff_summaries, diff_to_dict
    except Exception:
        pass
    try:
        from iwc.diff.core import diff_summaries, diff_to_dict  # type: ignore
        return diff_summaries, diff_to_dict
    except Exception:
        pass
    from iwc.diff import diff_summaries, diff_to_dict  # type: ignore
    return diff_summaries, diff_to_dict


def test_diff_golden_session_vs_cumulative() -> None:
    from iwc.analyze.summary import build_summary
    from iwc.analyze.read_jsonl import iter_requests_jsonl

    diff_summaries, diff_to_dict = _import_diff_fns()

    root = _repo_root()
    examples = root / "examples"
    golden = root / "tests" / "golden" / "diff_session_vs_cumulative.golden.json"

    a = examples / "session_chat_5turns.jsonl"
    b = examples / "session_chat_5turns_cumulative.jsonl"

    a_sum = build_summary(list(iter_requests_jsonl(str(a))), tokenizer_prefer="tiktoken", tokenizer_model="gpt-4o-mini")
    b_sum = build_summary(list(iter_requests_jsonl(str(b))), tokenizer_prefer="tiktoken", tokenizer_model="gpt-4o-mini")

    d = diff_summaries(a_sum, b_sum)
    got = diff_to_dict(d, a_label=str(a), b_label=str(b))
    exp = _load_json(golden)

    # Labels are environment-specific (absolute paths differ on CI vs local), so don't snapshot-test them.
    got.pop("a_label", None)
    got.pop("b_label", None)
    exp.pop("a_label", None)
    exp.pop("b_label", None)

    assert got == exp