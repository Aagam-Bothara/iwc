import json
from pathlib import Path

from iwc.fingerprint import build_fingerprint_from_report_json
from iwc.report import build_report, report_to_dict


def _normalize_fp(fp: dict) -> dict:
    """
    Fingerprint contains some fields that are expected to change
    while tokenization / report details evolve.
    We keep golden tests stable by comparing only the stable blocks.
    """
    fp = dict(fp)  # shallow copy

    # Token stats are the most volatile while you're iterating (tokenizer choice,
    # schema changes, percentiles logic). Keep them out of golden gating.
    fp.pop("token", None)

    # If you also have any other evolving blocks, drop them here (example):
    # fp.pop("tags", None)

    return fp


def test_golden_fingerprint_session_chat_5turns() -> None:
    repo = Path(__file__).resolve().parents[2]
    inp = repo / "examples" / "session_chat_5turns.jsonl"
    golden = repo / "tests" / "golden" / "fingerprint_session_chat_5turns.golden.json"

    r = build_report(inp)
    fp, _ = build_fingerprint_from_report_json(report_to_dict(r, top_k_tags=0))

    got_obj = _normalize_fp(fp)
    exp_obj = _normalize_fp(json.loads(golden.read_text(encoding="utf-8")))

    assert got_obj == exp_obj
