from pathlib import Path
from iwc.export import export_aiperf


def test_export_aiperf_single_request_matches_golden(tmp_path: Path) -> None:
    inp = Path("examples/single_request.jsonl")
    golden = Path("tests/golden/aiperf_single_request.jsonl")

    out = tmp_path / "out_single.jsonl"
    export_aiperf(inp, out)

    assert out.read_text(encoding="utf-8") == golden.read_text(encoding="utf-8")


def test_export_aiperf_bursty_10req_matches_golden(tmp_path: Path) -> None:
    inp = Path("examples/bursty_10req.jsonl")
    golden = Path("tests/golden/aiperf_bursty_10req.jsonl")

    out = tmp_path / "out_burst.jsonl"
    export_aiperf(inp, out)

    assert out.read_text(encoding="utf-8") == golden.read_text(encoding="utf-8")
def test_export_aiperf_session_chat_5turns_matches_golden(tmp_path: Path) -> None:
    inp = Path("examples/session_chat_5turns.jsonl")
    golden = Path("tests/golden/aiperf_session_chat_5turns.jsonl")

    out = tmp_path / "out_session.jsonl"
    export_aiperf(inp, out)

    assert out.read_text(encoding="utf-8") == golden.read_text(encoding="utf-8")
