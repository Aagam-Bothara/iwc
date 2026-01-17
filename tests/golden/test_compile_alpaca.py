from __future__ import annotations

import json
from pathlib import Path

import pytest

from iwc.compile_alpaca import AlpacaConfig, compile_alpaca


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_compile_alpaca_golden(tmp_path: Path) -> None:
    """Golden test: Alpaca compiler produces schema-compliant, deterministic output."""
    inp = tmp_path / "alpaca.json"
    outp = tmp_path / "workload.jsonl"
    manifest = tmp_path / "workload.jsonl.manifest.yaml"

    # Minimal Alpaca dataset
    inp.write_text(
        json.dumps(
            [
                {"instruction": "Explain KV cache in one sentence.", "input": "", "output": "dummy"},
                {"instruction": "Summarize TCP vs UDP.", "input": "Keep it short.", "output": "dummy"},
                {"instruction": "What is Python?", "output": "dummy"},  # missing input field
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = AlpacaConfig(
        max_output_tokens=128,
        max_output_policy="fixed",
        temperature=0.0,
        top_p=1.0,
        streaming=False,
        prompt_format="raw",
        arrival="fixed-step",
        arrival_step_ms=100,
        rate_rps=None,
        seed=42,
    )

    compile_alpaca(inp, outp, manifest, cfg)

    rows = _read_jsonl(outp)
    assert len(rows) == 3

    # 1. prompt_format must be "raw"
    for r in rows:
        assert r.get("prompt_format") == "raw", f"Expected prompt_format='raw', got {r.get('prompt_format')}"

    # 2. semantic must include alpaca tag and task=qa
    for r in rows:
        semantic = r.get("semantic", {})
        assert semantic.get("task") == "qa", f"Expected task='qa', got {semantic.get('task')}"
        tags = semantic.get("tags", [])
        assert "alpaca" in tags, f"Expected 'alpaca' in tags, got {tags}"

    # 3. Deterministic with same seed (arrival times must match exactly)
    assert rows[0]["arrival_time_ms"] == 0
    assert rows[1]["arrival_time_ms"] == 100
    assert rows[2]["arrival_time_ms"] == 200

    # 4. Prompt formatting is correct
    assert "KV cache" in rows[0]["prompt"]
    assert "TCP vs UDP" in rows[1]["prompt"]
    assert "Input:" in rows[1]["prompt"]  # input field should be appended
    assert "Keep it short" in rows[1]["prompt"]
    assert "Python" in rows[2]["prompt"]

    # 5. Required fields exist
    for r in rows:
        assert isinstance(r.get("request_id"), str) and r["request_id"]
        assert isinstance(r.get("prompt"), str) and r["prompt"].strip()
        assert isinstance(r.get("arrival_time_ms"), int)
        assert r.get("max_output_tokens") == 128
        assert r.get("temperature") == 0.0
        assert r.get("top_p") == 1.0
        assert r.get("streaming") is False


def test_compile_alpaca_poisson_seeded(tmp_path: Path) -> None:
    """Poisson arrivals must be deterministic with --seed."""
    inp = tmp_path / "alpaca.json"
    outp1 = tmp_path / "w1.jsonl"
    outp2 = tmp_path / "w2.jsonl"
    manifest = tmp_path / "dummy.yaml"

    inp.write_text(
        json.dumps([{"instruction": "Task 1"}, {"instruction": "Task 2"}]) + "\n",
        encoding="utf-8",
    )

    cfg = AlpacaConfig(
        arrival="poisson",
        rate_rps=10.0,
        seed=123,
    )

    compile_alpaca(inp, outp1, manifest, cfg)
    compile_alpaca(inp, outp2, manifest, cfg)

    rows1 = _read_jsonl(outp1)
    rows2 = _read_jsonl(outp2)

    # Same seed => identical arrival times
    assert rows1[0]["arrival_time_ms"] == rows2[0]["arrival_time_ms"]
    assert rows1[1]["arrival_time_ms"] == rows2[1]["arrival_time_ms"]

    # Poisson should NOT be uniform (verify it's not fixed-step)
    delta = rows1[1]["arrival_time_ms"] - rows1[0]["arrival_time_ms"]
    assert delta != 100, "Poisson arrival should not have fixed 100ms steps"
