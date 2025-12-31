from __future__ import annotations

import json
from pathlib import Path

import pytest

from iwc.compile import SimpleJsonConfig, compile_jsonl_prompts


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_compile_jsonl_prompts_emits_prompt_format_and_preserves_semantic(tmp_path: Path) -> None:
    inp = tmp_path / "prompts.jsonl"
    outp = tmp_path / "canon.jsonl"
    manifest = tmp_path / "canon.jsonl.manifest.yaml"

    inp.write_text(
        "\n".join(
            [
                json.dumps("Translate hello to French"),
                json.dumps({"prompt": "Summarize: LLMs are useful.", "semantic": {"task": "summarization"}}),
                json.dumps({"prompt": "Write a Python function to add two numbers"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = SimpleJsonConfig(
        max_output_tokens=128,
        temperature=0.0,
        top_p=1.0,
        streaming=False,
        arrival="fixed-step",
        arrival_step_ms=100,
        rate_rps=None,
        seed=None,
    )

    compile_jsonl_prompts(inp, outp, manifest, cfg, prompt_format="text")

    rows = _read_jsonl(outp)
    assert len(rows) == 3

    # prompt_format must exist and be correct
    for r in rows:
        assert r.get("prompt_format") == "text"

    # semantic passthrough should be preserved for the one row that had it
    assert rows[1].get("semantic", {}).get("task") == "summarization"

    # basic required fields sanity
    for r in rows:
        assert isinstance(r.get("request_id"), str) and r["request_id"]
        assert isinstance(r.get("prompt"), str) and r["prompt"].strip()
        assert isinstance(r.get("arrival_time_ms"), int)
        assert isinstance(r.get("max_output_tokens"), int)
