# iwc/compile_alpaca.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from iwc.compile import _arrival_times, _canonical_json_line, _write_manifest


@dataclass(frozen=True)
class AlpacaConfig:
    max_output_tokens: int = 128
    max_output_policy: str = "fixed"  # "fixed" | "from_dataset" | "clamp" (future)
    temperature: float = 0.0
    top_p: float = 1.0
    streaming: bool = False

    # schema enum requires raw|chatml|openai_messages
    prompt_format: str = "raw"

    arrival: str = "fixed-step"
    arrival_step_ms: int = 100
    rate_rps: Optional[float] = None
    seed: Optional[int] = None


def _alpaca_to_prompt(rec: dict[str, Any]) -> str:
    """
    Typical Alpaca fields: instruction, input, output.
    We'll compile prompt = instruction + optional input block.
    """
    instr = rec.get("instruction")
    inp = rec.get("input")

    if not isinstance(instr, str) or not instr.strip():
        raise ValueError("alpaca record missing non-empty 'instruction'")

    prompt = instr.strip()
    if isinstance(inp, str) and inp.strip():
        prompt += "\n\nInput:\n" + inp.strip()
    return prompt


def compile_alpaca(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    cfg: AlpacaConfig,
) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("alpaca input must be a JSON list of records")

    prompts: list[str] = []
    skipped = 0
    for obj in data:
        if not isinstance(obj, dict):
            skipped += 1
            continue
        try:
            p = _alpaca_to_prompt(obj)
        except Exception:
            skipped += 1
            continue
        if p.strip():
            prompts.append(p)

    if not prompts:
        raise ValueError("alpaca input produced 0 prompts")

    n = len(prompts)
    arrivals_ms = _arrival_times(n, cfg.arrival, cfg.arrival_step_ms, cfg.rate_rps, cfg.seed)
    arrival_span_ms = int(max(arrivals_ms) - min(arrivals_ms)) if arrivals_ms else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, (prompt, at_ms) in enumerate(zip(prompts, arrivals_ms), start=1):
            req = {
                "request_id": f"req-{i:06d}",
                "prompt": prompt,
                "prompt_format": cfg.prompt_format,  # MUST be schema-valid
                "max_output_tokens": int(cfg.max_output_tokens),
                "arrival_time_ms": int(at_ms),
                "temperature": float(cfg.temperature),
                "top_p": float(cfg.top_p),
                "streaming": bool(cfg.streaming),
                "semantic": {
                    "task": "qa",
                    "tags": ["alpaca", "instruction-following"],
                },
            }
            f.write(_canonical_json_line(req) + "\n")

    _write_manifest(
        compiler="alpaca",
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        summary={"num_requests": n, "arrival_span_ms": arrival_span_ms, "skipped_records": skipped},
        cfg={
            "prompt_format": cfg.prompt_format,
            "max_output_tokens": cfg.max_output_tokens,
            "max_output_policy": cfg.max_output_policy,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "streaming": cfg.streaming,
            "arrival": cfg.arrival,
            "arrival_step_ms": cfg.arrival_step_ms,
            "rate_rps": cfg.rate_rps,
            "seed": cfg.seed,
        },
    )
