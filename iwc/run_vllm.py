# iwc/run_vllm.py
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp

from iwc.analyze.read_jsonl import iter_requests_jsonl, Request
from importlib.metadata import version as _pkg_version

# Result schema version - increment when output format changes
RESULT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class VllmRunConfig:
    base_url: str
    model: str
    out_path: Path

    concurrency: int = 4
    timeout_s: float = 60.0
    max_retries: int = 2

    # If you later support chat/messages, you can switch endpoint.
    endpoint: str = "/v1/completions"


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _post_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
    max_retries: int,
) -> tuple[bool, int, dict[str, Any], Optional[str], int]:
    """
    Returns: (success, http_status, response_json, error_msg, attempts_used)
    """
    last_err: Optional[str] = None
    last_status = 0
    for attempt in range(max_retries + 1):
        try:
            async with session.post(url, json=payload, timeout=timeout_s) as resp:
                last_status = resp.status
                text = await resp.text()
                if resp.status >= 200 and resp.status < 300:
                    try:
                        return True, resp.status, json.loads(text), None, attempt + 1
                    except Exception:
                        return False, resp.status, {}, f"invalid_json_response", attempt + 1
                last_err = f"http_{resp.status}"
        except asyncio.TimeoutError:
            last_err = "timeout"
            last_status = 0
        except Exception as e:
            last_err = f"request_error: {type(e).__name__}"
            last_status = 0
        if attempt < max_retries:
            await asyncio.sleep(0.2 * (attempt + 1))
    return False, last_status, {}, last_err, max_retries + 1


def _extract_usage(resp_json: dict[str, Any]) -> dict[str, Any]:
    """Extract token usage from OpenAI-compatible response."""
    usage = resp_json.get("usage", {})
    if not isinstance(usage, dict):
        return {"tokens_in": None, "tokens_out": None, "tokens_total": None}

    return {
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
        "tokens_total": usage.get("total_tokens"),
    }


def _extract_finish_reason(resp_json: dict[str, Any]) -> Optional[str]:
    """Extract finish_reason from first choice."""
    choices = resp_json.get("choices", [])
    if choices and isinstance(choices, list) and len(choices) > 0:
        first = choices[0]
        if isinstance(first, dict):
            return first.get("finish_reason")
    return None


def _extract_text_preview(resp_json: dict[str, Any], max_len: int = 80) -> Optional[str]:
    """Extract first 80 chars of response text for sanity checking."""
    choices = resp_json.get("choices", [])
    if choices and isinstance(choices, list) and len(choices) > 0:
        first = choices[0]
        if isinstance(first, dict):
            text = first.get("text", "")
            if isinstance(text, str) and text:
                return text[:max_len]
    return None


async def _run_one(
    sem: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    url: str,
    base_url: str,
    r: Request,
    model: str,
    timeout_s: float,
    max_retries: int,
    t0_ms: int,
) -> dict[str, Any]:
    async with sem:
        # honor arrival times (relative)
        target_ms = t0_ms + int(r.arrival_time_ms)
        delay_ms = target_ms - _now_ms()
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

        t_start = _now_ms()

        payload = {
            "model": model,
            "prompt": r.prompt,
            "max_tokens": int(r.max_output_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }

        ok, http_status, resp_json, err, attempts = await _post_with_retries(
            session, url, payload, timeout_s=timeout_s, max_retries=max_retries
        )

        t_end = _now_ms()

        # Extract usage and metadata (gracefully handle missing fields)
        if ok:
            usage = _extract_usage(resp_json)
            finish_reason = _extract_finish_reason(resp_json)
            text_preview = _extract_text_preview(resp_json)

            # Warn if usage is missing (server doesn't support it)
            if usage["tokens_in"] is None or usage["tokens_out"] is None:
                # Server didn't return usage - fall back to approximation
                usage["_usage_source"] = "missing"
        else:
            usage = {"tokens_in": None, "tokens_out": None, "tokens_total": None, "_usage_source": "error"}
            finish_reason = None
            text_preview = None

        # Production-grade output
        out: dict[str, Any] = {
            # Schema metadata
            "_schema_version": RESULT_SCHEMA_VERSION,
            "_runner_version": _pkg_version("iwc"),
            # Request identity
            "request_id": r.request_id,
            "arrival_time_ms": int(r.arrival_time_ms),
            "session_id": getattr(r, "session_id", None),
            "attempts": attempts,
            "retries_used": attempts - 1,
            # Timing
            "t_start_ms": t_start,
            "t_end_ms": t_end,
            "latency_ms": int(t_end - t_start),
            "ttft_ms": None,  # Set by streaming runner
            # Token accounting
            "tokens_in": usage["tokens_in"],
            "tokens_out": usage["tokens_out"],
            "tokens_total": usage["tokens_total"],
            "finish_reason": finish_reason,
            "usage_source": usage.get("_usage_source", "server"),  # "server" | "missing" | "error"
            # Request parameters (for reproducibility)
            "request_params": {
                "max_output_tokens": int(r.max_output_tokens),
                "temperature": payload.get("temperature"),
                "top_p": payload.get("top_p"),
                "prompt_format": getattr(r, "prompt_format", "raw"),
                "streaming": payload.get("stream", False),
            },
            # Response metadata
            "status": "ok" if ok else "error",
            "http_status": http_status,
            "error_type": err if not ok else None,
            "error_msg": err if not ok else None,
            "model": model,
            "server_base_url": base_url,
            "text_preview": text_preview,
        }
        return out


async def run_vllm_async(workload_path: Path, cfg: VllmRunConfig) -> int:
    reqs = list(iter_requests_jsonl(str(workload_path)))
    if not reqs:
        raise ValueError("workload has 0 requests")

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    url = cfg.base_url.rstrip("/") + cfg.endpoint

    sem = asyncio.Semaphore(max(1, int(cfg.concurrency)))
    t0_ms = _now_ms()

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _run_one(
                sem, session, url, cfg.base_url, r, cfg.model, cfg.timeout_s, cfg.max_retries, t0_ms
            )
            for r in reqs
        ]
        results = await asyncio.gather(*tasks)

    failures = 0
    with cfg.out_path.open("w", encoding="utf-8") as f:
        for row in results:
            if row.get("status") != "ok":
                failures += 1
            f.write(json.dumps(row, sort_keys=True) + "\n")

    if failures == 0:
        return 0
    if failures < len(results):
        return 2
    return 1


def run_vllm(workload_path: Path, cfg: VllmRunConfig) -> int:
    return asyncio.run(run_vllm_async(workload_path, cfg))
