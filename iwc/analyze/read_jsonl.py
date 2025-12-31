from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: str
    max_output_tokens: int
    arrival_time_ms: int
    session_id: Optional[str] = None


def iter_requests_jsonl(path: str) -> Iterator[Request]:
    """
    Reads workload JSONL. One JSON object per line.
    Required fields:
      - request_id
      - prompt
      - max_output_tokens
      - arrival_time_ms
    Optional:
      - session_id
    """
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} invalid JSON: {e}") from e

            # ---- Adjust here if your schema differs ----
            missing = [k for k in ("request_id", "prompt", "max_output_tokens", "arrival_time_ms") if k not in obj]
            if missing:
                raise ValueError(f"{path}:{lineno} missing fields: {missing}")

            yield Request(
                request_id=str(obj["request_id"]),
                prompt=str(obj["prompt"]),
                max_output_tokens=int(obj["max_output_tokens"]),
                arrival_time_ms=int(obj["arrival_time_ms"]),
                session_id=(None if obj.get("session_id") in (None, "") else str(obj.get("session_id"))),
            )
