from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, List


class Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]: ...


@dataclass
class SimpleWhitespaceTokenizer:
    """
    Fallback tokenizer if tiktoken isn't installed.
    Not accurate, but gives consistent relative stats for quick analysis.
    """
    def encode(self, text: str) -> List[int]:
        # Treat whitespace-separated chunks as "tokens"
        # Keep it deterministic.
        parts = text.strip().split()
        return list(range(len(parts)))


def get_tokenizer(prefer: str = "tiktoken", model: str = "gpt-4o-mini") -> Tokenizer:
    """
    prefer:
      - "tiktoken": use tiktoken if installed
      - "simple": always use fallback
    model: only used when prefer="tiktoken"
    """
    if prefer == "simple":
        return SimpleWhitespaceTokenizer()

    if prefer == "tiktoken":
        try:
            import tiktoken  # type: ignore
        except Exception:
            return SimpleWhitespaceTokenizer()

        # Try model-specific encoding; fallback to cl100k_base
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return enc

    raise ValueError(f"Unknown tokenizer preference: {prefer}")
