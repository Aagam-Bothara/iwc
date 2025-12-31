from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Tuple


# -------------------------
# Public configuration
# -------------------------

DIFFICULTY_LEVELS = ("low", "medium", "high")
DEFAULT_TASK = "chat"

# If you ever want to tune without changing logic, these are the knobs.
DIFFICULTY_CHAR_THRESHOLDS = (200, 800)  # <200 low, 200-800 medium, >800 high


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: Pattern[str]


# -------------------------
# Heuristic rules (ordered)
# -------------------------
# IMPORTANT: order matters. First match wins.

_TASK_RULES: List[Rule] = [
    Rule("translation", re.compile(r"\b(translate|translation|translate\s+to)\b", re.I)),
    Rule("summarization", re.compile(r"\b(summarize|summary|tl;dr|tldr)\b", re.I)),
    # Keep "code_generation" fairly broad but not too broad.
    Rule(
        "code_generation",
        re.compile(
            r"\b("
            r"write\s+code|implement|function|class|debug|fix\s+this|refactor|"
            r"sql|query|select\s+.+\s+from"
            r")\b",
            re.I | re.S,
        ),
    ),
    Rule("rewrite", re.compile(r"\b(rewrite|improve|proofread|polish|refine|edit)\b", re.I)),
    Rule("classification", re.compile(r"\b(classify|categorize|label)\b", re.I)),
    Rule("qa", re.compile(r"\b(what\s+is|explain|why|how\s+does|how\s+to)\b", re.I)),
]

# Tags are multi-label; order doesn’t matter much.
_TAG_RULES: List[Rule] = [
    Rule("sql", re.compile(r"\b(sql|postgres|mysql|sqlite|query|database)\b", re.I)),
    Rule("python", re.compile(r"\b(python|pytest|pip|venv|conda)\b", re.I)),
    Rule("java", re.compile(r"\b(java)\b", re.I)),
    Rule("cpp", re.compile(r"\b(c\+\+|cpp)\b", re.I)),
    Rule("javascript", re.compile(r"\b(javascript|node|react|typescript|npm)\b", re.I)),
    Rule("formal", re.compile(r"\b(formal|professional|email|cover\s+letter)\b", re.I)),
    Rule("creative", re.compile(r"\b(story|poem|creative|character|novel)\b", re.I)),
]


# -------------------------
# Core heuristics
# -------------------------

def _difficulty_from_prompt(prompt: str) -> str:
    """
    Schema-compatible difficulty buckets: low | medium | high

    Uses char-length as a stable proxy (no tokenizer dependency).
    """
    n = len(prompt)
    lo, hi = DIFFICULTY_CHAR_THRESHOLDS
    if n < lo:
        return "low"
    if n <= hi:
        return "medium"
    return "high"


def _task_from_prompt(prompt: str) -> str:
    """
    Returns a task label. First matching rule wins. Deterministic.
    """
    for rule in _TASK_RULES:
        if rule.pattern.search(prompt):
            return rule.name
    return DEFAULT_TASK


def _tags_from_prompt(prompt: str) -> List[str]:
    """
    Returns a de-duplicated list of tags (stable order by rule list).
    """
    tags: List[str] = []
    for rule in _TAG_RULES:
        if rule.pattern.search(prompt):
            tags.append(rule.name)
    # stable de-dupe (though rules list should already prevent dupes)
    return list(dict.fromkeys(tags))


def _ensure_dict(x: Any) -> Dict[str, Any]:
    return dict(x) if isinstance(x, dict) else {}


def _merge_tags(existing: Any, new_tags: List[str], *, overwrite: bool) -> List[str]:
    """
    - If overwrite: return new_tags
    - Else: union existing(list[str]) + new_tags (stable)
    """
    if overwrite:
        return new_tags
    if not isinstance(existing, list):
        return new_tags
    keep = [t for t in existing if isinstance(t, str) and t.strip()]
    merged = list(dict.fromkeys(keep + new_tags))
    return merged


def label_record(record: Dict[str, Any], *, overwrite: bool = False) -> Dict[str, Any]:
    """
    Adds semantic.task, semantic.difficulty, semantic.tags using heuristics.

    Contract:
    - Deterministic: same input => same output
    - Safe by default: DOES NOT overwrite existing semantic.task/difficulty unless overwrite=True
    - Never deletes existing semantic fields
    - Adds semantic.source="heuristic" and semantic.version="v1" if missing (traceability)
    """
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        # Leave malformed records unchanged; schema validation should catch issues elsewhere.
        return dict(record)

    out = dict(record)

    semantic = _ensure_dict(out.get("semantic"))

    # Compute new labels
    new_task = _task_from_prompt(prompt)
    new_difficulty = _difficulty_from_prompt(prompt)
    new_tags = _tags_from_prompt(prompt)

    # Apply (fill missing unless overwrite)
    if overwrite or not isinstance(semantic.get("task"), str) or not semantic.get("task").strip():
        semantic["task"] = new_task

    if overwrite or semantic.get("difficulty") not in DIFFICULTY_LEVELS:
        semantic["difficulty"] = new_difficulty

    semantic["tags"] = _merge_tags(semantic.get("tags"), new_tags, overwrite=overwrite)

    # Traceability metadata (won’t break schema if schema allows additional semantic props; if not, remove)
    # If your schema is strict on semantic keys, comment these out.
    #semantic.setdefault("source", "heuristic")
    #semantic.setdefault("version", "v1")

    out["semantic"] = semantic
    return out
