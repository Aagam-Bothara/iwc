from __future__ import annotations

import argparse

from .read_jsonl import iter_requests_jsonl
from .summary import build_summary, render_summary


def add_analyze_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("analyze", help="Analyze a workload JSONL trace")
    p.add_argument("trace", help="Path to workload JSONL")
    p.add_argument("--tokenizer", default="tiktoken", choices=["tiktoken", "simple"],
                   help="Tokenizer to estimate prompt tokens")
    p.add_argument("--tokenizer-model", default="gpt-4o-mini",
                   help="Model name used for tiktoken encoding_for_model (if available)")
    p.set_defaults(func=_run_analyze)


def _run_analyze(args: argparse.Namespace) -> int:
    reqs = list(iter_requests_jsonl(args.trace))
    summ = build_summary(reqs, tokenizer_prefer=args.tokenizer, tokenizer_model=args.tokenizer_model)
    print(render_summary(summ))
    return 0
