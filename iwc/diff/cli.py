from __future__ import annotations

import argparse
from pathlib import Path

from iwc.analyze.read_jsonl import iter_requests_jsonl
from iwc.analyze.summary import build_summary
from iwc.diff.core import diff_summaries, render_diff


def add_diff_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("diff", help="Compare two workload JSONL traces (A vs B).")
    p.add_argument("a", help="Path to workload/trace JSONL A (baseline).")
    p.add_argument("b", help="Path to workload/trace JSONL B (candidate).")
    p.add_argument("--tokenizer", choices=["simple", "tiktoken"], default="tiktoken")
    p.add_argument("--tokenizer-model", default="gpt-4o-mini")
    p.set_defaults(func=_run_diff)


def _run_diff(args: argparse.Namespace) -> None:
    a_path = Path(args.a)
    b_path = Path(args.b)

    a_reqs = list(iter_requests_jsonl(str(a_path)))
    b_reqs = list(iter_requests_jsonl(str(b_path)))

    a_sum = build_summary(a_reqs, tokenizer_prefer=args.tokenizer, tokenizer_model=args.tokenizer_model)
    b_sum = build_summary(b_reqs, tokenizer_prefer=args.tokenizer, tokenizer_model=args.tokenizer_model)

    d = diff_summaries(a_sum, b_sum)
    print(render_diff(d, a_label=str(a_path), b_label=str(b_path)))
