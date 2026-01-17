# iwc/report_results.py
"""Aggregate and report statistics from runner results JSONL."""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ResultsReport:
    """Comprehensive statistics from runner results."""

    # Summary counts
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float

    # Latency statistics (ms)
    latency_p50: float
    latency_p90: float
    latency_p95: float
    latency_p99: float
    latency_mean: float
    latency_min: float
    latency_max: float

    # TTFT statistics (ms) - only if streaming was used
    ttft_p50: Optional[float]
    ttft_p90: Optional[float]
    ttft_p95: Optional[float]
    ttft_p99: Optional[float]
    ttft_mean: Optional[float]

    # Token statistics
    tokens_in_total: int
    tokens_out_total: int
    tokens_total_total: int
    tokens_in_mean: float
    tokens_out_mean: float

    # Throughput
    throughput_rps: float  # requests per second
    throughput_tps_in: float  # input tokens per second
    throughput_tps_out: float  # output tokens per second
    throughput_tps_total: float  # total tokens per second

    # Quality metrics
    truncation_rate: float  # % of requests with finish_reason=length
    truncated_requests: int

    # Error analysis
    error_types: dict[str, int]  # error_type -> count
    retry_rate: float  # % of requests that needed retries
    retried_requests: int

    # Timing
    duration_s: float  # total wall-clock time
    t_start_min: int  # earliest request start
    t_end_max: int  # latest request end

    # Metadata
    schema_version: Optional[str]
    runner_version: Optional[str]
    models: set[str]
    servers: set[str]


def _percentile(data: list[float], p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


def build_results_report(results_path: Path) -> ResultsReport:
    """Build aggregate report from results JSONL."""
    records = []

    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        raise ValueError(f"No valid records found in {results_path}")

    # Collect statistics
    total = len(records)
    successful = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") != "ok"]

    success_count = len(successful)
    fail_count = len(failed)
    success_rate = success_count / total if total > 0 else 0.0

    # Latency
    latencies = [r["latency_ms"] for r in successful if "latency_ms" in r]
    latency_stats = {
        "p50": _percentile(latencies, 50) if latencies else 0.0,
        "p90": _percentile(latencies, 90) if latencies else 0.0,
        "p95": _percentile(latencies, 95) if latencies else 0.0,
        "p99": _percentile(latencies, 99) if latencies else 0.0,
        "mean": statistics.mean(latencies) if latencies else 0.0,
        "min": min(latencies) if latencies else 0.0,
        "max": max(latencies) if latencies else 0.0,
    }

    # TTFT (if available)
    ttfts = [r["ttft_ms"] for r in successful if r.get("ttft_ms") is not None]
    if ttfts:
        ttft_stats = {
            "p50": _percentile(ttfts, 50),
            "p90": _percentile(ttfts, 90),
            "p95": _percentile(ttfts, 95),
            "p99": _percentile(ttfts, 99),
            "mean": statistics.mean(ttfts),
        }
    else:
        ttft_stats = {"p50": None, "p90": None, "p95": None, "p99": None, "mean": None}

    # Tokens
    tokens_in = [r.get("tokens_in", 0) or 0 for r in successful]
    tokens_out = [r.get("tokens_out", 0) or 0 for r in successful]
    tokens_total = [r.get("tokens_total", 0) or 0 for r in successful]

    tokens_in_total = sum(tokens_in)
    tokens_out_total = sum(tokens_out)
    tokens_total_total = sum(tokens_total)

    tokens_in_mean = statistics.mean(tokens_in) if tokens_in else 0.0
    tokens_out_mean = statistics.mean(tokens_out) if tokens_out else 0.0

    # Truncation
    truncated = [r for r in successful if r.get("finish_reason") == "length"]
    truncation_rate = len(truncated) / success_count if success_count > 0 else 0.0

    # Errors
    error_types: dict[str, int] = {}
    for r in failed:
        err_type = r.get("error_type") or "unknown"
        error_types[err_type] = error_types.get(err_type, 0) + 1

    # Retries
    retried = [r for r in records if r.get("retries_used", 0) > 0]
    retry_rate = len(retried) / total if total > 0 else 0.0

    # Timing
    t_starts = [r["t_start_ms"] for r in records if "t_start_ms" in r]
    t_ends = [r["t_end_ms"] for r in records if "t_end_ms" in r]

    t_start_min = min(t_starts) if t_starts else 0
    t_end_max = max(t_ends) if t_ends else 0
    duration_s = (t_end_max - t_start_min) / 1000.0 if t_start_min and t_end_max else 0.0

    # Throughput
    throughput_rps = total / duration_s if duration_s > 0 else 0.0
    throughput_tps_in = tokens_in_total / duration_s if duration_s > 0 else 0.0
    throughput_tps_out = tokens_out_total / duration_s if duration_s > 0 else 0.0
    throughput_tps_total = tokens_total_total / duration_s if duration_s > 0 else 0.0

    # Metadata
    schema_versions = set(r.get("_schema_version") for r in records if "_schema_version" in r)
    runner_versions = set(r.get("_runner_version") for r in records if "_runner_version" in r)
    models = set(r.get("model") for r in records if "model" in r)
    servers = set(r.get("server_base_url") for r in records if "server_base_url" in r)

    return ResultsReport(
        total_requests=total,
        successful_requests=success_count,
        failed_requests=fail_count,
        success_rate=success_rate,
        latency_p50=latency_stats["p50"],
        latency_p90=latency_stats["p90"],
        latency_p95=latency_stats["p95"],
        latency_p99=latency_stats["p99"],
        latency_mean=latency_stats["mean"],
        latency_min=latency_stats["min"],
        latency_max=latency_stats["max"],
        ttft_p50=ttft_stats["p50"],
        ttft_p90=ttft_stats["p90"],
        ttft_p95=ttft_stats["p95"],
        ttft_p99=ttft_stats["p99"],
        ttft_mean=ttft_stats["mean"],
        tokens_in_total=tokens_in_total,
        tokens_out_total=tokens_out_total,
        tokens_total_total=tokens_total_total,
        tokens_in_mean=tokens_in_mean,
        tokens_out_mean=tokens_out_mean,
        throughput_rps=throughput_rps,
        throughput_tps_in=throughput_tps_in,
        throughput_tps_out=throughput_tps_out,
        throughput_tps_total=throughput_tps_total,
        truncation_rate=truncation_rate,
        truncated_requests=len(truncated),
        error_types=error_types,
        retry_rate=retry_rate,
        retried_requests=len(retried),
        duration_s=duration_s,
        t_start_min=t_start_min,
        t_end_max=t_end_max,
        schema_version=list(schema_versions)[0] if schema_versions else None,
        runner_version=list(runner_versions)[0] if runner_versions else None,
        models=models,
        servers=servers,
    )


def format_results_report(report: ResultsReport) -> str:
    """Format report as human-readable text."""
    lines = []

    lines.append("=" * 70)
    lines.append("RUNNER RESULTS SUMMARY")
    lines.append("=" * 70)

    # Overview
    lines.append("\nOVERVIEW")
    lines.append("-" * 70)
    lines.append(f"Total requests:       {report.total_requests}")
    lines.append(f"Successful:           {report.successful_requests} ({report.success_rate * 100:.1f}%)")
    lines.append(f"Failed:               {report.failed_requests}")
    lines.append(f"Retried:              {report.retried_requests} ({report.retry_rate * 100:.1f}%)")
    lines.append(f"Duration:             {report.duration_s:.2f}s")

    # Latency
    lines.append("\nLATENCY (ms)")
    lines.append("-" * 70)
    lines.append(f"Mean:                 {report.latency_mean:.2f}")
    lines.append(f"P50:                  {report.latency_p50:.2f}")
    lines.append(f"P90:                  {report.latency_p90:.2f}")
    lines.append(f"P95:                  {report.latency_p95:.2f}")
    lines.append(f"P99:                  {report.latency_p99:.2f}")
    lines.append(f"Min:                  {report.latency_min:.2f}")
    lines.append(f"Max:                  {report.latency_max:.2f}")

    # TTFT (if available)
    if report.ttft_mean is not None:
        lines.append("\nTTFT (ms, streaming)")
        lines.append("-" * 70)
        lines.append(f"Mean:                 {report.ttft_mean:.2f}")
        lines.append(f"P50:                  {report.ttft_p50:.2f}")
        lines.append(f"P90:                  {report.ttft_p90:.2f}")
        lines.append(f"P95:                  {report.ttft_p95:.2f}")
        lines.append(f"P99:                  {report.ttft_p99:.2f}")

    # Throughput
    lines.append("\nTHROUGHPUT")
    lines.append("-" * 70)
    lines.append(f"Requests/sec:         {report.throughput_rps:.2f}")
    lines.append(f"Input tokens/sec:     {report.throughput_tps_in:.2f}")
    lines.append(f"Output tokens/sec:    {report.throughput_tps_out:.2f}")
    lines.append(f"Total tokens/sec:     {report.throughput_tps_total:.2f}")

    # Token statistics
    lines.append("\nTOKEN STATISTICS")
    lines.append("-" * 70)
    lines.append(f"Total input tokens:   {report.tokens_in_total}")
    lines.append(f"Total output tokens:  {report.tokens_out_total}")
    lines.append(f"Total tokens:         {report.tokens_total_total}")
    lines.append(f"Mean input tokens:    {report.tokens_in_mean:.1f}")
    lines.append(f"Mean output tokens:   {report.tokens_out_mean:.1f}")

    # Quality
    lines.append("\nQUALITY METRICS")
    lines.append("-" * 70)
    lines.append(f"Truncated (hit max):  {report.truncated_requests} ({report.truncation_rate * 100:.1f}%)")

    # Errors
    if report.error_types:
        lines.append("\nERROR BREAKDOWN")
        lines.append("-" * 70)
        for err_type, count in sorted(report.error_types.items(), key=lambda x: -x[1]):
            pct = (count / report.failed_requests * 100) if report.failed_requests > 0 else 0
            lines.append(f"{err_type:20s}: {count:4d} ({pct:5.1f}%)")

    # Metadata
    lines.append("\nMETADATA")
    lines.append("-" * 70)
    if report.runner_version:
        lines.append(f"Runner version:       {report.runner_version}")
    if report.schema_version:
        lines.append(f"Schema version:       {report.schema_version}")
    if report.models:
        lines.append(f"Models:               {', '.join(sorted(report.models))}")
    if report.servers:
        lines.append(f"Servers:              {', '.join(sorted(report.servers))}")

    lines.append("=" * 70)

    return "\n".join(lines)


def report_to_dict(report: ResultsReport) -> dict[str, Any]:
    """Convert report to JSON-serializable dict."""
    return {
        "overview": {
            "total_requests": report.total_requests,
            "successful_requests": report.successful_requests,
            "failed_requests": report.failed_requests,
            "success_rate": report.success_rate,
            "retried_requests": report.retried_requests,
            "retry_rate": report.retry_rate,
            "duration_s": report.duration_s,
        },
        "latency_ms": {
            "mean": report.latency_mean,
            "p50": report.latency_p50,
            "p90": report.latency_p90,
            "p95": report.latency_p95,
            "p99": report.latency_p99,
            "min": report.latency_min,
            "max": report.latency_max,
        },
        "ttft_ms": {
            "mean": report.ttft_mean,
            "p50": report.ttft_p50,
            "p90": report.ttft_p90,
            "p95": report.ttft_p95,
            "p99": report.ttft_p99,
        }
        if report.ttft_mean is not None
        else None,
        "throughput": {
            "requests_per_sec": report.throughput_rps,
            "input_tokens_per_sec": report.throughput_tps_in,
            "output_tokens_per_sec": report.throughput_tps_out,
            "total_tokens_per_sec": report.throughput_tps_total,
        },
        "tokens": {
            "total_input": report.tokens_in_total,
            "total_output": report.tokens_out_total,
            "total": report.tokens_total_total,
            "mean_input": report.tokens_in_mean,
            "mean_output": report.tokens_out_mean,
        },
        "quality": {
            "truncated_requests": report.truncated_requests,
            "truncation_rate": report.truncation_rate,
        },
        "errors": report.error_types,
        "metadata": {
            "runner_version": report.runner_version,
            "schema_version": report.schema_version,
            "models": list(report.models),
            "servers": list(report.servers),
        },
    }
