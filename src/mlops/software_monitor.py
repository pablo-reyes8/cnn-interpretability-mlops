import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_jsonl(path: str, window_size: int) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines()[-window_size:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out >= 0 else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[idx])


def _latency_summary(events: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    vals = [_safe_float(e.get(field)) for e in events]
    clean = [v for v in vals if v is not None]
    if not clean:
        return {"avg": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "avg": float(sum(clean) / len(clean)),
        "p50": _percentile(clean, 50),
        "p95": _percentile(clean, 95),
        "p99": _percentile(clean, 99),
        "max": float(max(clean)),
    }


def summarize_software_metrics(
    *,
    inference_log_path: str,
    report_path: str,
    window_size: int = 1000,
    max_p95_latency_ms: float = 2000.0,
    max_p95_inference_ms: float = 800.0,
) -> dict[str, Any]:
    events = _load_jsonl(inference_log_path, window_size=window_size)
    now = datetime.now(timezone.utc)
    status_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}

    for event in events:
        status = str(event.get("status") or "unknown")
        endpoint = str(event.get("endpoint") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1

    total = len(events)
    errors = sum(v for k, v in status_counts.items() if k != "ok")
    error_rate = errors / total if total else 0.0

    total_latency = _latency_summary(events, "total_latency_ms")
    preprocess_latency = _latency_summary(events, "preprocess_ms")
    inference_latency = _latency_summary(events, "inference_ms")
    degraded_latency = (
        total_latency["p95"] is not None and total_latency["p95"] > max_p95_latency_ms
    )
    degraded_inference = (
        inference_latency["p95"] is not None and inference_latency["p95"] > max_p95_inference_ms
    )

    report = {
        "timestamp_utc": now.isoformat(),
        "status": "insufficient_data" if total == 0 else "ok",
        "degraded": bool(degraded_latency or degraded_inference),
        "samples": total,
        "window_size": window_size,
        "counts": {
            "by_status": status_counts,
            "by_endpoint": endpoint_counts,
            "errors": errors,
            "error_rate": error_rate,
        },
        "latency_ms": {
            "total": total_latency,
            "preprocess": preprocess_latency,
            "inference": inference_latency,
        },
        "thresholds": {
            "max_p95_latency_ms": max_p95_latency_ms,
            "max_p95_inference_ms": max_p95_inference_ms,
        },
        "signals": {
            "degraded_total_latency": bool(degraded_latency),
            "degraded_inference_latency": bool(degraded_inference),
        },
    }

    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitoreo de software: latencia y errores de inferencia.")
    parser.add_argument("--inference-log-path", type=str, default="monitoring/inference_events.jsonl")
    parser.add_argument("--report-path", type=str, default="monitoring/software_metrics_report.json")
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--max-p95-latency-ms", type=float, default=2000.0)
    parser.add_argument("--max-p95-inference-ms", type=float, default=800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_software_metrics(
        inference_log_path=args.inference_log_path,
        report_path=args.report_path,
        window_size=args.window_size,
        max_p95_latency_ms=args.max_p95_latency_ms,
        max_p95_inference_ms=args.max_p95_inference_ms,
    )
    print(f"[OK] Software metrics report: {args.report_path}")
    print(f"[OK] Status: {report.get('status')}")
    print(f"[OK] Degraded: {report.get('degraded')}")


if __name__ == "__main__":
    main()
