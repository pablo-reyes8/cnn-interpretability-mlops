import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _load_jsonl(path: str, window_size: int) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-window_size:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_ts(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _compute_feedback_accuracy(feedback_events: list[dict[str, Any]]) -> tuple[int, float | None]:
    vals: list[bool] = []
    for e in feedback_events:
        if isinstance(e.get("prediction_correct"), bool):
            vals.append(bool(e["prediction_correct"]))
            continue

        pred = e.get("prediction")
        true_label = e.get("true_label")
        if isinstance(pred, str) and isinstance(true_label, str):
            vals.append(pred.strip().lower() == true_label.strip().lower())

    if not vals:
        return 0, None
    acc = sum(1 for x in vals if x) / len(vals)
    return len(vals), float(acc)


def evaluate_model_health(
    *,
    inference_log_path: str,
    feedback_log_path: str,
    report_path: str,
    window_size: int = 300,
    min_samples: int = 50,
    stale_hours: float = 48.0,
    min_avg_confidence: float = 0.60,
    uncertain_threshold: float = 0.55,
    max_uncertain_rate: float = 0.40,
    min_feedback_samples: int = 20,
    min_feedback_accuracy: float = 0.80,
) -> dict[str, Any]:
    events = _load_jsonl(inference_log_path, window_size=window_size)
    samples = len(events)

    now = datetime.now(timezone.utc)
    if samples == 0:
        report = {
            "timestamp_utc": now.isoformat(),
            "status": "insufficient_data",
            "degraded": False,
            "samples": 0,
            "window_size": window_size,
        }
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    confidences = [_safe_float(e.get("confidence"), 0.0) for e in events]
    avg_conf = sum(confidences) / max(1, len(confidences))
    uncertain_rate = sum(1 for c in confidences if c < uncertain_threshold) / max(1, len(confidences))

    latest_ts = _parse_ts(events[-1].get("timestamp_utc")) if events else None
    is_stale = False
    if latest_ts is not None and latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=timezone.utc)
    if latest_ts is not None:
        is_stale = (now - latest_ts) > timedelta(hours=stale_hours)

    feedback_events = _load_jsonl(feedback_log_path, window_size=window_size)
    feedback_count, feedback_acc = _compute_feedback_accuracy(feedback_events)

    degraded_conf = avg_conf < min_avg_confidence
    degraded_uncertain = uncertain_rate > max_uncertain_rate
    degraded_feedback = (
        feedback_acc is not None
        and feedback_count >= min_feedback_samples
        and feedback_acc < min_feedback_accuracy
    )

    if samples < min_samples:
        status = "insufficient_data"
    elif is_stale:
        status = "stale"
    else:
        status = "ok"

    degraded = bool(degraded_conf or degraded_uncertain or degraded_feedback)
    report = {
        "timestamp_utc": now.isoformat(),
        "status": status,
        "degraded": degraded,
        "samples": samples,
        "window_size": window_size,
        "latest_inference_ts": latest_ts.isoformat() if latest_ts else None,
        "metrics": {
            "avg_confidence": avg_conf,
            "uncertain_rate": uncertain_rate,
            "feedback_samples": feedback_count,
            "feedback_accuracy": feedback_acc,
        },
        "thresholds": {
            "min_avg_confidence": min_avg_confidence,
            "uncertain_threshold": uncertain_threshold,
            "max_uncertain_rate": max_uncertain_rate,
            "min_feedback_samples": min_feedback_samples,
            "min_feedback_accuracy": min_feedback_accuracy,
            "stale_hours": stale_hours,
        },
        "signals": {
            "degraded_confidence": degraded_conf,
            "degraded_uncertainty": degraded_uncertain,
            "degraded_feedback_accuracy": degraded_feedback,
            "stale_stream": is_stale,
        },
    }

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Evalua salud del modelo en produccion.")
    parser.add_argument("--inference-log-path", type=str, default="monitoring/inference_events.jsonl")
    parser.add_argument("--feedback-log-path", type=str, default="monitoring/feedback_events.jsonl")
    parser.add_argument("--report-path", type=str, default="monitoring/model_health_report.json")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--stale-hours", type=float, default=48.0)
    parser.add_argument("--min-avg-confidence", type=float, default=0.60)
    parser.add_argument("--uncertain-threshold", type=float, default=0.55)
    parser.add_argument("--max-uncertain-rate", type=float, default=0.40)
    parser.add_argument("--min-feedback-samples", type=int, default=20)
    parser.add_argument("--min-feedback-accuracy", type=float, default=0.80)
    return parser.parse_args()


def main():
    args = parse_args()
    report = evaluate_model_health(
        inference_log_path=args.inference_log_path,
        feedback_log_path=args.feedback_log_path,
        report_path=args.report_path,
        window_size=args.window_size,
        min_samples=args.min_samples,
        stale_hours=args.stale_hours,
        min_avg_confidence=args.min_avg_confidence,
        uncertain_threshold=args.uncertain_threshold,
        max_uncertain_rate=args.max_uncertain_rate,
        min_feedback_samples=args.min_feedback_samples,
        min_feedback_accuracy=args.min_feedback_accuracy,
    )

    print(f"[OK] Reporte de salud: {args.report_path}")
    print(f"[OK] Estado: {report.get('status')}")
    print(f"[OK] Degradado: {report.get('degraded')}")


if __name__ == "__main__":
    main()

