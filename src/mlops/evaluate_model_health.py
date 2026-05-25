import argparse
import json
import math
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


def _positive_score(event: dict[str, Any], positive_label: str) -> float | None:
    scores = event.get("scores")
    if isinstance(scores, dict):
        normalized_scores = {str(k).strip().lower(): v for k, v in scores.items()}
        if positive_label in normalized_scores:
            return _safe_float(normalized_scores.get(positive_label), math.nan)
    if "positive_score" in event:
        return _safe_float(event.get("positive_score"), math.nan)
    if "confidence" in event and event.get("prediction") == positive_label:
        return _safe_float(event.get("confidence"), math.nan)
    return None


def _binary_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    total = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _binary_roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = sum(1 for y in y_true if y == 1)
    negatives = sum(1 for y in y_true if y == 0)
    if positives == 0 or negatives == 0 or len(y_true) != len(y_score):
        return None

    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0])
    rank_sum_pos = 0.0
    idx = 0
    while idx < len(ranked):
        end = idx + 1
        while end < len(ranked) and ranked[end][0] == ranked[idx][0]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        rank_sum_pos += avg_rank * sum(1 for _, label in ranked[idx:end] if label == 1)
        idx = end

    auc = (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _compute_feedback_metrics(
    feedback_events: list[dict[str, Any]],
    *,
    positive_label: str = "dog",
) -> dict[str, Any]:
    positive_label = positive_label.strip().lower()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []

    for e in feedback_events:
        pred = e.get("prediction")
        true_label = e.get("true_label")
        if isinstance(pred, str) and isinstance(true_label, str):
            pred_norm = pred.strip().lower()
            true_norm = true_label.strip().lower()
            y_true.append(1 if true_norm == positive_label else 0)
            y_pred.append(1 if pred_norm == positive_label else 0)

            score = _positive_score(e, positive_label)
            if score is not None and not math.isnan(score):
                y_score.append(float(score))
            continue

        if isinstance(e.get("prediction_correct"), bool):
            y_true.append(1)
            y_pred.append(1 if e["prediction_correct"] else 0)

    if not y_true:
        return {
            "samples": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
        }

    base_metrics = _binary_classification_metrics(y_true, y_pred)
    roc_auc = None
    if len(y_score) == len(y_true) and len(set(y_true)) > 1:
        roc_auc = _binary_roc_auc(y_true, y_score)

    return {
        "samples": len(y_true),
        "accuracy": base_metrics["accuracy"],
        "precision": base_metrics["precision"],
        "recall": base_metrics["recall"],
        "f1": base_metrics["f1"],
        "roc_auc": roc_auc,
    }


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
    min_feedback_roc_auc: float = 0.80,
    positive_label: str = "dog",
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
    feedback_metrics = _compute_feedback_metrics(feedback_events, positive_label=positive_label)
    feedback_count = int(feedback_metrics["samples"])
    feedback_acc = feedback_metrics["accuracy"]
    feedback_auc = feedback_metrics["roc_auc"]

    degraded_conf = avg_conf < min_avg_confidence
    degraded_uncertain = uncertain_rate > max_uncertain_rate
    degraded_feedback = (
        feedback_acc is not None
        and feedback_count >= min_feedback_samples
        and feedback_acc < min_feedback_accuracy
    )
    degraded_feedback_auc = (
        feedback_auc is not None
        and feedback_count >= min_feedback_samples
        and feedback_auc < min_feedback_roc_auc
    )

    if samples < min_samples:
        status = "insufficient_data"
    elif is_stale:
        status = "stale"
    else:
        status = "ok"

    degraded = bool(degraded_conf or degraded_uncertain or degraded_feedback or degraded_feedback_auc)
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
            "feedback_precision": feedback_metrics["precision"],
            "feedback_recall": feedback_metrics["recall"],
            "feedback_f1": feedback_metrics["f1"],
            "feedback_roc_auc": feedback_auc,
        },
        "thresholds": {
            "min_avg_confidence": min_avg_confidence,
            "uncertain_threshold": uncertain_threshold,
            "max_uncertain_rate": max_uncertain_rate,
            "min_feedback_samples": min_feedback_samples,
            "min_feedback_accuracy": min_feedback_accuracy,
            "min_feedback_roc_auc": min_feedback_roc_auc,
            "positive_label": positive_label,
            "stale_hours": stale_hours,
        },
        "signals": {
            "degraded_confidence": degraded_conf,
            "degraded_uncertainty": degraded_uncertain,
            "degraded_feedback_accuracy": degraded_feedback,
            "degraded_feedback_roc_auc": degraded_feedback_auc,
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
    parser.add_argument("--min-feedback-roc-auc", type=float, default=0.80)
    parser.add_argument("--positive-label", type=str, default="dog")
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
        min_feedback_roc_auc=args.min_feedback_roc_auc,
        positive_label=args.positive_label,
    )

    print(f"[OK] Reporte de salud: {args.report_path}")
    print(f"[OK] Estado: {report.get('status')}")
    print(f"[OK] Degradado: {report.get('degraded')}")


if __name__ == "__main__":
    main()
