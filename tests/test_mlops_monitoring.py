import json
from pathlib import Path

from src.mlops.drift_core import analyze_drift
from src.mlops.evaluate_model_health import evaluate_model_health
from src.mlops.software_monitor import summarize_software_metrics


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_software_monitor_reports_latency_percentiles_and_error_rate(tmp_path):
    log_path = tmp_path / "inference.jsonl"
    report_path = tmp_path / "software.json"
    _write_jsonl(
        log_path,
        [
            {"endpoint": "/predict", "status": "ok", "total_latency_ms": 100, "preprocess_ms": 20, "inference_ms": 50},
            {"endpoint": "/predict", "status": "ok", "total_latency_ms": 200, "preprocess_ms": 30, "inference_ms": 80},
            {"endpoint": "/predict/advanced", "status": "error", "total_latency_ms": 300, "preprocess_ms": 40, "inference_ms": 90},
        ],
    )

    report = summarize_software_metrics(
        inference_log_path=str(log_path),
        report_path=str(report_path),
        window_size=10,
        max_p95_latency_ms=1000,
        max_p95_inference_ms=1000,
    )

    assert report["status"] == "ok"
    assert report["samples"] == 3
    assert report["counts"]["error_rate"] == 1 / 3
    assert report["counts"]["by_endpoint"]["/predict"] == 2
    assert report["latency_ms"]["total"]["p95"] >= 200
    assert report_path.exists()


def test_model_health_computes_supervised_binary_metrics_and_auc(tmp_path):
    inference_path = tmp_path / "inference.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    report_path = tmp_path / "health.json"
    _write_jsonl(
        inference_path,
        [
            {"timestamp_utc": "2026-05-25T10:00:00+00:00", "confidence": 0.90},
            {"timestamp_utc": "2026-05-25T10:01:00+00:00", "confidence": 0.80},
            {"timestamp_utc": "2026-05-25T10:02:00+00:00", "confidence": 0.70},
            {"timestamp_utc": "2026-05-25T10:03:00+00:00", "confidence": 0.60},
        ],
    )
    _write_jsonl(
        feedback_path,
        [
            {"prediction": "dog", "true_label": "dog", "scores": {"cat": 0.10, "dog": 0.90}},
            {"prediction": "cat", "true_label": "cat", "scores": {"cat": 0.80, "dog": 0.20}},
            {"prediction": "dog", "true_label": "dog", "scores": {"cat": 0.30, "dog": 0.70}},
            {"prediction": "cat", "true_label": "cat", "scores": {"cat": 0.60, "dog": 0.40}},
        ],
    )

    report = evaluate_model_health(
        inference_log_path=str(inference_path),
        feedback_log_path=str(feedback_path),
        report_path=str(report_path),
        min_samples=1,
        min_feedback_samples=1,
    )

    assert report["status"] == "ok"
    assert report["metrics"]["feedback_accuracy"] == 1.0
    assert report["metrics"]["feedback_precision"] == 1.0
    assert report["metrics"]["feedback_recall"] == 1.0
    assert report["metrics"]["feedback_f1"] == 1.0
    assert report["metrics"]["feedback_roc_auc"] == 1.0
    assert report["degraded"] is False


def test_drift_report_separates_data_model_problem_and_concept_drift(tmp_path):
    reference_path = tmp_path / "stats.json"
    inference_path = tmp_path / "inference.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    reference_path.write_text(
        json.dumps({"loc": [0.5, 0.5, 0.5], "scale": [0.1, 0.1, 0.1]}),
        encoding="utf-8",
    )
    _write_jsonl(
        inference_path,
        [
            {
                "prediction": "dog",
                "confidence": 0.40,
                "raw_channel_mean": [0.9, 0.9, 0.9],
                "raw_channel_std": [0.2, 0.2, 0.2],
            },
            {
                "prediction": "dog",
                "confidence": 0.45,
                "raw_channel_mean": [0.95, 0.95, 0.95],
                "raw_channel_std": [0.2, 0.2, 0.2],
            },
        ],
    )
    _write_jsonl(
        feedback_path,
        [
            {"prediction": "dog", "true_label": "cat"},
            {"prediction": "dog", "true_label": "cat"},
        ],
    )

    report = analyze_drift(
        reference_stats_path=str(reference_path),
        inference_log_path=str(inference_path),
        feedback_log_path=str(feedback_path),
        min_samples=1,
        min_feedback_samples=1,
        mean_shift_threshold=0.1,
        scale_shift_threshold=0.1,
        min_avg_confidence=0.9,
        prediction_shift_threshold=0.1,
        label_shift_threshold=0.1,
        min_feedback_accuracy=0.8,
    )

    assert report["drift_detected"] is True
    assert report["drift_types"]["data_drift"] is True
    assert report["drift_types"]["model_drift"] is True
    assert report["drift_types"]["problem_drift"] is True
    assert report["drift_types"]["concept_drift"] is True
