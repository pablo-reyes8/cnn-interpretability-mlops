# Monitoring MLOps

This folder contains operational logs and reports for the deployed model. It does not store user images; it records numeric signals needed for software health, drift and performance monitoring.

## Main Logs

- `inference_events.jsonl`: one event per successful prediction.
- `feedback_events.jsonl`: real labels or later corrections, when available.
- `deployment_history.jsonl`: model promotions.
- `rollback_history.jsonl`: operational rollbacks.

## Generated Reports

- `software_metrics_report.json`: total latency, preprocessing, inference, errors and endpoint counts.
- `drift_report.json`: data, model, problem and concept drift.
- `model_health_report.json`: confidence, uncertainty and supervised metrics.
- `quality_gate_report.json`: pre-promotion model validation.
- `dataops_preprocess_report.json`: reproducible data preparation contract.

## Commands

```bash
python3 -m src.mlops.software_monitor \
  --inference-log-path monitoring/inference_events.jsonl \
  --report-path monitoring/software_metrics_report.json

python3 -m src.mlops.detect_drift \
  --reference-stats-path data/pet_stats.json \
  --inference-log-path monitoring/inference_events.jsonl \
  --feedback-log-path monitoring/feedback_events.jsonl \
  --report-path monitoring/drift_report.json

python3 -m src.mlops.evaluate_model_health \
  --inference-log-path monitoring/inference_events.jsonl \
  --feedback-log-path monitoring/feedback_events.jsonl \
  --report-path monitoring/model_health_report.json
```

## Monitoring Types

- Software: `total_latency_ms`, `preprocess_ms`, `inference_ms`, endpoint counts and error rate.
- Data drift: RGB mean and standard deviation shifts against `data/pet_stats.json`.
- Model drift: lower confidence, higher uncertainty or anomalous prediction distribution.
- Problem drift: shift in the observed real-label distribution.
- Concept/performance drift: deterioration in accuracy, precision, recall, F1 or ROC-AUC with labeled feedback.

## Feedback Format

Each line in `feedback_events.jsonl` must be JSON. Example:

```json
{"timestamp_utc":"2026-05-25T10:00:00+00:00","prediction":"dog","true_label":"cat","scores":{"cat":0.31,"dog":0.69}}
```

With this format, accuracy, precision, recall, F1 and ROC-AUC are computed for the positive class `dog`.
