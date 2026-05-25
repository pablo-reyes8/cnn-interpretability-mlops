import argparse
import json
import sys
from pathlib import Path

from src.mlops.drift_core import analyze_drift


def parse_args():
    parser = argparse.ArgumentParser(description="Deteccion de drift de datos en inferencia.")
    parser.add_argument("--reference-stats-path", type=str, default="data/pet_stats.json")
    parser.add_argument("--inference-log-path", type=str, default="monitoring/inference_events.jsonl")
    parser.add_argument("--feedback-log-path", type=str, default="monitoring/feedback_events.jsonl")
    parser.add_argument("--report-path", type=str, default="monitoring/drift_report.json")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--mean-shift-threshold", type=float, default=0.35)
    parser.add_argument("--scale-shift-threshold", type=float, default=0.25)
    parser.add_argument("--min-avg-confidence", type=float, default=0.60)
    parser.add_argument("--prediction-shift-threshold", type=float, default=0.30)
    parser.add_argument("--label-shift-threshold", type=float, default=0.30)
    parser.add_argument("--min-feedback-samples", type=int, default=20)
    parser.add_argument("--min-feedback-accuracy", type=float, default=0.80)
    parser.add_argument("--exit-code-on-drift", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    report = analyze_drift(
        reference_stats_path=args.reference_stats_path,
        inference_log_path=args.inference_log_path,
        feedback_log_path=args.feedback_log_path,
        window_size=args.window_size,
        min_samples=args.min_samples,
        mean_shift_threshold=args.mean_shift_threshold,
        scale_shift_threshold=args.scale_shift_threshold,
        min_avg_confidence=args.min_avg_confidence,
        prediction_shift_threshold=args.prediction_shift_threshold,
        label_shift_threshold=args.label_shift_threshold,
        min_feedback_samples=args.min_feedback_samples,
        min_feedback_accuracy=args.min_feedback_accuracy,
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Reporte de drift: {report_path}")
    print(f"[OK] Estado: {report.get('status')}")
    print(f"[OK] Drift detectado: {report.get('drift_detected')}")

    if report.get("drift_detected"):
        sys.exit(args.exit_code_on_drift)
    sys.exit(0)


if __name__ == "__main__":
    main()

