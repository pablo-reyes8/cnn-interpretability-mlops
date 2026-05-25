import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.mlops.drift_core import analyze_drift


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecta drift y dispara reentrenamiento automatico si aplica."
    )
    parser.add_argument("--reference-stats-path", type=str, default="data/pet_stats.json")
    parser.add_argument("--inference-log-path", type=str, default="monitoring/inference_events.jsonl")
    parser.add_argument("--drift-report-path", type=str, default="monitoring/drift_report.json")
    parser.add_argument("--retrain-report-path", type=str, default="monitoring/retrain_report.json")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--mean-shift-threshold", type=float, default=0.35)
    parser.add_argument("--scale-shift-threshold", type=float, default=0.25)
    parser.add_argument("--min-avg-confidence", type=float, default=0.60)

    parser.add_argument(
        "--train-config",
        type=str,
        default="config/model/oxford_pets_binary_resnet101.yaml",
    )
    parser.add_argument(
        "--train-output-dir",
        type=str,
        default="resnet101/model_trained/mlops",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="file:./resnet101/mlruns",
    )
    parser.add_argument("--run-name", type=str, default="retrain-on-drift")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _run_training(args) -> tuple[int, list[str]]:
    cmd = [
        "python3",
        "-m",
        "resnet101.src.training.train_mlflow",
        "--config",
        args.train_config,
        "--output-dir",
        args.train_output_dir,
        "--tracking-uri",
        args.tracking_uri,
        "--run-name",
        args.run_name,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])

    print("[INFO] Ejecutando reentrenamiento por drift...")
    print("[INFO] " + " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode, cmd


def main():
    args = parse_args()
    drift_report = analyze_drift(
        reference_stats_path=args.reference_stats_path,
        inference_log_path=args.inference_log_path,
        window_size=args.window_size,
        min_samples=args.min_samples,
        mean_shift_threshold=args.mean_shift_threshold,
        scale_shift_threshold=args.scale_shift_threshold,
        min_avg_confidence=args.min_avg_confidence,
    )

    drift_report_path = Path(args.drift_report_path)
    drift_report_path.parent.mkdir(parents=True, exist_ok=True)
    drift_report_path.write_text(json.dumps(drift_report, indent=2), encoding="utf-8")

    retrain_payload = {
        "drift_detected": bool(drift_report.get("drift_detected", False)),
        "drift_status": drift_report.get("status"),
        "triggered_retraining": False,
        "training_command": [],
        "training_return_code": None,
        "drift_report_path": str(drift_report_path),
    }

    exit_code = 0
    if drift_report.get("drift_detected"):
        rc, cmd = _run_training(args)
        retrain_payload["triggered_retraining"] = True
        retrain_payload["training_command"] = cmd
        retrain_payload["training_return_code"] = rc
        exit_code = rc
    else:
        print("[INFO] No se detecto drift. No se ejecuta reentrenamiento.")

    retrain_report_path = Path(args.retrain_report_path)
    retrain_report_path.parent.mkdir(parents=True, exist_ok=True)
    retrain_report_path.write_text(json.dumps(retrain_payload, indent=2), encoding="utf-8")
    print(f"[OK] Reporte de reentrenamiento: {retrain_report_path}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

