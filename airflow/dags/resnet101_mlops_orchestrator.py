from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.branch import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule


def _project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # <repo>/airflow/dags/this_file.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _compose_file() -> Path:
    return _project_root() / "docker-compose.yml"


def _monitoring_dir() -> Path:
    return _project_root() / "monitoring"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _branch_bootstrap() -> str:
    root = _project_root()
    candidate_paths = [
        root / "resnet101" / "model_trained" / "mlops" / "best_model.pth",
        root / "resnet101" / "model_trained" / "ResNet101.pth",
    ]
    if any(p.exists() for p in candidate_paths):
        return "bootstrap_skip"
    return "bootstrap_ingestion"


def _branch_retrain() -> str:
    mdir = _monitoring_dir()
    drift_report = _read_json(mdir / "drift_report.json")
    health_report = _read_json(mdir / "model_health_report.json")

    drift_detected = bool(drift_report.get("drift_detected", False))
    model_degraded = bool(health_report.get("degraded", False))

    if drift_detected or model_degraded:
        return "trigger_retraining"
    return "skip_retraining"


def _write_orchestration_report(**context):
    mdir = _monitoring_dir()
    mdir.mkdir(parents=True, exist_ok=True)

    drift_report = _read_json(mdir / "drift_report.json")
    health_report = _read_json(mdir / "model_health_report.json")

    dag_run = context.get("dag_run")
    run_id = dag_run.run_id if dag_run else "unknown"
    execution_date = str(context.get("ds"))

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "execution_date": execution_date,
        "decision": {
            "drift_detected": bool(drift_report.get("drift_detected", False)),
            "model_degraded": bool(health_report.get("degraded", False)),
        },
        "drift_report": drift_report,
        "health_report": health_report,
    }
    (mdir / "orchestration_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


ROOT = _project_root()
COMPOSE = _compose_file()

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="resnet101_mlops_orchestrator",
    default_args=default_args,
    description="Orquestacion MLOps: bootstrap, monitoreo, drift/model health y reentrenamiento condicional.",
    start_date=datetime(2026, 2, 18),
    schedule="0 */2 * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=3),
    tags=["mlops", "resnet101", "drift", "retraining"],
) as dag:
    bootstrap_decision = BranchPythonOperator(
        task_id="bootstrap_decision",
        python_callable=_branch_bootstrap,
    )

    bootstrap_ingestion = BashOperator(
        task_id="bootstrap_ingestion",
        bash_command=(
            f"cd {ROOT} && "
            f"docker compose -f {COMPOSE} run --rm ingestion"
        ),
    )

    bootstrap_training = BashOperator(
        task_id="bootstrap_training",
        bash_command=(
            f"cd {ROOT} && "
            f"docker compose -f {COMPOSE} run --rm training"
        ),
    )

    bootstrap_deploy = BashOperator(
        task_id="bootstrap_deploy",
        bash_command=(
            f"cd {ROOT} && "
            f"docker compose -f {COMPOSE} up -d deploy"
        ),
    )

    bootstrap_skip = EmptyOperator(task_id="bootstrap_skip")

    bootstrap_join = EmptyOperator(
        task_id="bootstrap_join",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    ensure_deploy = BashOperator(
        task_id="ensure_deploy_running",
        bash_command=(
            f"cd {ROOT} && "
            f"docker compose -f {COMPOSE} up -d deploy"
        ),
    )

    detect_drift = BashOperator(
        task_id="detect_drift",
        bash_command=(
            f"cd {ROOT} && "
            f"docker compose -f {COMPOSE} exec -T deploy "
            "python3 -m src.mlops.detect_drift "
            "--reference-stats-path /workspace/data/pet_stats.json "
            "--inference-log-path /workspace/monitoring/inference_events.jsonl "
            "--report-path /workspace/monitoring/drift_report.json "
            "--window-size 500 "
            "--min-samples 50 "
            "--mean-shift-threshold 0.35 "
            "--scale-shift-threshold 0.25 "
            "--min-avg-confidence 0.60 "
            "--exit-code-on-drift 0"
        ),
    )

    evaluate_health = BashOperator(
        task_id="evaluate_model_health",
        bash_command=(
            f"cd {ROOT} && "
            f"docker compose -f {COMPOSE} exec -T deploy "
            "python3 -m src.mlops.evaluate_model_health "
            "--inference-log-path /workspace/monitoring/inference_events.jsonl "
            "--feedback-log-path /workspace/monitoring/feedback_events.jsonl "
            "--report-path /workspace/monitoring/model_health_report.json "
            "--window-size 500 "
            "--min-samples 50 "
            "--stale-hours 48 "
            "--min-avg-confidence 0.60 "
            "--uncertain-threshold 0.55 "
            "--max-uncertain-rate 0.40 "
            "--min-feedback-samples 20 "
            "--min-feedback-accuracy 0.80"
        ),
    )

    retrain_decision = BranchPythonOperator(
        task_id="retrain_decision",
        python_callable=_branch_retrain,
    )

    trigger_retraining = BashOperator(
        task_id="trigger_retraining",
        bash_command=(
            f"cd {ROOT} && "
            f"docker compose -f {COMPOSE} run --rm training"
        ),
    )

    rollout_new_model = BashOperator(
        task_id="rollout_new_model",
        bash_command=(
            f"cd {ROOT} && "
            f"docker compose -f {COMPOSE} up -d deploy"
        ),
    )

    skip_retraining = EmptyOperator(task_id="skip_retraining")

    retrain_join = EmptyOperator(
        task_id="retrain_join",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    orchestration_report = PythonOperator(
        task_id="write_orchestration_report",
        python_callable=_write_orchestration_report,
    )

    end = EmptyOperator(task_id="end")

    bootstrap_decision >> bootstrap_ingestion >> bootstrap_training >> bootstrap_deploy >> bootstrap_join
    bootstrap_decision >> bootstrap_skip >> bootstrap_join

    bootstrap_join >> ensure_deploy
    ensure_deploy >> [detect_drift, evaluate_health] >> retrain_decision

    retrain_decision >> trigger_retraining >> rollout_new_model >> retrain_join
    retrain_decision >> skip_retraining >> retrain_join

    retrain_join >> orchestration_report >> end
