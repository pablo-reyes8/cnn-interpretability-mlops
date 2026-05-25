from __future__ import annotations

import json
import os
import shlex
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.branch import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule


def _project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
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


def _sh(value: str | Path) -> str:
    return shlex.quote(str(value))


def _local(command: str) -> str:
    return f"cd {_sh(ROOT)} && {command}"


def _compose(command: str) -> str:
    return _local(f"docker compose -f {_sh(COMPOSE)} {command}")


def _deploy_exec(command: str) -> str:
    return _compose(f"exec -T deploy {command}")


def _quality_gate_command(report_name: str) -> str:
    return _local(
        "python3 -m src.mlops.quality_gate "
        "--candidate-metrics-path resnet101/model_trained/mlops/metrics.json "
        "--candidate-model-path resnet101/model_trained/mlops/best_model.pth "
        "--state-path monitoring/deployment_state.json "
        f"--report-path monitoring/{report_name} "
        "--history-path monitoring/quality_gate_history.jsonl "
        "--exit-code-on-fail 0"
    )


def _promote_command(report_name: str) -> str:
    return _local(
        "python3 -m src.mlops.deployment_manager "
        "--action promote "
        f"--gate-report-path monitoring/{report_name} "
        "--candidate-model-path resnet101/model_trained/mlops/best_model.pth "
        "--target-model-path resnet101/model_trained/ResNet101.pth "
        "--state-path monitoring/deployment_state.json "
        "--deployment-history-path monitoring/deployment_history.jsonl "
        "--rollback-history-path monitoring/rollback_history.jsonl "
        "--backup-dir resnet101/model_trained/backups"
    )


def _rollback_command(reason: str) -> str:
    return _local(
        "python3 -m src.mlops.deployment_manager "
        "--action rollback "
        "--target-model-path resnet101/model_trained/ResNet101.pth "
        "--state-path monitoring/deployment_state.json "
        "--rollback-history-path monitoring/rollback_history.jsonl "
        f"--reason {shlex.quote(reason)} "
        "--allow-noop-rollback"
    )


def _model_health_command(report_path: str, *, window_size: int, min_samples: int) -> str:
    return _deploy_exec(
        "python3 -m src.mlops.evaluate_model_health "
        "--inference-log-path /workspace/monitoring/inference_events.jsonl "
        "--feedback-log-path /workspace/monitoring/feedback_events.jsonl "
        f"--report-path /workspace/monitoring/{report_path} "
        f"--window-size {window_size} "
        f"--min-samples {min_samples} "
        "--stale-hours 48 "
        "--min-avg-confidence 0.60 "
        "--uncertain-threshold 0.55 "
        "--max-uncertain-rate 0.40 "
        "--min-feedback-samples 20 "
        "--min-feedback-accuracy 0.80 "
        "--min-feedback-roc-auc 0.80 "
        "--positive-label dog"
    )


def _branch_bootstrap() -> str:
    root = _project_root()
    candidate_paths = [
        root / "resnet101" / "model_trained" / "ResNet101.pth",
        root / "monitoring" / "deployment_state.json",
    ]
    if any(p.exists() for p in candidate_paths):
        return "bootstrap.bootstrap_skip"
    return "bootstrap.bootstrap_ingestion"


def _branch_retrain_need() -> str:
    mdir = _monitoring_dir()
    drift_report = _read_json(mdir / "drift_report.json")
    health_report = _read_json(mdir / "model_health_report.json")
    drift_detected = bool(drift_report.get("drift_detected", False))
    model_degraded = bool(health_report.get("degraded", False))
    if drift_detected or model_degraded:
        return "retraining.retrain_training"
    return "retraining.skip_retraining"


def _branch_quality_gate(report_file: str, pass_task_id: str, fail_task_id: str) -> str:
    report = _read_json(_monitoring_dir() / report_file)
    return pass_task_id if bool(report.get("gate_passed", False)) else fail_task_id


def _branch_post_deploy_health(report_file: str, healthy_task_id: str, rollback_task_id: str) -> str:
    report = _read_json(_monitoring_dir() / report_file)
    degraded = bool(report.get("degraded", False))
    status = str(report.get("status", "unknown"))
    if degraded and status in {"ok", "stale"}:
        return rollback_task_id
    return healthy_task_id


def _write_orchestration_report(**context):
    mdir = _monitoring_dir()
    mdir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "run_id": context.get("dag_run").run_id if context.get("dag_run") else "unknown",
        "execution_date": str(context.get("ds")),
        "monitoring": {
            "software": _read_json(mdir / "software_metrics_report.json"),
            "drift": _read_json(mdir / "drift_report.json"),
            "model_health": _read_json(mdir / "model_health_report.json"),
        },
        "quality_gates": {
            "bootstrap": _read_json(mdir / "quality_gate_report_bootstrap.json"),
            "retrain": _read_json(mdir / "quality_gate_report_retrain.json"),
        },
        "post_deploy_health": {
            "bootstrap": _read_json(mdir / "post_deploy_health_bootstrap.json"),
            "retrain": _read_json(mdir / "post_deploy_health_retrain.json"),
        },
        "deployment_state": _read_json(mdir / "deployment_state.json"),
    }
    (mdir / "orchestration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


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
    description="MLOps orchestration with DataOps validation, monitoring, retraining, promotion and rollback.",
    start_date=datetime(2026, 2, 18),
    schedule="0 */2 * * *",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=4),
    tags=["mlops", "resnet101", "monitoring", "retraining", "rollback"],
) as dag:
    with TaskGroup("bootstrap") as bootstrap:
        bootstrap_decision = BranchPythonOperator(
            task_id="bootstrap_decision",
            python_callable=_branch_bootstrap,
        )

        bootstrap_ingestion = BashOperator(
            task_id="bootstrap_ingestion",
            bash_command=_compose("run --rm ingestion"),
        )

        bootstrap_dataops_preprocess = BashOperator(
            task_id="bootstrap_dataops_preprocess",
            bash_command=_local(
                "python3 data/create_dataset/preprocess_training_data.py "
                "--config-path config/model/oxford_pets_binary_resnet101.yaml "
                "--report-path monitoring/dataops_preprocess_report.json"
            ),
        )

        bootstrap_training = BashOperator(
            task_id="bootstrap_training",
            bash_command=_compose("run --rm training"),
        )

        bootstrap_quality_gate = BashOperator(
            task_id="bootstrap_quality_gate",
            bash_command=_quality_gate_command("quality_gate_report_bootstrap.json"),
        )

        bootstrap_gate_decision = BranchPythonOperator(
            task_id="bootstrap_gate_decision",
            python_callable=_branch_quality_gate,
            op_kwargs={
                "report_file": "quality_gate_report_bootstrap.json",
                "pass_task_id": "bootstrap.bootstrap_promote",
                "fail_task_id": "bootstrap.bootstrap_gate_failed_rollback",
            },
        )

        bootstrap_promote = BashOperator(
            task_id="bootstrap_promote",
            bash_command=_promote_command("quality_gate_report_bootstrap.json"),
        )

        bootstrap_deploy = BashOperator(
            task_id="bootstrap_deploy",
            bash_command=_compose("up -d deploy"),
        )

        bootstrap_post_deploy_health = BashOperator(
            task_id="bootstrap_post_deploy_health",
            bash_command=_model_health_command("post_deploy_health_bootstrap.json", window_size=200, min_samples=20),
        )

        bootstrap_post_deploy_decision = BranchPythonOperator(
            task_id="bootstrap_post_deploy_decision",
            python_callable=_branch_post_deploy_health,
            op_kwargs={
                "report_file": "post_deploy_health_bootstrap.json",
                "healthy_task_id": "bootstrap.bootstrap_ready",
                "rollback_task_id": "bootstrap.bootstrap_rollback",
            },
        )

        bootstrap_rollback = BashOperator(
            task_id="bootstrap_rollback",
            bash_command=_rollback_command("bootstrap_post_deploy_degraded"),
        )

        bootstrap_redeploy_after_rollback = BashOperator(
            task_id="bootstrap_redeploy_after_rollback",
            bash_command=_compose("up -d deploy"),
        )

        bootstrap_gate_failed_rollback = BashOperator(
            task_id="bootstrap_gate_failed_rollback",
            bash_command=_rollback_command("bootstrap_quality_gate_failed"),
        )

        bootstrap_skip = EmptyOperator(task_id="bootstrap_skip")

        bootstrap_ready = EmptyOperator(
            task_id="bootstrap_ready",
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        bootstrap_decision >> bootstrap_ingestion >> bootstrap_dataops_preprocess >> bootstrap_training
        bootstrap_training >> bootstrap_quality_gate >> bootstrap_gate_decision
        bootstrap_gate_decision >> bootstrap_promote >> bootstrap_deploy >> bootstrap_post_deploy_health
        bootstrap_post_deploy_health >> bootstrap_post_deploy_decision
        bootstrap_post_deploy_decision >> bootstrap_rollback >> bootstrap_redeploy_after_rollback >> bootstrap_ready
        bootstrap_post_deploy_decision >> bootstrap_ready
        bootstrap_gate_decision >> bootstrap_gate_failed_rollback >> bootstrap_ready
        bootstrap_decision >> bootstrap_skip >> bootstrap_ready

    ensure_deploy_running = BashOperator(
        task_id="ensure_deploy_running",
        bash_command=_compose("up -d deploy"),
    )

    with TaskGroup("monitoring") as monitoring:
        monitor_software = BashOperator(
            task_id="monitor_software",
            bash_command=_deploy_exec(
                "python3 -m src.mlops.software_monitor "
                "--inference-log-path /workspace/monitoring/inference_events.jsonl "
                "--report-path /workspace/monitoring/software_metrics_report.json "
                "--window-size 500 "
                "--max-p95-latency-ms 2000 "
                "--max-p95-inference-ms 800"
            ),
        )

        detect_drift = BashOperator(
            task_id="detect_drift",
            bash_command=_deploy_exec(
                "python3 -m src.mlops.detect_drift "
                "--reference-stats-path /workspace/data/pet_stats.json "
                "--inference-log-path /workspace/monitoring/inference_events.jsonl "
                "--feedback-log-path /workspace/monitoring/feedback_events.jsonl "
                "--report-path /workspace/monitoring/drift_report.json "
                "--window-size 500 "
                "--min-samples 50 "
                "--mean-shift-threshold 0.35 "
                "--scale-shift-threshold 0.25 "
                "--min-avg-confidence 0.60 "
                "--prediction-shift-threshold 0.30 "
                "--label-shift-threshold 0.30 "
                "--min-feedback-samples 20 "
                "--min-feedback-accuracy 0.80 "
                "--exit-code-on-drift 0"
            ),
        )

        evaluate_model_health = BashOperator(
            task_id="evaluate_model_health",
            bash_command=_model_health_command("model_health_report.json", window_size=500, min_samples=50),
        )

    retrain_decision = BranchPythonOperator(
        task_id="retrain_decision",
        python_callable=_branch_retrain_need,
    )

    with TaskGroup("retraining") as retraining:
        retrain_training = BashOperator(
            task_id="retrain_training",
            bash_command=_compose("run --rm training"),
        )

        retrain_quality_gate = BashOperator(
            task_id="retrain_quality_gate",
            bash_command=_quality_gate_command("quality_gate_report_retrain.json"),
        )

        retrain_gate_decision = BranchPythonOperator(
            task_id="retrain_gate_decision",
            python_callable=_branch_quality_gate,
            op_kwargs={
                "report_file": "quality_gate_report_retrain.json",
                "pass_task_id": "retraining.retrain_promote",
                "fail_task_id": "retraining.retrain_gate_failed_rollback",
            },
        )

        retrain_promote = BashOperator(
            task_id="retrain_promote",
            bash_command=_promote_command("quality_gate_report_retrain.json"),
        )

        rollout_new_model = BashOperator(
            task_id="rollout_new_model",
            bash_command=_compose("up -d deploy"),
        )

        retrain_post_deploy_health = BashOperator(
            task_id="retrain_post_deploy_health",
            bash_command=_model_health_command("post_deploy_health_retrain.json", window_size=300, min_samples=20),
        )

        retrain_post_deploy_decision = BranchPythonOperator(
            task_id="retrain_post_deploy_decision",
            python_callable=_branch_post_deploy_health,
            op_kwargs={
                "report_file": "post_deploy_health_retrain.json",
                "healthy_task_id": "retraining.retrain_done",
                "rollback_task_id": "retraining.retrain_rollback",
            },
        )

        retrain_rollback = BashOperator(
            task_id="retrain_rollback",
            bash_command=_rollback_command("retrain_post_deploy_degraded"),
        )

        retrain_redeploy_after_rollback = BashOperator(
            task_id="retrain_redeploy_after_rollback",
            bash_command=_compose("up -d deploy"),
        )

        retrain_gate_failed_rollback = BashOperator(
            task_id="retrain_gate_failed_rollback",
            bash_command=_rollback_command("retrain_quality_gate_failed"),
        )

        skip_retraining = EmptyOperator(task_id="skip_retraining")

        retrain_done = EmptyOperator(
            task_id="retrain_done",
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        retrain_training >> retrain_quality_gate >> retrain_gate_decision
        retrain_gate_decision >> retrain_promote >> rollout_new_model >> retrain_post_deploy_health
        retrain_post_deploy_health >> retrain_post_deploy_decision
        retrain_post_deploy_decision >> retrain_rollback >> retrain_redeploy_after_rollback >> retrain_done
        retrain_post_deploy_decision >> retrain_done
        retrain_gate_decision >> retrain_gate_failed_rollback >> retrain_done
        skip_retraining >> retrain_done

    write_orchestration_report = PythonOperator(
        task_id="write_orchestration_report",
        python_callable=_write_orchestration_report,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    end = EmptyOperator(task_id="end")

    bootstrap >> ensure_deploy_running >> monitoring >> retrain_decision
    retrain_decision >> retraining
    retraining >> write_orchestration_report >> end
