# Airflow Orchestration (MLOps)

## Qué incluye

- DAG: `airflow/dags/resnet101_mlops_orchestrator.py`
- Configuración centralizada: `config/params.yaml`, `config/mlops.yaml`, `config/api.yaml`
- Validación DataOps: `data/create_dataset/preprocess_training_data.py`
- Monitoreo de software: `src/mlops/software_monitor.py`
- Drift por tipo: `src/mlops/detect_drift.py`
- Health check del modelo: `src/mlops/evaluate_model_health.py`
- Quality gate post-entrenamiento: `src/mlops/quality_gate.py`
- Gestor de promotion/rollback: `src/mlops/deployment_manager.py`
- Stack Airflow: `docker-compose.airflow.yml` + `docker/airflow.Dockerfile`

## Diseño del orquestador

Se mantiene un solo DAG porque el flujo tiene dependencias fuertes entre bootstrap, monitoreo, reentrenamiento, promotion y rollback. En vez de partirlo en DAGs separados, se organiza con `TaskGroup`:

- `bootstrap`: prepara datos, valida DataOps, entrena si no existe modelo activo, ejecuta quality gate, promueve y valida post-deploy.
- `monitoring`: asegura deploy activo y genera reportes de software, drift y salud del modelo.
- `retraining`: se ejecuta solo si el branch detecta drift o degradación; entrena, valida, promueve, verifica post-deploy y hace rollback si corresponde.

Esta estructura evita un DAG plano gigante, mantiene trazabilidad en una sola corrida y deja claro qué parte del ciclo MLOps falló.

## Lógica del DAG

1. Bootstrap (si no hay modelo en producción):
   - `ingestion -> dataops_preprocess -> training -> quality_gate`.
   - Si pasa gate: `promote -> deploy -> post_deploy_health`.
   - Si falla gate o post-deploy sale degradado: `rollback`.
2. Monitoreo continuo:
   - `monitor_software`: latencia, preprocesamiento, inferencia y error rate.
   - `detect_drift`: data drift, model drift, problem drift y concept drift.
   - `evaluate_model_health`: confianza, incertidumbre, accuracy, precision, recall, F1 y ROC-AUC si hay feedback.
3. Reentrenamiento condicional:
   - Si `drift_report.drift_detected=true` o `model_health_report.degraded=true`: `training -> quality_gate`.
   - Si pasa gate: `promote -> rollout`.
   - Si falla o se degrada post-rollout: `rollback` automático.
4. Trazabilidad:
   - `quality_gate_history.jsonl`
   - `deployment_history.jsonl`
   - `rollback_history.jsonl`
   - `orchestration_report.json`

## Levantar Airflow

```bash
docker compose -f docker-compose.airflow.yml up -d --build
```

Airflow UI:

- `http://localhost:8080`

## DAG

- Nombre: `resnet101_mlops_orchestrator`
- Schedule: cada 2 horas (`0 */2 * * *`)

## Archivos de monitoreo usados por el DAG

- `monitoring/inference_events.jsonl` (inferencia real)
- `monitoring/feedback_events.jsonl` (opcional con etiquetas reales)
- `monitoring/software_metrics_report.json`
- `monitoring/drift_report.json`
- `monitoring/model_health_report.json`
- `monitoring/dataops_preprocess_report.json`
- `monitoring/quality_gate_report_bootstrap.json`
- `monitoring/quality_gate_report_retrain.json`
- `monitoring/deployment_state.json`
- `monitoring/deployment_history.jsonl`
- `monitoring/rollback_history.jsonl`

## Formato recomendado para feedback

```json
{"timestamp_utc":"2026-02-18T20:00:00+00:00","prediction":"cat","true_label":"dog","scores":{"cat":0.72,"dog":0.28}}
```

o

```json
{"timestamp_utc":"2026-02-18T20:00:00+00:00","prediction_correct":false}
```
