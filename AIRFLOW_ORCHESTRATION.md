# Airflow Orchestration (MLOps)

## Qué incluye

- DAG: `airflow/dags/resnet101_mlops_orchestrator.py`
- Health check de comportamiento del modelo: `src/mlops/evaluate_model_health.py`
- Stack Airflow: `docker-compose.airflow.yml` + `docker/airflow.Dockerfile`

## Lógica del DAG

1. `bootstrap_decision`:
   - Si no existe modelo entrenado, ejecuta `ingestion -> training -> deploy`.
   - Si existe, salta bootstrap.
2. `ensure_deploy_running`:
   - Garantiza API desplegada.
3. Monitoreo:
   - `detect_drift` (datos de inferencia).
   - `evaluate_model_health` (confianza, incertidumbre y accuracy de feedback si hay etiquetas reales).
4. `retrain_decision`:
   - Reentrena si `drift_detected` o `model_degraded`.
5. Si reentrena:
   - `trigger_retraining` -> `rollout_new_model`.
6. Siempre:
   - escribe `monitoring/orchestration_report.json`.

## Levantar Airflow

```bash
docker compose -f docker-compose.airflow.yml up -d --build
```

Airflow UI:

- `http://localhost:8080`

El contenedor se levanta con `airflow standalone` y crea usuario/admin automáticamente en logs.

## Ejecutar DAG

Nombre DAG:

- `resnet101_mlops_orchestrator`

Puedes dispararlo manualmente desde UI o dejar schedule activo (`cada 2 horas`).

## Archivos de monitoreo usados por el DAG

- `monitoring/inference_events.jsonl` (generado por la API en inferencia).
- `monitoring/feedback_events.jsonl` (opcional, labels reales para medir accuracy real).
- `monitoring/drift_report.json`
- `monitoring/model_health_report.json`
- `monitoring/orchestration_report.json`

## Formato recomendado para feedback real

Una línea JSON por predicción validada:

```json
{"timestamp_utc":"2026-02-18T20:00:00+00:00","prediction":"cat","true_label":"dog"}
```

o

```json
{"timestamp_utc":"2026-02-18T20:00:00+00:00","prediction_correct":false}
```

