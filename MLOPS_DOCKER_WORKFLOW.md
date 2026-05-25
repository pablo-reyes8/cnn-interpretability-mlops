# MLOps Docker Workflow

## Arquitectura de contenedores

El `docker-compose.yml` ahora separa el ciclo MLOps en 3 servicios:

- `ingestion`: descarga dataset y genera stats de referencia.
- `training`: entrena con MLflow y guarda artefactos del modelo.
- `deploy`: sirve la API de inferencia en FastAPI.

## Comandos recomendados

### 1) Ingesta de datos

```bash
docker compose run --rm ingestion
```

Salida esperada:

- `data/pet_stats.json`
- `monitoring/ingestion_report.json`

### 2) Entrenamiento

```bash
docker compose run --rm training
```

Salida esperada:

- `resnet101/model_trained/mlops/best_model.pth`
- `resnet101/model_trained/mlops/last_model.pth`
- `resnet101/model_trained/mlops/metrics.json`
- `resnet101/mlruns/`

### 3) Despliegue de inferencia

```bash
docker compose up -d deploy
```

API:

- `http://localhost:8000/docs`

Durante inferencia se registra monitoreo en:

- `monitoring/inference_events.jsonl`

## Drift detection (inferencia)

```bash
python3 -m src.mlops.detect_drift \
  --reference-stats-path data/pet_stats.json \
  --inference-log-path monitoring/inference_events.jsonl \
  --report-path monitoring/drift_report.json
```

Notas:

- Exit code `0`: sin drift.
- Exit code `2`: drift detectado.

## Reentrenamiento automatico por drift

```bash
python3 -m src.mlops.retrain_if_drift \
  --reference-stats-path data/pet_stats.json \
  --inference-log-path monitoring/inference_events.jsonl \
  --drift-report-path monitoring/drift_report.json \
  --retrain-report-path monitoring/retrain_report.json \
  --train-config config/model/oxford_pets_binary_resnet101.yaml \
  --train-output-dir resnet101/model_trained/mlops \
  --tracking-uri file:./resnet101/mlruns
```

Si hay drift, ejecuta `train_mlflow`; si no, deja reporte y termina sin reentrenar.

