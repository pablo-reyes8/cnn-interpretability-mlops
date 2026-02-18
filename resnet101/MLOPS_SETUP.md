# MLOps base para ResNet101 (MLflow + DVC)

## 1) Instalar dependencias MLOps

```bash
pip install -r requirements-mlops.txt
```

## 2) Ejecutar entrenamiento con MLflow

```bash
python3 -m resnet101.src.training.train_mlflow \
  --config resnet101/oxford_pets_binary_resnet101.yaml \
  --output-dir resnet101/model_trained/mlops \
  --tracking-uri file:./resnet101/mlruns
```

Artefactos principales:

- `resnet101/model_trained/mlops/best_model.pth`
- `resnet101/model_trained/mlops/last_model.pth`
- `resnet101/model_trained/mlops/metrics.json`
- `resnet101/mlruns/` (tracking local de MLflow)

## 3) Visualizar MLflow UI

```bash
mlflow ui --backend-store-uri file:./resnet101/mlruns --port 5000
```

Luego abrir: `http://127.0.0.1:5000`

## 4) Flujo base DVC

Inicializar DVC (solo una vez por repo):

```bash
dvc init
```

Ejecutar pipeline:

```bash
dvc repro
```

Ver métricas:

```bash
dvc metrics show
```

## 5) Versionado del dataset con DVC

Cuando ya tengas el dataset local (por ejemplo `data/oxford-iiit-pet/`):

```bash
dvc add data/oxford-iiit-pet
git add data/oxford-iiit-pet.dvc .gitignore
git commit -m "Track Oxford-IIIT dataset with DVC"
```

Opcional: configurar remote (S3, GDrive, etc.) para compartir datos del equipo.
