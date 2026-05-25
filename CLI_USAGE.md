# CLI Usage

## Proyecto principal (`scripts/`)

- API:
  - `python3 scripts/run_api.py --host 0.0.0.0 --port 8000`
- App Streamlit:
  - `python3 scripts/run_app.py --host 0.0.0.0 --port 8501`

## Módulo `resnet101` (`resnet101/scripts/`)

- Ingesta:
  - `python3 resnet101/scripts/cli_ingest.py --data-dir data --stats-path data/pet_stats.json`
- Entrenamiento:
  - `python3 resnet101/scripts/cli_train.py --config config/model/oxford_pets_binary_resnet101.yaml --output-dir resnet101/model_trained/mlops --tracking-uri file:./resnet101/mlruns`
- Inferencia:
  - `python3 resnet101/scripts/cli_infer.py --image-path data/processed/examples_oxford/cat_example.jpg --pretty`
  - o `python3 resnet101/scripts/cli_infer.py --url https://... --pretty`

