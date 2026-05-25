# DataOps for Oxford-IIIT Pets

This directory defines the operational data contract used by the project. The goal is for training, validation, inference and monitoring to share the same data rules, not only the same files.

## Scope

- Source dataset: Oxford-IIIT Pet Dataset.
- Problem: binary classification `cat` vs `dog`.
- Data unit: RGB JPEG/PNG image.
- Model input: `1x3x224x224` tensor.
- Normalization contract: `data/pet_stats.json`.
- Reproducible preparation code: `data/create_dataset/preprocess_training_data.py`.

## Governance

- The full dataset is not versioned in Git. It must be downloaded from the original source and used according to its license.
- Versionable artifacts are contracts, statistics, minimal examples and reproducible reports.
- Raw data is treated as immutable; any preprocessing change must generate a new report and update the contract.
- The positive class for binary metrics is `dog`.
- Operational outputs are stored under `monitoring/` for auditability.

## Data Contract

The `data/data_governance.yaml` file declares:

- source, license and restrictions;
- expected image schema;
- split rules;
- training and inference transformations;
- minimum quality checks;
- production monitoring signals.

The `data/pet_stats.json` file is a contract artifact. If it changes, the model must be retrained or at least pass a formal compatibility validation.

## Reproducible Preprocessing

Run:

```bash
python3 data/create_dataset/preprocess_training_data.py \
  --config-path resnet101/oxford_pets_binary_resnet101.yaml \
  --report-path monitoring/dataops_preprocess_report.json
```

The script downloads/prepares Oxford Pets when needed, applies the configured split and validates that the training loader uses the same statistics declared for inference.

## DataOps Checks

The minimum expected checks are:

- dataset is downloadable and readable;
- `cat` and `dog` classes are present;
- splits are reproducible by seed;
- normalization has three channels;
- model input image is `224x224`;
- train/validation splits are not empty;
- JSON report is persisted for auditability.

## Relationship With Monitoring

Monitoring compares inference against this contract:

- data drift: RGB means and standard deviations vs `pet_stats.json`;
- model drift: confidence, uncertainty and prediction distribution;
- problem drift: real-label distribution observed via feedback;
- concept/performance drift: accuracy, precision, recall, F1 and ROC-AUC when labels are available.
