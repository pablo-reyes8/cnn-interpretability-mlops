import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from resnet101.src.data.load_data import get_oxford_pet_loaders
from resnet101.src.model.resnet import ResNet
from resnet101.src.model.save_model import save_checkpoint
from resnet101.src.testing_utils.evaluate_model import evaluate_model
from resnet101.src.training.train_loop import train_epoch_classification


def _read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_optimizer(model, optimizer_cfg):
    name = str(optimizer_cfg.get("name", "Adam")).lower()
    lr = float(optimizer_cfg.get("lr", 1e-3))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    if name == "adam":
        betas = optimizer_cfg.get("betas", [0.9, 0.999])
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(float(betas[0]), float(betas[1])),
            weight_decay=weight_decay,
        )
    if name == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Optimizer no soportado: {optimizer_cfg.get('name')}")


def _build_scheduler(optimizer, scheduler_cfg):
    if not scheduler_cfg:
        return None

    name = str(scheduler_cfg.get("name", "")).lower()
    if not name or name == "none":
        return None
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_cfg.get("step_size", 10)),
            gamma=float(scheduler_cfg.get("gamma", 0.1)),
        )
    if name == "reducelronplateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "min")),
            factor=float(scheduler_cfg.get("factor", 0.1)),
            patience=int(scheduler_cfg.get("patience", 5)),
            min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
        )
    raise ValueError(f"Scheduler no soportado: {scheduler_cfg.get('name')}")


def _build_criterion(criterion_cfg):
    name = str(criterion_cfg.get("name", "CrossEntropyLoss")).lower()
    if name == "crossentropyloss":
        return torch.nn.CrossEntropyLoss()
    if name == "bcewithlogitsloss":
        return torch.nn.BCEWithLogitsLoss()
    raise ValueError(f"Criterio no soportado: {criterion_cfg.get('name')}")


def _import_mlflow():
    try:
        import mlflow
        import mlflow.pytorch
    except ImportError as exc:
        raise RuntimeError(
            "MLflow no esta instalado. Instala dependencias MLOps con: "
            "`pip install -r requirements-mlops.txt`"
        ) from exc
    return mlflow


def _collect_run_params(cfg, seed, device, epochs, output_dir):
    data_cfg = cfg.get("data", {})
    loader_cfg = data_cfg.get("loader", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    optimizer_cfg = training_cfg.get("optimizer", {})
    scheduler_cfg = training_cfg.get("scheduler", {})

    blocks = model_cfg.get("blocks_per_stage", [])
    return {
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "data.dataset": data_cfg.get("dataset"),
        "data.task": data_cfg.get("task"),
        "data.data_dir": data_cfg.get("data_dir"),
        "data.img_size": data_cfg.get("img_size"),
        "data.batch_size": loader_cfg.get("batch_size"),
        "data.num_workers": loader_cfg.get("num_workers"),
        "model.first_block": model_cfg.get("first_block"),
        "model.init": model_cfg.get("init"),
        "model.blocks_per_stage": ",".join(str(x) for x in blocks),
        "optimizer.name": optimizer_cfg.get("name"),
        "optimizer.lr": optimizer_cfg.get("lr"),
        "optimizer.weight_decay": optimizer_cfg.get("weight_decay"),
        "scheduler.name": scheduler_cfg.get("name"),
        "output_dir": str(output_dir),
    }


def run_training(args):
    cfg = _read_yaml(args.config)
    experiment_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("split", {})
    loader_cfg = data_cfg.get("loader", {})
    norm_cfg = data_cfg.get("normalization", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})

    seed = int(args.seed if args.seed is not None else experiment_cfg.get("seed", 42))
    epochs = int(args.epochs if args.epochs is not None else training_cfg.get("epochs", 30))
    amp = bool(training_cfg.get("amp", False))
    pos_label = int(training_cfg.get("pos_label", 1))
    task = str(data_cfg.get("task", "binary_classification")).lower()
    mode = "multiclass" if "multi" in task else "binary"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = output_dir / "best_model.pth"
    last_checkpoint_path = output_dir / "last_model.pth"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "best_classification_report.txt"

    device = _resolve_device(args.device)
    _set_seed(seed)

    train_loader, val_loader, class_names, num_classes, _ = get_oxford_pet_loaders(
        data_dir=data_cfg.get("data_dir", "./data"),
        batch_size=int(loader_cfg.get("batch_size", 32)),
        val_split=float(split_cfg.get("val_split", 0.2)),
        num_workers=int(loader_cfg.get("num_workers", 2)),
        seed=seed,
        mode=mode,
        img_size=int(data_cfg.get("img_size", 224)),
        robust=bool(norm_cfg.get("robust", False)),
        stats_cache_path=norm_cfg.get("stats_cache_path"),
        use_cached_if_available=bool(norm_cfg.get("use_cached_if_available", True)),
    )

    model = ResNet(
        num_classes=int(model_cfg.get("num_classes", num_classes)),
        first_block=str(model_cfg.get("first_block", "conv")),
        init=str(model_cfg.get("init", "kaiming")),
        blocks_per_stage=tuple(model_cfg.get("blocks_per_stage", [3, 4, 23, 3])),
    ).to(device)

    criterion = _build_criterion(training_cfg.get("criterion", {}))
    optimizer = _build_optimizer(model, training_cfg.get("optimizer", {}))
    scheduler = _build_scheduler(optimizer, training_cfg.get("scheduler", {}))
    multiclass = mode == "multiclass"

    mlflow = _import_mlflow()
    mlflow.set_tracking_uri(args.tracking_uri)
    experiment_name = args.experiment_name or str(
        experiment_cfg.get("name", "resnet101-mlflow")
    )
    mlflow.set_experiment(experiment_name)

    run_name = args.run_name or f"{experiment_name}-seed{seed}"
    history = []
    best = {
        "epoch": 0,
        "val_roc_auc": float("-inf"),
        "val_loss": float("inf"),
        "val_acc": 0.0,
        "report": "",
    }

    with mlflow.start_run(run_name=run_name) as run:
        run_params = _collect_run_params(cfg, seed, device, epochs, output_dir)
        mlflow.log_params(run_params)
        mlflow.set_tag("project", "deep-cnn-interpretability")
        mlflow.set_tag("component", "resnet101")
        mlflow.set_tag("class_names", ",".join(str(c) for c in class_names))

        for epoch in range(1, epochs + 1):
            train_metrics = train_epoch_classification(
                train_loader,
                model,
                optimizer,
                criterion,
                device=device,
                amp=amp,
                desc=f"Train [{epoch}/{epochs}]",
                scheduler=None,
                pos_label=pos_label,
            )
            val_metrics = evaluate_model(
                model,
                val_loader,
                criterion,
                device=device,
                multiclass=multiclass,
                plot=False,
            )

            if scheduler is not None:
                if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    scheduler.step(val_metrics["val_loss"])
                else:
                    scheduler.step()

            train_acc = float(train_metrics["acc"]) / 100.0
            train_f1 = float(train_metrics["f1"])
            lr = float(optimizer.param_groups[0]["lr"])

            epoch_summary = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": float(train_metrics["loss"]),
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_loss": float(val_metrics["val_loss"]),
                "val_roc_auc": float(val_metrics["roc_auc"]),
                "val_acc": float(val_metrics.get("val_acc", 0.0)),
                "val_macro_f1": float(val_metrics.get("macro_f1", 0.0)),
                "val_weighted_f1": float(val_metrics.get("weighted_f1", 0.0)),
            }
            history.append(epoch_summary)
            mlflow.log_metrics(
                {k: v for k, v in epoch_summary.items() if k != "epoch"},
                step=epoch,
            )

            is_best = epoch_summary["val_roc_auc"] > best["val_roc_auc"]
            if is_best:
                best.update(
                    {
                        "epoch": epoch,
                        "val_roc_auc": epoch_summary["val_roc_auc"],
                        "val_loss": epoch_summary["val_loss"],
                        "val_acc": epoch_summary["val_acc"],
                        "report": val_metrics["report"],
                    }
                )
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best["val_acc"],
                    path=str(best_checkpoint_path),
                )
                report_path.write_text(best["report"], encoding="utf-8")

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best["val_acc"],
                path=str(last_checkpoint_path),
            )

        summary = {
            "run_id": run.info.run_id,
            "experiment_name": experiment_name,
            "config_path": str(args.config),
            "best_epoch": best["epoch"],
            "best_val_roc_auc": best["val_roc_auc"],
            "best_val_loss": best["val_loss"],
            "best_val_acc": best["val_acc"],
            "history": history,
        }
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        mlflow.log_artifact(args.config, artifact_path="config")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        if report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="reports")
        if best_checkpoint_path.exists():
            mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")
        if last_checkpoint_path.exists():
            mlflow.log_artifact(str(last_checkpoint_path), artifact_path="checkpoints")

        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
        mlflow.pytorch.log_model(model, artifact_path="model")

        if args.register_model_name:
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                mlflow.register_model(model_uri=model_uri, name=args.register_model_name)
            except Exception as exc:
                print(f"[WARN] No se pudo registrar el modelo: {exc}")

        print(f"[OK] Run finalizado. Run ID: {run.info.run_id}")
        print(f"[OK] Mejor modelo: {best_checkpoint_path}")
        print(f"[OK] Metricas: {metrics_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenamiento ResNet101 con tracking profesional en MLflow."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="resnet101/oxford_pets_binary_resnet101.yaml",
        help="Ruta al YAML de configuracion de experimento.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="resnet101/model_trained/mlops",
        help="Directorio para checkpoints y metricas.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="file:./resnet101/mlruns",
        help="MLflow Tracking URI (ej: file:./resnet101/mlruns).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Nombre de experimento en MLflow (si no se pasa, usa el YAML).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nombre opcional del run de MLflow.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device explicito (ej: cuda, cpu, cuda:0).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override de epocas (si no se pasa, usa el YAML).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override de semilla.",
    )
    parser.add_argument(
        "--register-model-name",
        type=str,
        default=None,
        help="Nombre para registrar el modelo en MLflow Registry (opcional).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
