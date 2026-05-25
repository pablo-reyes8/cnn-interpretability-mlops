import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from resnet101.src.data.load_data import get_oxford_pet_loaders


def _read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _round_list(values: list[float], ndigits: int = 6) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def build_preprocess_report(config_path: str, report_path: str) -> dict[str, Any]:
    cfg = _read_yaml(config_path)
    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("split", {})
    loader_cfg = data_cfg.get("loader", {})
    norm_cfg = data_cfg.get("normalization", {})
    experiment_cfg = cfg.get("experiment", {})

    img_size = int(data_cfg.get("img_size", 224))
    batch_size = int(loader_cfg.get("batch_size", 32))
    num_workers = int(loader_cfg.get("num_workers", 2))
    seed = int(experiment_cfg.get("seed", 42))
    val_split = float(split_cfg.get("val_split", 0.2))
    task = str(data_cfg.get("task", "binary_classification")).lower()
    mode = "multiclass" if "multi" in task else "binary"

    train_loader, val_loader, class_names, num_classes, stats = get_oxford_pet_loaders(
        data_dir=data_cfg.get("data_dir", "./data"),
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
        mode=mode,
        img_size=img_size,
        robust=bool(norm_cfg.get("robust", False)),
        stats_cache_path=norm_cfg.get("stats_cache_path"),
        use_cached_if_available=bool(norm_cfg.get("use_cached_if_available", True)),
    )
    loc, scale = stats

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    expected_classes = list(data_cfg.get("class_names", ["cat", "dog"]))
    classes_ok = list(class_names) == expected_classes if mode == "binary" else len(class_names) == num_classes
    stats_ok = len(loc) == 3 and len(scale) == 3 and all(float(x) > 0 for x in scale)

    checks = {
        "dataset_available": train_size > 0 and val_size > 0,
        "non_empty_train_split": train_size > 0,
        "non_empty_validation_split": val_size > 0,
        "expected_classes_present": bool(classes_ok),
        "normalization_has_three_channels": bool(stats_ok),
        "model_input_size_matches_config": img_size == 224,
    }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "dataset": data_cfg.get("dataset", "OxfordIIITPet"),
        "task": task,
        "mode": mode,
        "data_dir": data_cfg.get("data_dir", "./data"),
        "seed": seed,
        "splits": {
            "strategy": "seeded_random_split",
            "val_split": val_split,
            "train_samples": train_size,
            "validation_samples": val_size,
            "train_batches": len(train_loader),
            "validation_batches": len(val_loader),
        },
        "classes": {
            "class_names": list(class_names),
            "num_classes": int(num_classes),
            "positive_class": "dog" if "dog" in class_names else None,
        },
        "preprocessing": {
            "img_size": img_size,
            "train": [
                f"Resize({int(img_size * 1.14)})",
                "RandomResizedCrop",
                "RandomHorizontalFlip",
                "ColorJitter",
                "ToTensor",
                "Normalize",
            ],
            "validation": [
                f"Resize({int(img_size * 1.14)})",
                "CenterCrop",
                "ToTensor",
                "Normalize",
            ],
            "normalization": {
                "stats_cache_path": norm_cfg.get("stats_cache_path"),
                "robust": bool(norm_cfg.get("robust", False)),
                "loc": _round_list(loc),
                "scale": _round_list(scale),
            },
        },
        "quality_checks": checks,
        "status": "passed" if all(checks.values()) else "failed",
    }

    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataOps: materializa y valida el preprocesamiento de entrenamiento."
    )
    parser.add_argument("--config-path", type=str, default="resnet101/oxford_pets_binary_resnet101.yaml")
    parser.add_argument("--report-path", type=str, default="monitoring/dataops_preprocess_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_preprocess_report(args.config_path, args.report_path)
    print(f"[OK] DataOps preprocess report: {args.report_path}")
    print(f"[OK] Status: {report.get('status')}")


if __name__ == "__main__":
    main()
