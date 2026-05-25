import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable

import torch

def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _enabled() -> bool:
    return _as_bool(os.environ.get("ENABLE_INFERENCE_LOGGING", "true"))


def _log_path() -> Path:
    return Path(os.environ.get("INFERENCE_LOG_PATH", "monitoring/inference_events.jsonl"))


@lru_cache(maxsize=4)
def _cached_stats(stats_path: str | None):
    path = Path(stats_path or os.environ.get("PET_STATS_PATH", "data/pet_stats.json"))
    if not path.exists():
        raise FileNotFoundError(f"No existe archivo de stats: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))

    if "loc" in payload and "scale" in payload:
        return {"loc": payload["loc"], "scale": payload["scale"]}

    norm = payload.get("normalization", {})
    if isinstance(norm.get("mean"), list) and isinstance(norm.get("std"), list):
        return {"loc": norm["mean"], "scale": norm["std"]}

    raise ValueError("Stats invalidas: se esperaba loc/scale o normalization.mean/std")


def _unnormalize_tensor(x: torch.Tensor, stats_path: str | None = None) -> torch.Tensor:
    stats = _cached_stats(stats_path)
    loc = torch.tensor(stats["loc"], dtype=torch.float32).view(3, 1, 1)
    scale = torch.tensor(stats["scale"], dtype=torch.float32).view(3, 1, 1)

    t = x.detach().cpu().float()
    if t.ndim == 4:
        t = t[0]
    raw = (t * scale + loc).clamp(0.0, 1.0)
    return raw


def _safe_scores(scores: Dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "unknown", 0.0
    label, conf = max(scores.items(), key=lambda kv: kv[1])
    return str(label), float(conf)


def _to_float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def log_inference_event(
    *,
    tensor: torch.Tensor,
    scores: Dict[str, float],
    endpoint: str,
    source: str,
    model_version: str,
    device: str,
    stats_path: str | None = None,
    preprocess_ms: float | None = None,
    inference_ms: float | None = None,
    total_latency_ms: float | None = None,
    input_width: int | None = None,
    input_height: int | None = None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> None:
    raw = _unnormalize_tensor(tensor, stats_path=stats_path)
    means = raw.mean(dim=(1, 2)).tolist()
    stds = raw.std(dim=(1, 2), unbiased=False).tolist()
    pred_label, confidence = _safe_scores(scores)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "source": source,
        "prediction": pred_label,
        "confidence": confidence,
        "model_version": model_version,
        "device": device,
        "status": status,
        "preprocess_ms": float(preprocess_ms) if preprocess_ms is not None else None,
        "inference_ms": float(inference_ms) if inference_ms is not None else None,
        "total_latency_ms": float(total_latency_ms) if total_latency_ms is not None else None,
        "input_width": int(input_width) if input_width is not None else None,
        "input_height": int(input_height) if input_height is not None else None,
        "raw_channel_mean": _to_float_list(means),
        "raw_channel_std": _to_float_list(stds),
        "input_size": int(raw.shape[-1]),
    }
    if extra:
        record["extra"] = extra

    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def safe_log_inference_event(
    *,
    tensor: torch.Tensor,
    scores: Dict[str, float],
    endpoint: str,
    source: str,
    model_version: str,
    device: str,
    stats_path: str | None = None,
    preprocess_ms: float | None = None,
    inference_ms: float | None = None,
    total_latency_ms: float | None = None,
    input_width: int | None = None,
    input_height: int | None = None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> None:
    if not _enabled():
        return
    try:
        log_inference_event(
            tensor=tensor,
            scores=scores,
            endpoint=endpoint,
            source=source,
            model_version=model_version,
            device=device,
            stats_path=stats_path,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            total_latency_ms=total_latency_ms,
            input_width=input_width,
            input_height=input_height,
            status=status,
            extra=extra,
        )
    except Exception as exc:
        print(f"[monitor] No se pudo registrar evento de inferencia: {exc}")
