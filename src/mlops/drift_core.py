import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_reference_stats(reference_stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = Path(reference_stats_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de referencia: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    loc = np.asarray(payload.get("loc", []), dtype=np.float64)
    scale = np.asarray(payload.get("scale", []), dtype=np.float64)
    if loc.size != 3 or scale.size != 3:
        raise ValueError("El archivo de referencia debe tener 'loc' y 'scale' con 3 canales.")
    return loc, scale


def load_inference_events(log_path: str, window_size: int) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.exists():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    events = []
    for line in lines[-window_size:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _extract_matrix(events: list[dict[str, Any]], field: str) -> np.ndarray:
    rows = []
    for e in events:
        v = e.get(field)
        if isinstance(v, list) and len(v) == 3:
            try:
                rows.append([float(x) for x in v])
            except (TypeError, ValueError):
                continue
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _extract_confidence(events: list[dict[str, Any]]) -> np.ndarray:
    out = []
    for e in events:
        try:
            out.append(float(e.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue
    if not out:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _distribution(events: list[dict[str, Any]], field: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    for e in events:
        value = e.get(field)
        if isinstance(value, str) and value.strip():
            key = value.strip().lower()
            counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in sorted(counts.items())}


def _total_variation_distance(current: dict[str, float], expected: dict[str, float]) -> float | None:
    if not current or not expected:
        return None
    keys = set(current) | set(expected)
    return float(0.5 * sum(abs(current.get(k, 0.0) - expected.get(k, 0.0)) for k in keys))


def _feedback_accuracy(feedback_events: list[dict[str, Any]]) -> tuple[int, float | None]:
    vals: list[bool] = []
    for e in feedback_events:
        if isinstance(e.get("prediction_correct"), bool):
            vals.append(bool(e["prediction_correct"]))
            continue
        pred = e.get("prediction")
        true_label = e.get("true_label")
        if isinstance(pred, str) and isinstance(true_label, str):
            vals.append(pred.strip().lower() == true_label.strip().lower())
    if not vals:
        return 0, None
    return len(vals), float(sum(1 for v in vals if v) / len(vals))


def analyze_drift(
    *,
    reference_stats_path: str,
    inference_log_path: str,
    feedback_log_path: str | None = None,
    window_size: int = 300,
    min_samples: int = 50,
    mean_shift_threshold: float = 0.35,
    scale_shift_threshold: float = 0.25,
    min_avg_confidence: float = 0.60,
    prediction_shift_threshold: float = 0.30,
    label_shift_threshold: float = 0.30,
    min_feedback_samples: int = 20,
    min_feedback_accuracy: float = 0.80,
    expected_prediction_distribution: dict[str, float] | None = None,
    expected_label_distribution: dict[str, float] | None = None,
) -> Dict[str, Any]:
    ref_loc, ref_scale = load_reference_stats(reference_stats_path)
    events = load_inference_events(inference_log_path, window_size=window_size)
    feedback_events = load_inference_events(feedback_log_path, window_size=window_size) if feedback_log_path else []

    channel_means = _extract_matrix(events, "raw_channel_mean")
    channel_stds = _extract_matrix(events, "raw_channel_std")
    confidence = _extract_confidence(events)
    prediction_distribution = _distribution(events, "prediction")
    true_label_distribution = _distribution(feedback_events, "true_label")

    sample_count = int(min(len(channel_means), len(channel_stds)))
    if sample_count < min_samples:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "insufficient_data",
            "drift_detected": False,
            "samples": sample_count,
            "min_samples_required": min_samples,
            "window_size": window_size,
            "drift_types": {
                "data_drift": False,
                "model_drift": False,
                "problem_drift": False,
                "concept_drift": False,
            },
        }

    means_window = channel_means[:sample_count]
    stds_window = channel_stds[:sample_count]
    confidence_window = confidence[:sample_count] if confidence.size else np.empty((0,))

    curr_loc = means_window.mean(axis=0)
    curr_scale = stds_window.mean(axis=0)

    eps = 1e-8
    mean_shift_sigma = np.abs(curr_loc - ref_loc) / np.maximum(ref_scale, eps)
    scale_shift_ratio = np.abs(curr_scale / np.maximum(ref_scale, eps) - 1.0)
    avg_conf = float(confidence_window.mean()) if confidence_window.size else 0.0

    mean_flag = bool(np.any(mean_shift_sigma > mean_shift_threshold))
    scale_flag = bool(np.any(scale_shift_ratio > scale_shift_threshold))
    conf_flag = bool(avg_conf < min_avg_confidence) if confidence_window.size else False
    expected_pred_dist = expected_prediction_distribution or {"cat": 0.5, "dog": 0.5}
    expected_label_dist = expected_label_distribution or expected_pred_dist
    prediction_tvd = _total_variation_distance(prediction_distribution, expected_pred_dist)
    label_tvd = _total_variation_distance(true_label_distribution, expected_label_dist)
    feedback_count, feedback_acc = _feedback_accuracy(feedback_events)

    prediction_flag = prediction_tvd is not None and prediction_tvd > prediction_shift_threshold
    label_flag = label_tvd is not None and label_tvd > label_shift_threshold
    concept_flag = (
        feedback_acc is not None
        and feedback_count >= min_feedback_samples
        and feedback_acc < min_feedback_accuracy
    )

    data_drift = bool(mean_flag or scale_flag)
    model_drift = bool(conf_flag or prediction_flag)
    problem_drift = bool(label_flag)
    drift_detected = bool(data_drift or model_drift or problem_drift or concept_flag)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "drift_detected": drift_detected,
        "samples": sample_count,
        "window_size": window_size,
        "reference": {
            "loc": ref_loc.tolist(),
            "scale": ref_scale.tolist(),
        },
        "current": {
            "loc": curr_loc.tolist(),
            "scale": curr_scale.tolist(),
            "avg_confidence": avg_conf,
            "prediction_distribution": prediction_distribution,
            "true_label_distribution": true_label_distribution,
            "feedback_accuracy": feedback_acc,
            "feedback_samples": feedback_count,
        },
        "thresholds": {
            "mean_shift_sigma": mean_shift_threshold,
            "scale_shift_ratio": scale_shift_threshold,
            "min_avg_confidence": min_avg_confidence,
            "prediction_shift_tvd": prediction_shift_threshold,
            "label_shift_tvd": label_shift_threshold,
            "min_feedback_samples": min_feedback_samples,
            "min_feedback_accuracy": min_feedback_accuracy,
        },
        "signals": {
            "mean_shift_sigma": mean_shift_sigma.tolist(),
            "scale_shift_ratio": scale_shift_ratio.tolist(),
            "low_confidence": conf_flag,
            "prediction_distribution_tvd": prediction_tvd,
            "true_label_distribution_tvd": label_tvd,
            "low_feedback_accuracy": concept_flag,
        },
        "flags": {
            "mean_shift": mean_flag,
            "scale_shift": scale_flag,
            "confidence_shift": conf_flag,
            "prediction_distribution_shift": prediction_flag,
            "true_label_distribution_shift": label_flag,
            "feedback_performance_shift": concept_flag,
        },
        "drift_types": {
            "data_drift": data_drift,
            "model_drift": model_drift,
            "problem_drift": problem_drift,
            "concept_drift": concept_flag,
        },
    }

