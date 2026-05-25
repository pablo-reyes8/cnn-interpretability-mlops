import base64
import io
import time
from typing import Dict, Optional, List

import numpy as np
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from src.api.deps import (
    get_model,
    get_id_to_label,
    get_model_version,
    get_device,)

from src.inference.pipeline import (
    prepare_from_bytes,
    prepare_from_url,)

from src.inference.validate import InvalidImageError
from src.mlops.inference_monitor import safe_log_inference_event
from src.utils.config import get_config

from advance_visualization.graphs_advance import (
    kernels_depth_matrix,
    feature_maps_depth_matrix,
    gradcam_grid_panel_using_your_fn,
    integrated_gradients_overlay,
    occlusion_sensitivity_overlay,)

router = APIRouter(tags=["predict"])



def _validate_inputs(file: Optional[UploadFile], url: Optional[str]) -> None:
    if (file is None and not url) or (file is not None and url):
        raise HTTPException(
            status_code=400,
            detail="Debes enviar exactamente uno: 'file' (multipart) o 'url'.",)


def _validate_mime(file: UploadFile) -> None:
    cfg = get_config()
    if (file.content_type or "").lower() not in cfg.ALLOWED_MIMES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo no soportado: {file.content_type}. Solo JPEG/PNG.",)
    

def _np_to_png_b64(arr: np.ndarray) -> str:
    """
    Espera arr en HxWx3 uint8 (RGB). Devuelve 'data:image/png;base64,...'
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


async def _prep_input_from_request(file: Optional[UploadFile], url: Optional[str]):
    _validate_inputs(file, url)
    cfg = get_config()

    try:
        if file is not None:
            _validate_mime(file)
            raw = await file.read()  # type: ignore  # solo será llamado en endpoints async
            if len(raw) > cfg.MAX_IMAGE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"Archivo excede {cfg.MAX_IMAGE_MB} MB.",
                )
            return prepare_from_bytes(raw)
        else:
            return prepare_from_url(url)
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Timeout al descargar la imagen.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {e}") from e




@router.post(
    "/predict",
    summary="Clasifica una imagen como 'cat' o 'dog'",
    description=(
        "Envía **exactamente uno** de:\n\n"
        "- `file` (multipart/form-data) con imagen JPEG/PNG\n"
        "- `url` (form/query) con un http/https a imagen pública\n\n"
        "Preprocesamiento: resize(1.14×) → center-crop(224) → normalize(mean/std)."
    ),)

async def predict(
    file: Optional[UploadFile] = File(default=None, description="Imagen JPEG/PNG"),
    url: Optional[str] = Form(default=None, description="URL http/https a una imagen")):

    request_t0 = time.perf_counter()
    out = await _prep_input_from_request(file, url)
    x = out["tensor"]
    meta = dict(out.get("meta", {}))

    model = get_model()
    id_to_label = get_id_to_label()

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    t1 = time.perf_counter()
    inference_ms = round((t1 - t0) * 1000, 2)
    total_latency_ms = round((time.perf_counter() - request_t0) * 1000, 2)

    scores: Dict[str, float] = {id_to_label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    label = max(scores.items(), key=lambda kv: kv[1])[0]
    source = "file" if file is not None else "url"

    safe_log_inference_event(
        tensor=x,
        scores=scores,
        endpoint="/predict",
        source=source,
        model_version=get_model_version(),
        device=get_device(),
        preprocess_ms=meta.get("preprocess_ms"),
        inference_ms=inference_ms,
        total_latency_ms=total_latency_ms,
        input_width=meta.get("width"),
        input_height=meta.get("height"),
    )

    meta.update(
        {
            "inference_ms": inference_ms,
            "total_latency_ms": total_latency_ms,
            "model_version": get_model_version(),
            "device": get_device(),})
    return {"label": label, "scores": scores, "meta": meta}




@router.post(
    "/predict/advanced",
    summary="Predicción + artefactos de interpretabilidad",
    description=(
        "Entrega la predicción **y** paneles de interpretabilidad: kernels, feature maps, Grad-CAM, "
        "Integrated Gradients y Occlusion. Los paneles se devuelven como PNG en base64.\n\n"
        "**Parámetro opcional** `what` (form): `all` (default) o lista separada por comas con "
        "`kernels,feature_maps,gradcam,integrated_gradients,occlusion`."
    ),
)


async def predict_advanced(
    file: Optional[UploadFile] = File(default=None, description="Imagen JPEG/PNG"),
    url: Optional[str] = Form(default=None, description="URL http/https a una imagen"),
    what: str = Form(default="all", description="Qué artefactos retornar (all | lista separada por comas)")):

    request_t0 = time.perf_counter()
    out = await _prep_input_from_request(file, url)
    x = out["tensor"]
    meta = dict(out.get("meta", {}))

    model = get_model()
    id_to_label = get_id_to_label()

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    t1 = time.perf_counter()
    inference_ms = round((t1 - t0) * 1000, 2)

    scores: Dict[str, float] = {id_to_label.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    label = max(scores.items(), key=lambda kv: kv[1])[0]
    source = "file" if file is not None else "url"

    requested: List[str] = (
        ["kernels", "feature_maps", "gradcam", "integrated_gradients", "occlusion"]
        if what.strip().lower() == "all"
        else [w.strip().lower() for w in what.split(",") if w.strip()])

    artifacts: Dict[str, str] = {} 

    if "kernels" in requested:
        try:
            panel = kernels_depth_matrix(
                model,
                cols=12,
                tile_px=150,
                col_gap=20,
                pad_out_x=40,
                pad_out_y=20,
                pad_row=12,
            )  # np.uint8 HxWx3
            artifacts["kernels_panel"] = _np_to_png_b64(panel)
        except Exception as e:
            artifacts["kernels_panel_error"] = f"{e}"

    if "feature_maps" in requested:
        try:
            panel_fm = feature_maps_depth_matrix(
                model,
                x,
                cols=12,
                tile_px=140,
                col_gap=16,
                row_title_px=20,
            )
            artifacts["feature_maps_panel"] = _np_to_png_b64(panel_fm)
        except Exception as e:
            artifacts["feature_maps_panel_error"] = f"{e}"

    if "gradcam" in requested:
        try:
            panel_3x5 = gradcam_grid_panel_using_your_fn(
                model,
                x,
                ncols=5,
                tile_px=256,
                title_px=22,
                alpha=0.42,
                use_recolor=True,
                cmap_mode="magma",
                show_layer_path=False,
            )
            artifacts["gradcam_panel"] = _np_to_png_b64(panel_3x5)
        except Exception as e:
            artifacts["gradcam_panel_error"] = f"{e}"

    if "integrated_gradients" in requested:
        try:
            overlay = integrated_gradients_overlay(
                model,
                x,
                steps=32,
                smooth_samples=8,
                smooth_sigma=0.01,
                baseline="blurred",
                alpha=0.50,
                cmap_mode="magma",
                percentile_clip=99.5,
                border_suppress=0.02,)
            
            artifacts["integrated_gradients_overlay"] = _np_to_png_b64(overlay)
        except Exception as e:
            artifacts["integrated_gradients_error"] = f"{e}"

    if "occlusion" in requested:
        try:
            overlay, heat, cls_id, p0 = occlusion_sensitivity_overlay(
                model,
                x,
                patch=32,
                stride=12,
                baseline="mean",
                batch_size=64,
                alpha=0.45,
                cmap_mode="magma",
                agg="prob_drop",)
            
            artifacts["occlusion_overlay"] = _np_to_png_b64(overlay)
        except Exception as e:
            artifacts["occlusion_error"] = f"{e}"


    meta.update(
        {
            "inference_ms": inference_ms,
            "total_latency_ms": round((time.perf_counter() - request_t0) * 1000, 2),
            "model_version": get_model_version(),
            "device": get_device(),
            "requested": requested,})

    safe_log_inference_event(
        tensor=x,
        scores=scores,
        endpoint="/predict/advanced",
        source=source,
        model_version=get_model_version(),
        device=get_device(),
        preprocess_ms=meta.get("preprocess_ms"),
        inference_ms=inference_ms,
        total_latency_ms=meta.get("total_latency_ms"),
        input_width=meta.get("width"),
        input_height=meta.get("height"),
        extra={"requested_interpretability": requested},
    )

    return {
        "label": label,
        "scores": scores,
        "meta": meta,
        "artifacts": artifacts,}



