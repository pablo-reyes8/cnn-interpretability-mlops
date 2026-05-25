from fastapi import APIRouter
from src.api.deps import get_device, get_id_to_label, get_model_version
from src.schemas.health import HealthResponse

router = APIRouter(tags=["health"])

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    description="Devuelve el estado del API y metadatos del modelo cargado.")

def health():
    return {
        "status": "ok",
        "model": "ResNet101",
        "version": get_model_version(),
        "device": get_device(),
        "input_size": 224,
        "classes": list(get_id_to_label().values()),}
