from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field("ok", description="Estado general del servicio.")
    model: str = Field(..., description="Nombre del modelo cargado.")
    version: str = Field(..., description="Versión del modelo.")
    device: str = Field(..., description="Dispositivo en uso (cpu/cuda).")
    input_size: int = Field(..., description="Tamaño de entrada esperado.")
    classes: list[str] = Field(..., description="Clases soportadas por el modelo.")

    
