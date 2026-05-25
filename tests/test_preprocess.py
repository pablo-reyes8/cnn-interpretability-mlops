import io
from PIL import Image
import torch
from src.inference.pipeline import prepare_from_bytes

def test_prepare_from_bytes_tensor_shape_ok():
    img = Image.new("RGB", (128, 128), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    out = prepare_from_bytes(buf.getvalue())
    x = out["tensor"]
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 1 and x.shape[1] == 3
    assert x.shape[-1] == 224 and x.shape[-2] == 224


def test_prepare_from_bytes_rejects_tiny_image():
    img = Image.new("RGB", (20, 20), (200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    import pytest
    from src.inference.validate import InvalidImageError

    with pytest.raises(InvalidImageError):
        prepare_from_bytes(buf.getvalue())


def test_prepare_from_bytes_accepts_grayscale_and_converts_to_rgb():
    img = Image.new("L", (128, 128), 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    out = prepare_from_bytes(buf.getvalue())
    assert out["tensor"].shape == (1, 3, 224, 224)
    assert out["meta"]["width"] == 128
    assert out["meta"]["height"] == 128
