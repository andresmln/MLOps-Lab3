"""
Integration testing with the API

"""

import io
import json
import pytest
from api.api import app
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    """Testing client from FastAPI."""
    return TestClient(app)


def get_test_image_bytes():
    """Función auxiliar para crear una imagen en memoria (bytes)."""
    img = Image.new("RGB", (100, 100), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)
    return img_byte_arr


# def test_home_endpoint(client):
#     """Verify that the endpoint / returns the right message."""
#     response = client.get("/")
#     assert response.status_code == 200


def test_home(client):
    """Verificar que el home carga correctamente."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_endpoint(client):
    """Verificar el endpoint /predict enviando una imagen."""
    img_bytes = get_test_image_bytes()

    response = client.post(
        "/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "filename" in data


def test_resize_endpoint(client):
    """Verificar el endpoint /resize enviando imagen + parámetros."""
    img_bytes = get_test_image_bytes()

    response = client.post(
        "/resize",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"width": 50, "height": 50},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Image resized successfully"
    assert data["new_size"] == [50, 50]
