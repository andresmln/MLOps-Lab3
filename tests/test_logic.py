"""
Unit Testing of the application's logic

"""

import os
import json
import pytest
from PIL import Image
from mylib.logic import predict, resize, convert_to_grayscale, flatten_image

def get_valid_classes():
    """Carga las clases válidas desde el archivo JSON generado."""
    if os.path.exists("classes.json"):
        with open("classes.json", "r") as f:
            return json.load(f)
    return []
def test_predict():
    """Probar que predict devuelve una clase válida del modelo."""
    # 1. Creamos una imagen dummy
    # Usamos 224x224 que es lo que espera el modelo, aunque el resize interno lo arreglaría
    img = Image.new("RGB", (224, 224), color="green")

    # 2. Ejecutamos la predicción
    result = predict(img)

    # 3. Verificaciones
    assert isinstance(result, str)

    # Cargamos las clases reales del entrenamiento
    valid_classes = get_valid_classes()

    # Si tenemos clases (el json existe), verificamos que la predicción sea una de ellas
    if valid_classes:
        assert result in valid_classes, f"La predicción '{result}' no está en la lista de clases conocidas."
    else:
        # Si no hay json (en un entorno CI limpio sin entrenar), al menos que no dé error
        assert len(result) > 0

def test_resize():
    """Probar que resize cambia las dimensiones correctamente."""
    img = Image.new("RGB", (60, 30), color="red")
    target_width = 100
    target_height = 100

    new_img = resize(img, target_width, target_height)

    assert new_img.size == (target_width, target_height)


def test_convert_to_grayscale():
    """Probar que la imagen se convierte a escala de grises (Modo L)."""
    img = Image.new("RGB", (50, 50), color="blue")

    gray_img = convert_to_grayscale(img)

    # 'L' significa Luminance en PIL
    assert gray_img.mode == "L"

    # El tamaño debe mantenerse igual
    assert gray_img.size == (50, 50)


def test_flatten_image():
    """Probar que la imagen se aplana en una lista de píxeles."""
    width, height = 10, 10
    img = Image.new("RGB", (width, height), color="green")

    flat_data = flatten_image(img)

    # Debe ser una lista
    assert isinstance(flat_data, list)

    # La longitud debe ser igual al número total de píxeles (10 * 10 = 100)
    assert len(flat_data) == width * height

    # Verificamos que el contenido sean los datos del píxel (verde)
    assert flat_data[0] == (
        0,
        128,
        0,
    )  # or flat_data[0] == (0, 255, 0)    # El verde por defecto en PIL suele ser (0, 128, 0) o (0, 255, 0)
