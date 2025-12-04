"""
Unit Testing of the application's logic

"""

from PIL import Image
from mylib.logic import predict, resize, convert_to_grayscale, flatten_image


def test_predict():
    """Probar que predict devuelve una cadena válida."""
    # Creamos una imagen vacía pequeña para probar
    img = Image.new("RGB", (60, 30), color="red")
    result = predict(img)

    # Verificamos que el resultado sea una de las clases esperadas
    assert isinstance(result, str)
    assert result in ["gato", "perro", "coche", "avión"]


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
