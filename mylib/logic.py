"""
Logic module for image processing: prediction, resizing, grayscale and flattening.
"""

import random
from PIL import Image


def predict(_image: Image.Image) -> str:
    """
    Predecir la clase de una imagen dada aleatoriamente.
    El argumento _image lleva guion bajo para indicar que no se usa.
    (Se usara en la proxima practica)
    """
    # Definimos un set de clases de ejemplo
    classes = ["gato", "perro", "coche", "avión"]
    return random.choice(classes)


def resize(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    Redimensionar una imagen a un ancho y alto específicos.
    """
    return image.resize((width, height))


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convierte la imagen a escala de grises.
    """
    return image.convert("L")


def flatten_image(image: Image.Image) -> list:
    """
    Convierte la imagen en una lista plana de píxeles (Reshape 2D -> 1D).
    Esto nos sirve para pasar la imagen a una red neuronal simple (Perceptrón).
    """
    return list(image.getdata())
