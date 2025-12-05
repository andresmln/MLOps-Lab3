"""
Integration testing the model

"""

import os

def test_model_files_exist():
    """Verifica que los archivos críticos del modelo existen."""
    assert os.path.exists("model.onnx"), "¡ERROR! No se encuentra model.onnx"
    assert os.path.exists("classes.json"), "¡ERROR! No se encuentra classes.json"