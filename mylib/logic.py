"""
Logic Module for MLOps Lab3.
Contains the MobileNetV2 inference logic and image processing utilities.
"""
import os
import json
import numpy as np
import onnxruntime as rt

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

model_path = os.path.join(root_dir, "model.onnx")
classes_path = os.path.join(root_dir, "classes.json")

# --- CARGA DEL MODELO (Singleton) ---
session = None
input_name = None
classes = {}
load_error = "No iniciado"  # Variable para guardar el error exacto

def load_resources():
    """Carga el modelo y las clases si no están cargados."""
    # pylint: disable=global-statement
    global session, input_name, classes, load_error

    # 1. Cargar Modelo ONNX
    if session is None:
        try:
            if os.path.exists(model_path):
                session = rt.InferenceSession(model_path)
                input_name = session.get_inputs()[0].name
                print(f"✅ [Logic] Modelo cargado desde: {model_path}")
                load_error = None  # Limpiamos el error si funciona
            else:
                load_error = f"Archivo no encontrado en: {model_path}"
                print(f"⚠️ [Logic] {load_error}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            load_error = str(e)
            print(f"❌ [Logic] Error cargando modelo: {e}")

    # 2. Cargar Clases
    if not classes:
        try:
            if os.path.exists(classes_path):
                # Importante: encoding='utf-8' para contentar a Pylint
                with open(classes_path, "r", encoding="utf-8") as f:
                    classes = json.load(f)
                print(f"✅ [Logic] {len(classes)} clases cargadas.")
            else:
                print(f"⚠️ [Logic] No se encuentra classes.json en: {classes_path}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"❌ [Logic] Error cargando classes.json: {e}")

# Cargamos recursos al importar
load_resources()

def preprocess_image(image):
    """Preprocesamiento EXACTO para MobileNetV2 (Lab 3)."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((224, 224))
    img_data = np.array(image).astype('float32') / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    img_data = (img_data - mean) / std

    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)

    return img_data

def predict(image):
    """Predice la raza usando el modelo ONNX."""
    if session is None:
        return f"Error: Modelo no cargado. Causa: {load_error}"

    try:
        input_data = preprocess_image(image)
        output = session.run(None, {input_name: input_data})
        prediction_index = int(np.argmax(output[0]))

        # --- CORRECCIÓN AQUÍ: Soporte para Lista y Diccionario ---
        if isinstance(classes, list):
            # Si es lista (tu caso actual), accedemos por índice numérico
            if 0 <= prediction_index < len(classes):
                label = classes[prediction_index]
            else:
                label = "Desconocido"
        elif isinstance(classes, dict):
            # Si fuera diccionario, usamos .get()
            label = classes.get(str(prediction_index), classes.get(prediction_index, "Desconocido"))
        else:
            label = "Error: Formato de clases desconocido"
            
        return label
    except Exception as e:  # pylint: disable=broad-exception-caught
        return f"Error en predicción: {str(e)}"

def resize(image, width, height):
    """Redimensiona una imagen."""
    return image.resize((width, height))

# --- FUNCIONES AUXILIARES ---

def convert_to_grayscale(image):
    """Convierte imagen a blanco y negro (Modo L)."""
    return image.convert("L")

def flatten_image(image):
    """Aplana la imagen a una lista de píxeles."""
    return list(image.getdata())