import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

# Rutas a los archivos generados
MODEL_PATH = "model.onnx"
CLASSES_PATH = "classes.json"

# Variables globales para cargar el modelo solo una vez (Singleton pattern)
session = None
classes = None

def load_resources():
    """Carga el modelo ONNX y las clases si no están cargados."""
    global session, classes
    
    if session is None:
        # Cargar lista de clases
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH, "r") as f:
                classes = json.load(f)
        else:
            # Fallback por si no existe el fichero aún
            classes = ["Clase Desconocida"]

        # Cargar sesión de ONNX Runtime (CPU)
        if os.path.exists(MODEL_PATH):
            # Optimización básica para CPU
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            session = ort.InferenceSession(
                MODEL_PATH, 
                sess_options, 
                providers=["CPUExecutionProvider"]
            )

def preprocess_image(image: Image.Image):
    """
    Preprocesa la imagen igual que se hizo en el entrenamiento (MobileNetV2).
    1. Resize 224x224
    2. Normalización (Mean/Std de ImageNet)
    3. Transposición (HWC -> CHW)
    4. Batch dimension
    """
    # 1. Resize
    image = image.resize((224, 224)).convert("RGB")
    
    # 2. Convertir a Numpy y Normalizar (0-1)
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalización estándar de ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # 3. Transponer de (Alto, Ancho, Canales) a (Canales, Alto, Ancho)
    img_array = img_array.transpose(2, 0, 1)
    
    # 4. Añadir dimensión de Batch: (1, 3, 224, 224)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(image: Image.Image) -> str:
    """
    Realiza la inferencia usando el modelo ONNX.
    """
    try:
        # Aseguramos que los recursos estén cargados
        load_resources()
        
        if session is None:
            return "Error: Modelo no encontrado (model.onnx)"

        # Preprocesar
        input_tensor = preprocess_image(image)
        
        # Obtener el nombre de la entrada del modelo
        input_name = session.get_inputs()[0].name
        
        # Ejecutar inferencia
        outputs = session.run(None, {input_name: input_tensor})
        
        # Obtener la clase con mayor probabilidad (Argmax)
        logits = outputs[0][0]
        predicted_idx = np.argmax(logits)
        
        # Devolver la etiqueta correspondiente
        if classes and predicted_idx < len(classes):
            return classes[predicted_idx]
        else:
            return f"Clase {predicted_idx}"
            
    except Exception as e:
        return f"Error en predicción: {str(e)}"


def resize(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height))

def convert_to_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")

def flatten_image(image: Image.Image) -> list:
    return list(image.getdata())