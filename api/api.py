"""
API Module for MLOps Lab3.
Direct implementation of MobileNetV2 inference.
"""

import io
import json
import os
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

# --- GESTIÓN DE RUTAS (PATH) ---
# 1. Obtenemos la ruta api.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Subimos un nivel para llegar a la raíz (MLOps-Lab3/)
root_dir = os.path.dirname(current_dir)

# 3. Definimos las rutas absolutas a los archivos
model_path = os.path.join(root_dir, "model.onnx")
classes_path = os.path.join(root_dir, "classes.json")
templates_dir = os.path.join(root_dir, "templates")

# --- CARGA DEL MODELO ---
session = None
try:
    # Usamos la ruta absoluta calculada
    session = rt.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    print(f"✅ Modelo cargado desde: {model_path}")

except Exception as e:
    print(f"❌ Error cargando modelo en {model_path}: {e}")

# --- CARGA DE CLASES ---
classes = {}
try:
    with open(classes_path, "r") as f:
        classes = json.load(f)
    print(f"✅ Clases cargadas: {len(classes)} encontradas.")
except Exception as e:
    print(f"❌ Error cargando classes.json en {classes_path}: {e}")

# --- INICIO DE LA APP ---
app = FastAPI(
    title="MLOps Lab03 API",
    description="API to classify pet breeds using MobileNetV2",
    version="1.0.1",
)

# Configurar templates
if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None

def preprocess_image(image):
    """
    Same transformation that we use in train
    """
    # 1. Convertir a RGB y redimensionar
    image = image.convert("RGB")
    image = image.resize((224, 224))
    
    # 2. Convertir a Numpy y normalizar (0-1)
    img_data = np.array(image).astype('float32') / 255.0
    
    # 3. Normalización ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    img_data = (img_data - mean) / std
    
    # 4. Transponer y Batch
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if templates:
        return templates.TemplateResponse(request, "home.html")
    return HTMLResponse(content="<h1>Bienvenido a la API de Mascotas (Lab 3)</h1>")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Prediction of the breed of the class
    """
    if session is None:
        raise HTTPException(status_code=500, detail="El modelo no está disponible.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # 1. Preprocesar
        input_data = preprocess_image(image)
        
        # 2. Inferencia
        output = session.run(None, {input_name: input_data})
        
        # 3. Interpretar resultado
        prediction_index = np.argmax(output[0])
        # Convertimos el índice a string porque las claves del JSON son strings "0", "1"...
        prediction_label = classes.get(str(prediction_index), "Desconocido")
        
        return {"filename": file.filename, "prediction": prediction_label}
        
    except Exception as e:
        return {"error": f"Fallo en predicción: {str(e)}"}

@app.post("/resize")
async def resize_endpoint(file: UploadFile = File(...), width: int = Form(...), height: int = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    resized_image = image.resize((width, height))
    return {
        "original_size": image.size,
        "new_size": resized_image.size,
        "message": "Image resized successfully",
    }

if __name__ == "__main__":
    
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)