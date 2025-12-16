"""
API Module for MLOps Lab3.
Direct implementation of MobileNetV2 inference.
"""

import io
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

# Importamos de tu librería local
from mylib.logic import predict, resize

app = FastAPI(
    title="MLOps Lab03 API",
    description="API to classify pet breeds using MobileNetV2",
    version="1.0.3",
)

# Configuración de templates segura
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
templates_dir = os.path.join(root_dir, "templates")

if os.path.exists(templates_dir):
    templates = Jinja2Templates(directory=templates_dir)
else:
    templates = None

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Endpoint principal que sirve la página de inicio."""
    if templates:
        return templates.TemplateResponse(request, "home.html")
    return HTMLResponse(content="<h1>Bienvenido a la API de Mascotas (Lab 3)</h1>")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Endpoint para predecir la raza de una mascota."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Delegamos la lógica a mylib/logic.py
    prediction = predict(image)

    return {"filename": file.filename, "prediction": prediction}

@app.post("/resize")
async def resize_endpoint(
    file: UploadFile = File(...),
    width: int = Form(...),
    height: int = Form(...)
):
    """Endpoint para redimensionar una imagen y devolverla."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Procesamos la imagen
    resized_image = resize(image, width, height)

    # Guardamos la imagen procesada en un buffer de memoria
    img_byte_arr = io.BytesIO()
    # Guardamos como JPEG (puedes cambiar a PNG si prefieres)
    resized_image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0) # Rebobinamos al principio del archivo

    # Devolvemos la imagen directamente
    return StreamingResponse(img_byte_arr, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
    