"""
API Module for MLOps Lab1.
Provides endpoints for image prediction and processing.
"""

import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from mylib.logic import predict, resize

app = FastAPI(
    title="MLOps Lab03 API",
    description="API to classify and resize images for MLOps Lab3",
    version="1.0.1",
)

# We use the templates folder to obtain HTML files
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Endpoint principal que sirve la página de inicio.
    """
    return templates.TemplateResponse(request, "home.html")


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Endpoint para predecir la clase de una imagen subida.
    Recibe un archivo binario (UploadFile).
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    prediction = predict(image)
    return {"filename": file.filename, "prediction": prediction}


@app.post("/resize")
async def resize_endpoint(
    file: UploadFile = File(...), width: int = Form(...), height: int = Form(...)
):
    """
    Endpoint para redimensionar una imagen.
    Recibe archivo y parámetros width/height via Form.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        resized_image = resize(image, width, height)
        return {
            "original_size": image.size,
            "new_size": resized_image.size,
            "message": "Image resized successfully",
        }
    except Exception as e:
        # Usamos 'from e' para mantener la traza del error original (pylint W0707)
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
