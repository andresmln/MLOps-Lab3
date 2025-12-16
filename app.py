import gradio as gr
import requests
from PIL import Image
import io

# URL de tu API
URL_API = "https://mlops-lab2-api-ayg6.onrender.com"

# --- FUNCIÓN 1: CLASIFICAR ---
def predict(image_path):
    """Envía la imagen a /predict y devuelve el JSON"""
    if image_path is None:
        return "Por favor, sube una imagen."
    
    try:
        files = {'file': open(image_path, 'rb')}
        response = requests.post(f"{URL_API}/predict", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error de conexión: {str(e)}"

# --- FUNCIÓN 2: REDIMENSIONAR (La que faltaba) ---
def resize_image(image_path, width, height):
    """Envía la imagen a /resize (o como se llame tu endpoint) y devuelve una IMAGEN"""
    if image_path is None:
        return None
        
    try:
        # Preparamos los datos
        files = {'file': open(image_path, 'rb')}
        params = {'width': int(width), 'height': int(height)} # Aseguramos que sean enteros
        
        # IMPORTANTE: Asegúrate de que tu API tiene este endpoint '/resize'
        # Si tu API espera los params en la url, usa params=params. Si es en el body, data=params
        response = requests.post(f"{URL_API}/resize", files=files, params=params)
        
        if response.status_code == 200:
            # Convertimos los bytes recibidos en una imagen visible
            return Image.open(io.BytesIO(response.content))
        else:
            print(f"Error API: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- INTERFAZ VISUAL CON PESTAÑAS ---
with gr.Blocks() as demo:
    gr.Markdown("# MLOps Lab 2 - API Interface")
    
    with gr.Tabs():
        # PESTAÑA 1: CLASIFICADOR
        with gr.TabItem("Clasificador de Razas"):
            with gr.Row():
                input_cls = gr.Image(type="filepath", label="Imagen Original")
                output_cls = gr.Text(label="Predicción")
            btn_cls = gr.Button("Clasificar")
            btn_cls.click(fn=predict, inputs=input_cls, outputs=output_cls)
            
        # PESTAÑA 2: REDIMENSIONAR (La de tu captura)
        with gr.TabItem("Redimensionar (Resize Tool)"):
            with gr.Row():
                input_res = gr.Image(type="filepath", label="Imagen Original")
                output_res = gr.Image(label="Imagen Redimensionada") # Salida es IMAGEN, no texto
            
            # Sliders
            width_slider = gr.Slider(minimum=32, maximum=1024, value=416, step=32, label="Nuevo Ancho")
            height_slider = gr.Slider(minimum=32, maximum=1024, value=416, step=32, label="Nuevo Alto")
            
            btn_res = gr.Button("Redimensionar")
            
            # Conexión del botón
            btn_res.click(
                fn=resize_image, 
                inputs=[input_res, width_slider, height_slider], 
                outputs=output_res
            )

# Lanzar la app
demo.launch()