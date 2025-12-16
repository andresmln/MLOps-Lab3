import gradio as gr
import requests
from PIL import Image
import io

# URL de tu API
URL_API = "https://mlops-lab2-api-ayg6.onrender.com"

# --- FUNCIÓN 1: CLASIFICAR CON ENLACE ---
def predict_and_link(image_path):
    """
    Envía la imagen a la API, recibe la predicción y genera un enlace 
    de búsqueda en Google para la raza detectada.
    """
    if image_path is None:
        return "Por favor, sube una imagen.", ""
    
    try:
        files = {'file': open(image_path, 'rb')}
        response = requests.post(f"{URL_API}/predict", files=files)
        
        if response.status_code == 200:
            prediction = response.json()
            
            # Lógica para encontrar el nombre de la raza principal
            # Asumimos que la API devuelve algo tipo {"Pug": 0.98, "Beagle": 0.02} o directamente "Pug"
            raza_detectada = "Desconocida"
            
            if isinstance(prediction, dict):
                # Buscamos la clave con el valor más alto (la raza más probable)
                raza_detectada = max(prediction, key=prediction.get)
            else:
                raza_detectada = str(prediction)
            
            # Generamos el enlace de Google
            google_url = f"https://www.google.com/search?q={raza_detectada}+dog+breed"
            link_html = f"""
            <div style="text-align: center; padding: 10px; background-color: #f0f8ff; border-radius: 10px; border: 1px solid #add8e6;">
                <h3>🐕 Raza detectada: <b>{raza_detectada}</b></h3>
                <a href="{google_url}" target="_blank" style="font-size: 16px; color: #0066cc; text-decoration: none;">
                    👉 <b>Haz clic aquí para saber más sobre los {raza_detectada}</b> 🌐
                </a>
            </div>
            """
            
            return prediction, link_html
            
        else:
            return f"Error {response.status_code}: {response.text}", ""
            
    except Exception as e:
        return f"Error de conexión: {str(e)}", ""

# --- FUNCIÓN 2: REDIMENSIONAR ---
def resize_image(image_path, width, height):
    """Envía la imagen a /resize y devuelve una IMAGEN"""
    if image_path is None:
        return None
        
    try:
        files = {'file': open(image_path, 'rb')}
        params = {'width': int(width), 'height': int(height)}
        
        response = requests.post(f"{URL_API}/resize", files=files, params=params)
        
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            print(f"Error API: {response.text}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- DISEÑO DE LA INTERFAZ ---
# Usamos un tema 'Soft' para que se vea más moderno y limpio
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    # Encabezado bonito
    gr.Markdown(
        """
        # 🐶 MLOps Lab 3 - API Interface
        
        ¡Bienvenido! Esta aplicación interactúa con una API de Machine Learning desplegada en la nube.
        Utiliza el menú de abajo para elegir entre **clasificar** una mascota o **redimensionar** una imagen.
        """
    )
    
    with gr.Tabs():
        
        # --- PESTAÑA 1: CLASIFICADOR ---
        with gr.TabItem("🔍 Clasificador de Razas"):
            gr.Markdown("### Sube una foto de un perro y la IA adivinará su raza.")
            
            with gr.Row():
                with gr.Column():
                    input_cls = gr.Image(type="filepath", label="📸 Imagen Original", height=300)
                    btn_cls = gr.Button("🧠 Analizar Raza", variant="primary")
                
                with gr.Column():
                    # Aquí mostramos el JSON crudo
                    output_text = gr.Label(label="📊 Probabilidades / Resultado")
                    # Y aquí el enlace bonito
                    output_link = gr.HTML(label="🔗 Información Extra")
            
            # Conectamos: 1 input -> 2 outputs (Texto y Enlace HTML)
            btn_cls.click(fn=predict_and_link, inputs=input_cls, outputs=[output_text, output_link])
            
        # --- PESTAÑA 2: REDIMENSIONAR ---
        with gr.TabItem("🖼️ Redimensionar (Resize Tool)"):
            gr.Markdown("### Cambia el tamaño de tus imágenes usando la API.")
            
            with gr.Row():
                with gr.Column():
                    input_res = gr.Image(type="filepath", label="Imagen Original")
                    
                    with gr.Row():
                        width_slider = gr.Slider(32, 1024, value=416, step=32, label="Ancho (Width)")
                        height_slider = gr.Slider(32, 1024, value=416, step=32, label="Alto (Height)")
                    
                    btn_res = gr.Button("✂️ Redimensionar Ahora", variant="primary")
                
                with gr.Column():
                    output_res = gr.Image(label="Resultado")
            
            btn_res.click(
                fn=resize_image, 
                inputs=[input_res, width_slider, height_slider], 
                outputs=output_res
            )

    # Pie de página
    gr.Markdown("---")
    gr.Markdown("Example developed for MLOps Master - 2025")

# Lanzar la app
demo.launch()