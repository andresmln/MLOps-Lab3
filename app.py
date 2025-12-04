import gradio as gr
import requests

URL_API = "https://mlops-lab2-api-ayg6.onrender.com"

def predict(image_path):
    """Función que envía la imagen a Render y recibe la predicción"""
    try:
        # Preparamos la imagen para enviarla como archivo
        files = {'file': open(image_path, 'rb')}
        
        # Hacemos la petición POST a tu API en la nube
        response = requests.post(f"{URL_API}/predict", files=files)
        
        if response.status_code == 200:
            # Si todo va bien, devolvemos la etiqueta (JSON)
            return response.json()
        else:
            return f"Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Error de conexión: {str(e)}"

# Definimos la interfaz visual con Gradio
iface = gr.Interface(
    fn=predict,                          # Función a ejecutar
    inputs=gr.Image(type="filepath"),    # Input: Subir imagen
    outputs="text",                      # Output: Texto con la clase
    title="Classifier of an Image (Lab 2)",
    description="Sube una imagen para clasificarla usando la API desplegada en Render."
)

# Lanzamos la app
iface.launch()