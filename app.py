import gradio as gr
from PIL import Image
# IMPORTANTE: Importamos la lógica local que arreglamos en logic.py
# En lugar de llamar a una API externa, usamos tu CPU local.
from mylib.logic import predict, resize 

# --- FUNCIÓN 1: CLASIFICAR CON ENLACE ---
def predict_and_link(image):
    """
    Usa el modelo local ONNX para predecir y genera un enlace.
    """
    if image is None:
        return "Por favor, sube una imagen.", ""
    
    try:
        # 1. Predicción LOCAL (Usando model.onnx)
        # La función predict de logic.py ya devuelve el nombre de la raza (str)
        raza_detectada = predict(image)
        
        # 2. Generamos el enlace de Google
        google_url = f"https://www.google.com/search?q={raza_detectada}+dog+breed"
        
        # HTML bonito para el resultado
        link_html = f"""
        <div style="text-align: center; padding: 10px; background-color: #f0f8ff; border-radius: 10px; border: 1px solid #add8e6;">
            <h3>🐕 Raza detectada: <b>{raza_detectada}</b></h3>
            <a href="{google_url}" target="_blank" style="font-size: 16px; color: #0066cc; text-decoration: none;">
                👉 <b>Haz clic aquí para saber más sobre los {raza_detectada}</b> 🌐
            </a>
        </div>
        """
        
        # Devolvemos: (Etiqueta de texto, HTML con enlace)
        return raza_detectada, link_html
            
    except Exception as e:
        return f"Error interno: {str(e)}", ""

# --- FUNCIÓN 2: REDIMENSIONAR ---
def resize_image_fn(image, width, height):
    """Usa la función local resize de logic.py"""
    if image is None:
        return None
    try:
        # Convertimos sliders a int
        return resize(image, int(width), int(height))
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- DISEÑO DE LA INTERFAZ (Tu diseño Soft original) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    # Encabezado
    gr.Markdown(
        """
        # 🐶 MLOps Lab 3 - On-Device Inference
        
        ¡Bienvenido! Esta aplicación corre **100% localmente** usando un modelo ONNX.
        No envía datos a ninguna API externa.
        """
    )
    
    with gr.Tabs():
        
        # --- PESTAÑA 1: CLASIFICADOR ---
        with gr.TabItem("🔍 Clasificador de Razas"):
            gr.Markdown("### Sube una foto de un perro y el modelo ONNX adivinará su raza.")
            
            with gr.Row():
                with gr.Column():
                    # IMPORTANTE: type="pil" porque logic.py espera una imagen PIL, no una ruta de archivo
                    input_cls = gr.Image(type="pil", label="📸 Imagen Original", height=300)
                    btn_cls = gr.Button("🧠 Analizar Raza", variant="primary")
                
                with gr.Column():
                    output_text = gr.Label(label="Resultado del Modelo")
                    output_link = gr.HTML(label="🔗 Información Extra")
            
            btn_cls.click(fn=predict_and_link, inputs=input_cls, outputs=[output_text, output_link])
            
        # --- PESTAÑA 2: REDIMENSIONAR ---
        with gr.TabItem("🖼️ Redimensionar (Local Tool)"):
            gr.Markdown("### Cambia el tamaño de tus imágenes localmente.")
            
            with gr.Row():
                with gr.Column():
                    input_res = gr.Image(type="pil", label="Imagen Original")
                    
                    with gr.Row():
                        width_slider = gr.Slider(32, 1024, value=224, step=32, label="Ancho (Width)")
                        height_slider = gr.Slider(32, 1024, value=224, step=32, label="Alto (Height)")
                    
                    btn_res = gr.Button("✂️ Redimensionar Ahora", variant="primary")
                
                with gr.Column():
                    output_res = gr.Image(label="Resultado")
            
            btn_res.click(
                fn=resize_image_fn, 
                inputs=[input_res, width_slider, height_slider], 
                outputs=output_res
            )

    # Pie de página
    gr.Markdown("---")
    gr.Markdown("MLOps Lab 3 - Ejecutando modelo MobileNetV2 ONNX")

# Lanzar la app
if __name__ == "__main__":
    demo.launch()