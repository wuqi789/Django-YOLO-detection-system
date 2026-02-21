import os
import sys
import threading
import logging
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vggt'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_gradio_server = None
_gradio_port = 8080
_gradio_lock = threading.Lock()
_gradio_running = False
_current_glb_path = None

def create_gradio_app(target_dir, glb_path):
    import gradio as gr
    import numpy as np
    
    global _current_glb_path
    _current_glb_path = glb_path
    
    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )
    
    with gr.Blocks(
        theme=theme,
        css="""
        .custom-log * {
            font-style: italic;
            font-size: 18px !important;
            background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            font-weight: bold !important;
            color: transparent !important;
            text-align: center !important;
        }
        """,
    ) as demo:
        gr.HTML("""
        <h1 style="text-align: center;">3D Reconstruction Viewer</h1>
        <p style="text-align: center;">VGGT 3D Model Visualization</p>
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("**3D Reconstruction Result**")
                log_output = gr.Markdown(
                    "Loading 3D model...", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=500, zoom_speed=0.5, pan_speed=0.5)
                
                with gr.Row():
                    conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                
                with gr.Row():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
        
        def load_model():
            if _current_glb_path and os.path.exists(_current_glb_path):
                logger.info(f"Loading GLB model from: {_current_glb_path}")
                return _current_glb_path, "3D model loaded successfully!"
            else:
                logger.error(f"GLB file not found: {_current_glb_path}")
                return None, "Error: 3D model file not found."
        
        demo.load(fn=load_model, outputs=[reconstruction_output, log_output])
    
    return demo

def start_gradio_server(target_dir, glb_path, port=8080):
    global _gradio_server, _gradio_port, _gradio_running
    
    with _gradio_lock:
        if _gradio_running:
            try:
                logger.info("Stopping existing Gradio server...")
                if _gradio_server:
                    _gradio_server.close()
                _gradio_running = False
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error stopping existing Gradio server: {e}")
        
        _gradio_port = port
        
        def run_server():
            global _gradio_server, _gradio_running
            try:
                logger.info(f"Starting Gradio server on port {port}...")
                
                demo = create_gradio_app(target_dir, glb_path)
                _gradio_server = demo
                _gradio_running = True
                
                demo.launch(
                    server_name="0.0.0.0",
                    server_port=port,
                    share=False,
                    show_error=True,
                    prevent_thread_lock=True
                )
                
                logger.info(f"Gradio server started at http://localhost:{port}")
                
            except Exception as e:
                logger.error(f"Gradio server error: {e}")
                _gradio_running = False
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        
        time.sleep(4)
        
        if _gradio_running:
            gradio_url = f"http://localhost:{port}"
            logger.info(f"Gradio server started successfully at {gradio_url}")
            return gradio_url
        else:
            logger.error("Failed to start Gradio server")
            return None

def get_gradio_url():
    global _gradio_port
    return f"http://localhost:{_gradio_port}"

def is_gradio_running():
    global _gradio_running
    return _gradio_running

def stop_gradio_server():
    global _gradio_server, _gradio_running
    _gradio_running = False
    if _gradio_server is not None:
        try:
            _gradio_server.close()
        except:
            pass
        _gradio_server = None
