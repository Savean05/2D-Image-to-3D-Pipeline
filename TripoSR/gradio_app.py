import argparse
import tempfile
from functools import partial

import gradio as gr
import numpy as np
import rembg
import torch
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MC_RESOLUTION_DEFAULT = 256
FOREGROUND_RATIO_DEFAULT = 0.85
CHUNK_SIZE = 8192

model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(CHUNK_SIZE)
model.to(DEVICE)

rembg_session = rembg.new_session()


def check_input_image(input_image: Image.Image | None) -> None:
    """Validate that an image was provided."""
    if input_image is None:
        raise gr.Error("No image uploaded!")


def fill_background(image: Image.Image) -> Image.Image:
    """Fill transparent areas with gray background."""
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = image_array[:, :, :3] * image_array[:, :, 3:4] + (1 - image_array[:, :, 3:4]) * 0.5
    return Image.fromarray((image_array * 255.0).astype(np.uint8))


def preprocess(input_image: Image.Image, do_remove_background: bool, foreground_ratio: float) -> Image.Image:
    """Preprocess input image: remove background and resize."""
    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
    return image


def generate(image: Image.Image, mc_resolution: int, formats: list[str] = None) -> list[str]:
    """Generate 3D mesh from image in multiple formats."""
    if formats is None:
        formats = ["obj", "glb"]
    
    scene_codes = model(image, device=DEVICE)
    mesh = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)[0]
    mesh = to_gradio_3d_orientation(mesh)
    
    mesh_paths = []
    for format_type in formats:
        with tempfile.NamedTemporaryFile(suffix=f".{format_type}", delete=False) as f:
            mesh.export(f.name)
            mesh_paths.append(f.name)
    return mesh_paths


def run_example(image_pil: Image.Image) -> tuple:
    """Run full pipeline on example image."""
    preprocessed = preprocess(image_pil, do_remove_background=False, foreground_ratio=0.9)
    mesh_paths = generate(preprocessed, MC_RESOLUTION_DEFAULT, ["obj", "glb"])
    return preprocessed, mesh_paths[0], mesh_paths[1]


def build_interface() -> gr.Blocks:
    """Build Gradio UI."""
    with gr.Blocks(title="TripoSR") as interface:
        gr.Markdown("""
# TripoSR Demo
[TripoSR](https://github.com/VAST-AI-Research/TripoSR) is a state-of-the-art open-source model for **fast** 3D reconstruction from a single image.

**Tips:**
1. Adjust foreground ratio if results are unsatisfactory
2. Disable "Remove Background" for pre-processed examples
3. Disable "Remove Background" only if input is RGBA with centered content >70% of image
        """)
        
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGBA",
                        sources="upload",
                        type="pil",
                        elem_id="content_image",
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)
                
                with gr.Row():
                    with gr.Group():
                        do_remove_background = gr.Checkbox(
                            label="Remove Background",
                            value=True
                        )
                        foreground_ratio = gr.Slider(
                            label="Foreground Ratio",
                            minimum=0.5,
                            maximum=1.0,
                            value=FOREGROUND_RATIO_DEFAULT,
                            step=0.05,
                        )
                        mc_resolution = gr.Slider(
                            label="Marching Cubes Resolution",
                            minimum=32,
                            maximum=320,
                            value=MC_RESOLUTION_DEFAULT,
                            step=32
                        )
                
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")
            
            with gr.Column():
                with gr.Tab("OBJ"):
                    output_model_obj = gr.Model3D(label="Output Model (OBJ)", interactive=False)
                    gr.Markdown("Note: Download for correct orientation")
                
                with gr.Tab("GLB"):
                    output_model_glb = gr.Model3D(label="Output Model (GLB)", interactive=False)
                    gr.Markdown("Note: Download for correct appearance")
        
        with gr.Row(variant="panel"):
            gr.Examples(
                examples=[
                    "examples/hamburger.png",
                    "examples/poly_fox.png",
                    "examples/robot.png",
                    "examples/teapot.png",
                    "examples/tiger_girl.png",
                    "examples/horse.png",
                    "examples/flamingo.png",
                    "examples/unicorn.png",
                    "examples/chair.png",
                    "examples/iso_house.png",
                    "examples/marble.png",
                    "examples/police_woman.png",
                    "examples/captured.jpeg",
                ],
                inputs=[input_image],
                outputs=[processed_image, output_model_obj, output_model_glb],
                cache_examples=False,
                fn=run_example,
                label="Examples",
                examples_per_page=20,
            )
        
        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=preprocess,
            inputs=[input_image, do_remove_background, foreground_ratio],
            outputs=[processed_image],
        ).success(
            fn=generate,
            inputs=[processed_image, mc_resolution],
            outputs=[output_model_obj, output_model_glb],
        )
    
    return interface


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TripoSR Gradio Interface")
    parser.add_argument("--username", type=str, default=None, help="Username for authentication")
    parser.add_argument("--password", type=str, default=None, help="Password for authentication")
    parser.add_argument("--port", type=int, default=7860, help="Port to run server")
    parser.add_argument("--listen", action="store_true", help="Listen on 0.0.0.0")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--queuesize", type=int, default=1, help="Queue max size")
    
    args = parser.parse_args()
    interface = build_interface()
    interface.queue(max_size=args.queuesize)
    interface.launch(
        auth=(args.username, args.password) if (args.username and args.password) else None,
        share=args.share,
        server_name="0.0.0.0" if args.listen else None,
        server_port=args.port
    )