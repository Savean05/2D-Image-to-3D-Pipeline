# run.py
import argparse
import sys
import time
import warnings
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import rembg
import torch
import trimesh
from PIL import Image

# Swapped to Laplacian for the flat/smooth look
from trimesh.smoothing import filter_laplacian

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")
warnings.filterwarnings("ignore", message=".*Thresholded incomplete Cholesky decomposition failed.*")

from tsr.system import TSR
from tsr.utils import remove_background, repair_mesh, resize_foreground, to_gradio_3d_orientation


def configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_image",
        nargs="*",
        help="Input image file(s) or directory paths. Defaults to all PNGs in ../dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output mesh path for single-image runs only.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save generated meshes into. Defaults to an 'outputs' folder.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional fine-tuned checkpoint to overlay on top of the base model.",
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="stabilityai/TripoSR",
        help="Model id or local model directory.",
    )
    parser.add_argument(
        "--foreground-ratio",
        type=float,
        default=0.85,
        help="Foreground padding ratio applied after background removal.",
    )
    parser.add_argument(
        "--mc-resolution",
        type=int,
        default=256,
        help="Marching cubes resolution.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16384,
        help="Renderer chunk size. Prevents VRAM spillover/freezing.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=25.0,
        help="Mesh extraction threshold.",
    )
    return parser


def load_custom_state_dict(ckpt_path: Path) -> dict:
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")

    return {
        (key[6:] if key.startswith("model.") else key): value
        for key, value in state_dict.items()
    }


def collect_input_images(input_args: list, project_root: Path) -> list[Path]:
    if not input_args:
        dataset_dir = project_root / "dataset"
        paths = sorted(dataset_dir.glob("*.png"))
        if not paths:
            raise FileNotFoundError(f"No PNG images found in {dataset_dir}")
        return paths

    collected = []
    for input_arg in input_args:
        input_path = Path(input_arg).resolve()
        if input_path.is_dir():
            collected.extend(sorted(input_path.glob("*.png")))
        elif input_path.is_file():
            collected.append(input_path)
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")

    unique_paths = list(dict.fromkeys(p for p in collected if p.suffix.lower() == ".png"))

    if not unique_paths:
        raise FileNotFoundError("No PNG images were found in the provided input paths.")
    return unique_paths


def prepare_model_input(input_path: Path, foreground_ratio: float, rembg_session) -> Image.Image:
    with Image.open(input_path) as raw_image:
        raw_image = raw_image.convert("RGBA")
        processed_image = remove_background(
            raw_image,
            rembg_session=rembg_session,
            post_process_mask=True,
            alpha_matting=False
        )

    if processed_image.mode != "RGBA":
        processed_image = processed_image.convert("RGBA")
    processed_image = resize_foreground(processed_image, foreground_ratio)

    image_array = np.array(processed_image).astype(np.float32) / 255.0
    rgb_array = image_array[:, :, :3] * image_array[:, :, 3:4] + 0.5 * (
            1 - image_array[:, :, 3:4]
    )
    return Image.fromarray((rgb_array * 255.0).astype(np.uint8))


def resolve_output_dir(current_dir: Path, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir).resolve()
    else:
        output_dir = current_dir / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_output_path(
        output_dir: Path,
        input_path: Path,
        explicit_output: str | None,
        batch_mode: bool,
) -> Path:
    if explicit_output:
        if batch_mode:
            raise ValueError("--output can only be used when processing a single image.")
        return Path(explicit_output).resolve()

    subfolder = output_dir / input_path.stem
    subfolder.mkdir(parents=True, exist_ok=True)
    return subfolder / f"{input_path.stem}.glb"


def load_model(args: argparse.Namespace) -> TSR:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TSR.from_pretrained(
        args.pretrained_model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading custom checkpoint: {checkpoint_path.name}")
        model.load_state_dict(load_custom_state_dict(checkpoint_path), strict=False)

    model.renderer.set_chunk_size(args.chunk_size)
    model.to(device).eval()
    return model


def build_threshold_fallbacks(threshold: float) -> list[float]:
    floor = max(0.25, threshold * 0.5)
    return [
        max(0.5, threshold * 0.75),
        floor,
        threshold * 1.25,
    ]


def resolve_repair_resolution(mc_resolution: int) -> int:
    return 400


def get_adaptive_smoothing_iterations(mesh: trimesh.Trimesh) -> int:
    """
    Dynamically calculates the optimal number of Laplacian smoothing iterations.
    Dense/complex meshes safely absorb more passes to remove blocks.
    Simple/thin meshes get fewer passes to prevent melting.
    """
    if len(mesh.faces) == 0:
        return 0

    # Heuristic: 1 iteration per 1500 faces
    calculated_iters = int(len(mesh.faces) / 1500)

    # Clamp to ensure it always smooths a little, but never melts the object
    return max(15, min(85, calculated_iters))


def generate_mesh_for_image(
        model: TSR,
        device: str,
        input_path: Path,
        output_path: Path,
        args: argparse.Namespace,
        rembg_session,
) -> None:
    print(f"\n--- Generating mesh for {input_path.name} ---")

    print("  [1/8] Removing background...")
    t_start = time.time()
    final_input = prepare_model_input(input_path, args.foreground_ratio, rembg_session)
    print(f"        ✓ Completed in {time.time() - t_start:.2f}s")

    print("  [2/8] Running TripoSR model...")
    t_start = time.time()
    with torch.no_grad():
        scene_codes = model([final_input], device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"        ✓ Completed in {time.time() - t_start:.2f}s")

        print("  [3/8] Extracting mesh...")
        t_start = time.time()
        meshes = model.extract_mesh(
            scene_codes,
            has_vertex_color=True,
            resolution=args.mc_resolution,
            threshold=args.threshold,
            threshold_fallbacks=build_threshold_fallbacks(args.threshold),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"        ✓ Completed in {time.time() - t_start:.2f}s")

    if not meshes or len(meshes[0].vertices) == 0 or len(meshes[0].faces) == 0:
        raise RuntimeError(f"Mesh extraction returned no geometry for {input_path}")

    print("  [4/8] Repairing mesh...")
    t_start = time.time()
    mesh = trimesh.Trimesh(
        vertices=meshes[0].vertices,
        faces=meshes[0].faces,
        process=False,
    )
    mesh = repair_mesh(mesh, voxel_resolution=resolve_repair_resolution(args.mc_resolution))
    print(f"        ✓ Completed in {time.time() - t_start:.2f}s")

    print("  [5/8] Exporting Unsmoothed Preview...")
    t_start = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unsmoothed_mesh = to_gradio_3d_orientation(mesh.copy())
    unsmoothed_mesh.export(output_path.parent / f"{input_path.stem}_unsmoothed.glb")
    print(f"        ✓ Completed in {time.time() - t_start:.2f}s")

    print("  [6/8] Smoothing Mesh...")
    t_start = time.time()
    optimal_iters = get_adaptive_smoothing_iterations(mesh)
    print(f"        -> Using {optimal_iters} iterations for {len(mesh.faces):,} faces")
    filter_laplacian(mesh, iterations=optimal_iters)
    print(f"        ✓ Completed in {time.time() - t_start:.2f}s")

    print("  [7/8] Optimizing Mesh...")
    t_start = time.time()
    try:
        mesh = mesh.simplify_quadric_decimation(0.5)
        print(f"        ✓ Completed in {time.time() - t_start:.2f}s")
    except Exception as e:
        print(f"        ! Decimation skipped: {e}")

    print("  [8/8] Applying rotation and exporting...")
    t_start = time.time()
    mesh = to_gradio_3d_orientation(mesh)

    mesh.export(output_path)

    shutil.copy(input_path, output_path.parent / f"{input_path.stem}_original.png")
    final_input.save(output_path.parent / f"{input_path.stem}_edited.png")

    print(f"        ✓ Completed in {time.time() - t_start:.2f}s")


def main() -> None:
    configure_console_output()
    parser = build_parser()
    args = parser.parse_args()
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    try:
        input_paths = collect_input_images(args.input_image, project_root)
        output_dir = resolve_output_dir(current_dir, args.output_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {device}")
        print("Loading TripoSR model...")
        model = load_model(args)
        batch_mode = len(input_paths) > 1

        print(f"Processing {len(input_paths)} image(s)")
        print("Initializing background removal...\n")
        rembg_session = rembg.new_session("isnet-general-use")

        total_start_time = time.time()
        for input_path in input_paths:
            output_path = resolve_output_path(
                output_dir=output_dir,
                input_path=input_path,
                explicit_output=args.output,
                batch_mode=batch_mode,
            )
            generate_mesh_for_image(model, device, input_path, output_path, args, rembg_session)

        print(f"\n✓ All meshes generated in {time.time() - total_start_time:.2f}s total")
    except Exception as exc:
        print(f"✗ Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()