# take_pictures.py
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

RENDER_WIDTH = 1024
RENDER_HEIGHT = 1024
RENDER_ITERATIONS = 30
RENDER_SLEEP_MS = 0.01
DEFAULT_COLOR = [128, 128, 128, 255]


def load_mesh_with_colors(file_path: Path) -> o3d.geometry.TriangleMesh:
    t_mesh = trimesh.load(str(file_path), process=False)

    if isinstance(t_mesh, trimesh.Scene):
        t_mesh = t_mesh.to_geometry()

    if not hasattr(t_mesh.visual, 'vertex_colors') or len(t_mesh.visual.vertex_colors) == 0:
        t_mesh.visual.vertex_colors = np.full((len(t_mesh.vertices), 4), DEFAULT_COLOR)

    vertices = np.array(t_mesh.vertices)
    faces = np.array(t_mesh.faces)
    colors = np.array(t_mesh.visual.vertex_colors)[:, :3] / 255.0

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    return mesh


def configure_renderer(vis: o3d.visualization.Visualizer) -> None:
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.light_on = True
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    opt.point_size = 1
    opt.line_width = 1


def position_camera(vis: o3d.visualization.Visualizer, mesh: o3d.geometry.TriangleMesh) -> None:
    bbox = mesh.get_axis_aligned_bounding_box()
    mesh_center = bbox.get_center()

    ctr = vis.get_view_control()
    ctr.set_front([0, 0, 1])
    ctr.set_up([0, 1, 0])
    ctr.set_lookat(mesh_center)
    ctr.set_zoom(0.85)


def render_mesh_to_image(mesh: o3d.geometry.TriangleMesh, output_path: Path) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Render", width=RENDER_WIDTH, height=RENDER_HEIGHT, visible=False)
    vis.add_geometry(mesh)

    configure_renderer(vis)
    position_camera(vis, mesh)

    for _ in range(RENDER_ITERATIONS):
        vis.poll_events()
        vis.update_renderer()
        time.sleep(RENDER_SLEEP_MS)

    buffer = vis.capture_screen_float_buffer(do_render=True)
    image_np = (np.asarray(buffer) * 240).astype(np.uint8)
    Image.fromarray(image_np).save(str(output_path))
    vis.destroy_window()


def take_pictures() -> None:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "TripoSR" / "outputs"

    if not output_dir.exists():
        print(f"✗ Output directory not found: {output_dir}")
        print("  Run 'python TripoSR/run.py' first")
        return

    model_files = sorted(output_dir.glob("**/*.glb"))
    if not model_files:
        print(f"✗ No .glb models found in {output_dir}")
        return

    print(f"Found {len(model_files)} models. Generating previews...\n")

    successful = 0
    for file_path in model_files:
        image_path = file_path.with_suffix(".png")

        if "unsmoothed" in file_path.name.lower():
            model_name = f"{file_path.stem} (Unsmoothed)"
        else:
            model_name = f"{file_path.stem} (Final)"

        try:
            print(f"  {model_name}...", end=" ", flush=True)
            mesh = load_mesh_with_colors(file_path)
            render_mesh_to_image(mesh, image_path)
            print("✓")
            successful += 1
        except Exception as e:
            print(f"✗ ({e})")

    print(f"\n✓ Generated {successful}/{len(model_files)} previews")


if __name__ == "__main__":
    take_pictures()