import math
import os
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from omegaconf import OmegaConf
from PIL import Image

from .models.isosurface import MarchingCubeHelper
from .utils import (
    BaseModule,
    ImagePreprocessor,
    find_class,
    get_spherical_cameras,
    scale_tensor,
)


class TSR(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cond_image_size: int

        image_tokenizer_cls: str
        image_tokenizer: dict

        tokenizer_cls: str
        tokenizer: dict

        backbone_cls: str
        backbone: dict

        post_processor_cls: str
        post_processor: dict

        decoder_cls: str
        decoder: dict

        renderer_cls: str
        renderer: dict

    cfg: Config

    @staticmethod
    def _resolve_pretrained_file(
        pretrained_model_name_or_path: str, filename: str
    ) -> str:
        cached_path = try_to_load_from_cache(
            repo_id=pretrained_model_name_or_path, filename=filename
        )
        if isinstance(cached_path, str) and os.path.isfile(cached_path):
            return cached_path
        return hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename=filename
        )

    @staticmethod
    def _load_state_dict(weight_path: str):
        try:
            checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(weight_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint format: {weight_path}")

        return {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in checkpoint.items()
        }

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, config_name: str, weight_name: str
    ):
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, config_name)
            weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
            missing_files = [
                path
                for path in (config_path, weight_path)
                if not os.path.isfile(path)
            ]
            if missing_files:
                raise FileNotFoundError(
                    f"Missing pretrained model files: {', '.join(missing_files)}"
                )
        else:
            config_path = cls._resolve_pretrained_file(
                pretrained_model_name_or_path, config_name
            )
            weight_path = cls._resolve_pretrained_file(
                pretrained_model_name_or_path, weight_name
            )

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = cls(cfg)
        state_dict = cls._load_state_dict(weight_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Checkpoint did not match the TSR model. "
                f"Missing keys: {missing_keys[:5]} "
                f"Unexpected keys: {unexpected_keys[:5]}"
            )
        return model

    def configure(self):
        self.image_tokenizer = find_class(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )
        self.tokenizer = find_class(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = find_class(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = find_class(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )
        self.decoder = find_class(self.cfg.decoder_cls)(self.cfg.decoder)
        self.renderer = find_class(self.cfg.renderer_cls)(self.cfg.renderer)
        self.image_processor = ImagePreprocessor()
        self.isosurface_helper = None

    def forward(
        self,
        image: Union[
            PIL.Image.Image,
            np.ndarray,
            torch.FloatTensor,
            List[PIL.Image.Image],
            List[np.ndarray],
            List[torch.FloatTensor],
        ],
        device: str,
    ) -> torch.FloatTensor:
        rgb_cond = self.image_processor(image, self.cfg.cond_image_size)[:, None].to(
            device
        )
        batch_size = rgb_cond.shape[0]

        input_image_tokens: torch.Tensor = self.image_tokenizer(
            rearrange(rgb_cond, "B Nv H W C -> B Nv C H W", Nv=1),
        )

        input_image_tokens = rearrange(
            input_image_tokens, "B Nv C Nt -> B (Nv Nt) C", Nv=1
        )

        tokens: torch.Tensor = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        return scene_codes

    def render(
        self,
        scene_codes,
        n_views: int,
        elevation_deg: float = 0.0,
        camera_distance: float = 1.9,
        fovy_deg: float = 40.0,
        height: int = 256,
        width: int = 256,
        return_type: str = "pil",
    ):
        rays_o, rays_d = get_spherical_cameras(
            n_views, elevation_deg, camera_distance, fovy_deg, height, width
        )
        rays_o, rays_d = rays_o.to(scene_codes.device), rays_d.to(scene_codes.device)

        def process_output(image: torch.FloatTensor):
            if return_type == "pt":
                return image
            elif return_type == "np":
                return image.detach().cpu().numpy()
            elif return_type == "pil":
                return Image.fromarray(
                    (image.detach().cpu().numpy() * 255.0).astype(np.uint8)
                )
            else:
                raise NotImplementedError

        images = []
        for scene_code in scene_codes:
            images_ = []
            for i in range(n_views):
                with torch.no_grad():
                    image = self.renderer(
                        self.decoder, scene_code, rays_o[i], rays_d[i]
                    )
                images_.append(process_output(image))
            images.append(images_)

        return images

    @staticmethod
    def _cleanup_components(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        parts = mesh.split(only_watertight=False)
        if len(parts) <= 1:
            return mesh

        non_empty_parts = [
            part for part in parts if len(part.vertices) > 0 and len(part.faces) > 0
        ]
        if len(non_empty_parts) <= 1:
            return non_empty_parts[0] if non_empty_parts else mesh

        largest_face_count = max(len(part.faces) for part in non_empty_parts)
        min_face_count = max(8, int(largest_face_count * 0.00002))
        kept_parts = []
        for part in non_empty_parts:
            if len(part.faces) < min_face_count:
                continue
            kept_parts.append(part)

        if not kept_parts:
            kept_parts = [max(non_empty_parts, key=lambda part: len(part.faces))]

        cleaned = (
            kept_parts[0]
            if len(kept_parts) == 1
            else trimesh.util.concatenate(kept_parts)
        )
        cleaned.remove_unreferenced_vertices()
        return cleaned

    @staticmethod
    def _empty_mesh() -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )

    @staticmethod
    def _score_mesh_candidate(mesh: trimesh.Trimesh) -> float:
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return float("-inf")

        parts = [
            part
            for part in mesh.split(only_watertight=False)
            if len(part.vertices) > 0 and len(part.faces) > 0
        ]
        if not parts:
            return float("-inf")

        total_faces = sum(len(part.faces) for part in parts)
        largest_faces = max(len(part.faces) for part in parts)
        fragment_faces = total_faces - largest_faces
        bbox_volume = float(np.prod(np.maximum(mesh.bounding_box.extents, 1.0e-6)))
        return (
            float(total_faces)
            + 0.35 * float(largest_faces)
            - 0.15 * float(fragment_faces)
            + 500.0 * bbox_volume
        )

    def _extract_mesh_candidate(
        self,
        scene_code: torch.FloatTensor,
        density: torch.FloatTensor,
        has_vertex_color: bool,
        threshold: float,
    ) -> trimesh.Trimesh:
        v_pos, t_pos_idx = self.isosurface_helper(-(density - threshold))
        if v_pos.numel() == 0 or t_pos_idx.numel() == 0:
            return self._empty_mesh()

        v_pos = scale_tensor(
            v_pos,
            self.isosurface_helper.points_range,
            (-self.renderer.cfg.radius, self.renderer.cfg.radius),
        )
        color = None
        if has_vertex_color:
            with torch.no_grad():
                color = self.renderer.query_triplane(
                    self.decoder,
                    v_pos,
                    scene_code,
                )["color"]
        mesh = trimesh.Trimesh(
            vertices=v_pos.cpu().numpy(),
            faces=t_pos_idx.cpu().numpy(),
            vertex_colors=color.cpu().numpy() if has_vertex_color else None,
            process=False,
        )
        return self._cleanup_components(mesh)

    def set_marching_cubes_resolution(self, resolution: int):
        if (
            self.isosurface_helper is not None
            and self.isosurface_helper.resolution == resolution
        ):
            return
        self.isosurface_helper = MarchingCubeHelper(resolution)

    def extract_mesh(
        self,
        scene_codes,
        has_vertex_color,
        resolution: int = 256,
        threshold: float = 25.0,
        threshold_fallbacks: List[float] | None = None,
    ):
        self.set_marching_cubes_resolution(resolution)
        meshes = []
        threshold_candidates = [float(threshold)]
        if threshold_fallbacks:
            for fallback in threshold_fallbacks:
                fallback = float(fallback)
                if fallback > 0 and fallback not in threshold_candidates:
                    threshold_candidates.append(fallback)

        for scene_code in scene_codes:
            with torch.no_grad():
                density = self.renderer.query_triplane(
                    self.decoder,
                    scale_tensor(
                        self.isosurface_helper.grid_vertices.to(scene_codes.device),
                        self.isosurface_helper.points_range,
                        (-self.renderer.cfg.radius, self.renderer.cfg.radius),
                    ),
                    scene_code,
                )["density_act"]

            best_mesh = self._empty_mesh()
            best_score = float("-inf")
            for threshold_candidate in threshold_candidates:
                candidate_mesh = self._extract_mesh_candidate(
                    scene_code=scene_code,
                    density=density,
                    has_vertex_color=has_vertex_color,
                    threshold=threshold_candidate,
                )
                candidate_score = self._score_mesh_candidate(candidate_mesh)
                if candidate_score > best_score:
                    best_mesh = candidate_mesh
                    best_score = candidate_score

            meshes.append(best_mesh)
        return meshes
