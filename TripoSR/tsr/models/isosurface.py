from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from skimage.measure import marching_cubes as skimage_marching_cubes

try:
    from torchmcubes import marching_cubes as torchmcubes_marching_cubes
except ImportError:
    torchmcubes_marching_cubes = None


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self.mc_func: Optional[Callable] = torchmcubes_marching_cubes
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # Keep vertices on CPU so very large resolutions still fit.
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def _march_with_skimage(self, level: torch.FloatTensor):
        verts, faces, _, _ = skimage_marching_cubes(
            level.detach().cpu().numpy(), level=0.0
        )
        verts = torch.from_numpy(verts.copy()).float() / (self.resolution - 1.0)
        faces = torch.from_numpy(faces.copy().astype("int64"))
        return verts.to(level.device), faces.to(level.device)

    def forward(
        self,
        level: torch.FloatTensor,
    ):
        level = -level.view(self.resolution, self.resolution, self.resolution)

        if self.mc_func is None:
            return self._march_with_skimage(level)

        try:
            v_pos, t_pos_idx = self.mc_func(level.detach(), 0.0)
        except AttributeError:
            v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)

        v_pos = v_pos[..., [2, 1, 0]]
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
