import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray).to('cuda')
        # print("z val shape: ", z_vals.shape)
        # TODO (Q1.4): Sample points from z values
        origins = ray_bundle.origins.unsqueeze(1) # h*w,1, 3
        unsqueeze_z = z_vals.unsqueeze(1) # n_points, 1
        ray_direction_unsqueeze = torch.nn.functional.normalize(ray_bundle.directions, dim=1).unsqueeze(1) # h*w, 1, 3
        sampled =  ray_direction_unsqueeze * unsqueeze_z
        # print("type: ", type(sampled))
        sample_points = origins + sampled
        # print("type: ", type(origins))
        # print(' sampled shape: ', sample_points.shape)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=unsqueeze_z * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}