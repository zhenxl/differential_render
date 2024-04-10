import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase
import numpy as np

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]).cuda()
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).cuda()

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        # print("in compute weights")
        # print("delta shape: ", deltas.shape)
        # # print("density shape: ", rays_density.shape)
        # batch_size = deltas.shape[0]
        # num_points = deltas.shape[1]
        # weights = torch.zeros(batch_size, num_points, device='cuda')
        # T = torch.ones(batch_size, device='cuda')
        # rays_density_seq = rays_density.squeeze()
        # deltas_seq = deltas.squeeze()
        # # print("delta unique: ", torch.unique(deltas_seq))
        # # print("rays_density_seq unique: ", torch.unique(rays_density_seq))
        # for i in range(num_points):
        #     prod = -rays_density_seq[:, i] * deltas_seq[:, i]
        #     # print("prod: ", torch.unique(prod))
        #     weights[:, i] = T * (1 - torch.exp(prod))
        #     T *=torch.exp(prod)
        # # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        # # print("weights: ", torch.unique(weights))
        B = deltas.shape[0]
        n_points = deltas.shape[1]
        rays_density_sq = rays_density.squeeze() # (B, n_points)
        deltas_sq = deltas.squeeze() # (B, n_points)
        prods = -rays_density_sq * deltas_sq # (B, n_points)
        exp_prods = torch.exp(prods) # (B, n_points)
        one_minus_exp_prods = 1 - exp_prods # (B, n_points)
        T = torch.cumprod(torch.cat((torch.ones(B, device = "cuda").unsqueeze(1), exp_prods), dim = 1)[:, :-1], dim = 1) # (B, n_points)
        return T * one_minus_exp_prods  + eps # (B, n_points)
    
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        # print("weights shape: ", weights.shape)
        # print("ray features shape: ", rays_feature.shape)
        return torch.sum(weights.unsqueeze(-1) * rays_feature.reshape(weights.shape[0], weights.shape[1], -1), dim=1)

    def forward(
        self,
        sampler,
        implicit_fn,
        implicit_fn_fine,
        ray_bundle,
        Nimportance=5
    ):
        B = ray_bundle.shape[0]
        # print("ray bundle shape: ", ray_bundle.shape)
        # print("chunk size: ", self._chunk_size)
        # Process the chunks of rays.
        chunk_outputs = []
        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            # print("curr ray bundle shape: ", cur_ray_bundle.shape)
            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # print("sample length shape: ", cur_ray_bundle.sample_lengths.shape)
           
            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            # print("depth_values shape: ", depth_values.shape)
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights, feature)

            # TODO (1.5): Render depth map
            # print("in depth")
            # print("sample length: ", torch.unique(cur_ray_bundle.sample_lengths))
            depth = self._aggregate(weights, cur_ray_bundle.sample_lengths)
            # print("aggregate depth: ", depth)
            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            if Nimportance > 0:
                # print(" \n in fine model")
                zvals = cur_ray_bundle.sample_lengths
                # print("zvals shape: ", zvals.shape)
                zvals = zvals.squeeze()
                zvals_mid = .5 * (zvals[...,1:] + zvals[...,:-1])
                # print("zvals shape", zvals_mid.shape)
                # print("weights shape", weights[...,1:-1].shape)
                zvals_ = sample_pdf(zvals_mid, weights[...,1:-1],Nimportance).detach()
                # print("N importance shape: ", zvals.shape)
                zvals, _ = torch.sort(torch.cat([zvals, zvals_], -1), -1)

                ray_direction_unsqueeze = torch.nn.functional.normalize(cur_ray_bundle.directions, dim=1).unsqueeze(1)
                unsqueeze_zvals = zvals.unsqueeze(2)
                sampled = ray_direction_unsqueeze * unsqueeze_zvals
                origins =  ray_bundle.origins.unsqueeze(1)
                sample_points = origins + sampled
                # cur_ray_bundle
                # print("origins shape: ", cur_ray_bundle.origins.shape)
                # print("sampled shape: ", sampled.shape)
                
                # print("unsqueeze zval shape: ", unsqueeze_zvals.shape)
                cur_ray_bundle._replace(
                    sample_points=sample_points,
                    sample_lengths=unsqueeze_zvals * torch.ones_like(sample_points[..., :1])
                )
                fine_output = implicit_fn_fine(cur_ray_bundle)
                fine_density = fine_output['density']
                fine_feature = fine_output['feature']
                # print("sample fine_feature: ", torch.unique(fine_feature)) # it has value now
                # print("\n fine density value: ", torch.unique(fine_density))
                # print("sample length shape: ", cur_ray_bundle.sample_lengths.shape)
            
                # Compute length of each ray segment
                depth_values = cur_ray_bundle.sample_lengths[..., 0]
                # print("depth_values shape: ", depth_values.shape)
                deltas = torch.cat(
                    (
                        depth_values[..., 1:] - depth_values[..., :-1],
                        1e10 * torch.ones_like(depth_values[..., :1]),
                    ),
                    dim=-1,
                )[..., None]

                # Compute aggregation weights
                weights = self._compute_weights(
                    deltas.view(-1, n_pts + Nimportance, 1),
                    fine_density.view(-1, n_pts + Nimportance, 1)
                ) 
                # print("weights shape: {} {}", weights.shape, n_pts)

                # TODO (1.5): Render (color) features using weights
                fine_feature = self._aggregate(weights, fine_feature)

                # TODO (1.5): Render depth map
                # print("in depth")
                # print("sample length: ", torch.unique(cur_ray_bundle.sample_lengths))
                fine_depth = self._aggregate(weights, cur_ray_bundle.sample_lengths)
                # print("aggregate depth: ", depth")
                # print("feature shape: ", feature.shape)
                # print("fine_feature value: ",torch.unique(fine_feature))
                cur_out['fine_feature'] = fine_feature
                cur_out['fine_depth'] = fine_depth

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters
    
    def sphere_tracing(
        self,
        implicit_fn,
        origins, # Nx3
        directions, # Nx3
    ):
        '''
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        '''
        # TODO (Q5): Implement sphere tracing
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not
        N = origins.shape[0]
        mask = torch.ones(N, 1, dtype=bool, device= "cuda")
        points = origins 
        it = 0
        while it < self.max_iters:
            dists = implicit_fn(points)
            points += dists * directions
            mask = torch.logical_and(mask, dists < self.far)
            it += 1
        return (points, mask)

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1,3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q7): Convert signed distance to density with alpha, beta parameters
    density = torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / beta),
            1 - 0.5 * torch.exp(signed_distance / beta),
        ) * alpha
    return density

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta) # TODO (Q7): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}
