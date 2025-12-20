#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.rigid_utils import from_homogenous, to_homogenous
import torch.nn.functional as F


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, d_xyz, d_rotation, d_scaling, is_6dof=False,
           scaling_modifier=1.0, override_color=None, mask=False, tau=None, training=True):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_densify = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_densify.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_6dof:
        if torch.is_tensor(d_xyz) is False:
            means3D = pc.get_xyz
        else:
            means3D = from_homogenous(
                torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
    else:
        means3D = pc.get_xyz + d_xyz

    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if colors_precomp is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # if mask, all parameters are selected by mask.
    mask_index = None
    if mask:
        all_feature = torch.cat([means3D, screenspace_points, screenspace_points_densify, shs.view(means3D.shape[0], -1), \
                                 opacity, scales, rotations], dim=1)

        # 3+3+3+3x16+1+3+4 = 65
        feature_for_gumbel = pc.get_mlp_gumbel(all_feature)
        
        def gumbel_softmax(x, tau, hard, dim):
            if training:
                return F.gumbel_softmax(x, tau=tau, hard=hard, dim=dim)
            else:
                index = x.max(dim, keepdim=True)[1]
                return torch.zeros_like(x).scatter_(dim, index, 1.0)
        
        mask = gumbel_softmax(feature_for_gumbel, tau=tau, hard=True, dim=-1)
        mask_index = mask[:, 0].view(-1, 1).contiguous()

        # print(f"mask loss: {torch.mean(mask_index)}")

        means3D = means3D * mask_index
        screenspace_points = screenspace_points * mask_index
        screenspace_points_densify = screenspace_points_densify * mask_index
        shs = shs * mask_index.view(-1, 1, 1)
        colors_precomp = colors_precomp * mask_index if colors_precomp is not None else None
        opacity = opacity * mask_index
        scales = scales * mask_index
        rotations = rotations * mask_index
        cov3D_precomp = cov3D_precomp * mask_index if cov3D_precomp is not None else None

        # mask_index = pc.get_gumbel_mask
        # means3D = means3D[mask_index]
        # screenspace_points = screenspace_points[mask_index]
        # screenspace_points_densify = screenspace_points_densify[mask_index]
        # shs = shs[mask_index]
        # colors_precomp = colors_precomp[mask_index] if colors_precomp is not None else None
        # opacity = opacity[mask_index]
        # scales = scales[mask_index]
        # rotations = rotations[mask_index]
        # cov3D_precomp = cov3D_precomp[mask_index] if cov3D_precomp is not None else None

    # if all_mask is not None:
    #     means3D = means3D * all_mask
    #     # screenspace_points = screenspace_points * all_mask
    #     # screenspace_points_densify = screenspace_points_densify * all_mask
    #     shs = shs * all_mask.view(-1, 1, 1)
    #     colors_precomp = colors_precomp * all_mask if colors_precomp is not None else None
    #     opacity = opacity * all_mask
    #     scales = scales * all_mask
    #     rotations = rotations * all_mask
    #     cov3D_precomp = cov3D_precomp * all_mask if cov3D_precomp is not None else None

        # means3D = torch.where(all_mask > 0, means3D, torch.zeros_like(means3D))
        # screenspace_points = torch.where(all_mask > 0, screenspace_points, torch.zeros_like(screenspace_points))
        # screenspace_points_densify = torch.where(all_mask > 0, screenspace_points_densify, torch.zeros_like(screenspace_points_densify).requires_grad_(True))
        # shs = torch.where(all_mask.view(-1, 1, 1) > 0, shs, torch.zeros_like(shs))
        # colors_precomp = torch.where(all_mask > 0, colors_precomp, torch.zeros_like(colors_precomp)) if colors_precomp is not None else None
        # opacity = torch.where(all_mask > 0, opacity, torch.zeros_like(opacity))
        # scales = torch.where(all_mask > 0, scales, torch.zeros_like(scales))
        # rotations = torch.where(all_mask > 0, rotations, torch.zeros_like(rotations))
        # cov3D_precomp = torch.where(all_mask > 0, cov3D_precomp, torch.zeros_like(cov3D_precomp)) if cov3D_precomp is not None else None
        # pass

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        means2D_densify=screenspace_points_densify,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # print(screenspace_points_densify.requires_grad)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "viewspace_points_densify": screenspace_points_densify,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "mask": mask_index,}
