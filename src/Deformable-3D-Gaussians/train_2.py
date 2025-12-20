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

import sys
sys.path.append('./src')
sys.path.append('./src/Deformable-3D-Gaussians')
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import torchvision
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from datetime import datetime
import cv2
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from editing import Editing_Module

def training(dataset, opt, pipe, testing_iterations, saving_iterations, load_checkpoint=None, diffusion_ckpt=None, prompt=None, \
    first_idx=0, write_local_log=False, loss_selection='two', acc_step=16, args=None):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset.is_blender, dataset.is_6dof)
    deform.train_setting(opt)

    if load_checkpoint is not None:
        deform.load_weights(dataset.model_path)
        scene = Scene(dataset, gaussians, load_iteration=load_checkpoint, shuffle=False)
    else:
        scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    start_iter = load_checkpoint + 1 if load_checkpoint is not None else 1

    progress_bar = tqdm(range(start_iter, opt.iterations+1), desc="Unedited 3DGS Training progress")

    # print(start_iter)
    # sys.exit()
    for iteration in range(start_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # print(gt_image.shape)
        # sys.exit()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


    ########### Save Masks ################
    with torch.no_grad():

        viewpoint_stack = scene.getTrainCameras().copy()
        # 定义一个全零矩阵，维度为gaussians.get_xyz.shape[0] x 1
        all_mask = torch.zeros(gaussians.get_xyz.shape[0], 1, device='cuda')
        print(len(viewpoint_stack), "viewpoint_stack length")
        for idx, viewpoint_cam in enumerate(viewpoint_stack):
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
            fid = viewpoint_cam.fid
            xyz = gaussians.get_xyz
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

            # Render
            image_un_masked = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, mask=False, training=False)["render"]
            gt_path = os.path.join(args.model_path, "gt")
            os.makedirs(gt_path, exist_ok=True)
            torchvision.utils.save_image(image_un_masked, os.path.join(gt_path, \
                                    str(idx).zfill(6) + '.png'))

    sys.exit()
    # render 3DGS
    # with torch.no_grad():

    #     viewpoint_stack = scene.getTrainCameras().copy()
    #     print(len(viewpoint_stack), "viewpoint_stack length")
    #     for idx, viewpoint_cam in enumerate(viewpoint_stack):
    #         if dataset.load2gpu_on_the_fly:
    #             viewpoint_cam.load2device()
    #         fid = viewpoint_cam.fid
    #         xyz = gaussians.get_xyz
    #         time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
    #         d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
    #         # Render
    #         image_un_masked = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, mask=False, training=False)["render"]
    #         gt_path = os.path.join(args.model_path, "gt_first_render_3DGS")
    #         os.makedirs(gt_path, exist_ok=True)
    #         torchvision.utils.save_image(image_un_masked, os.path.join(gt_path, \
    #                                 str(idx).zfill(6) + '.png'))


    ############# Masked 3DGS Begin #####################
    gaussians.training_setup(opt, mask=True)
    mask_training_iterations = args.mask_iter # 5000
    # acc_step = 16
    # gaussians.apply_mask()
    progress_bar = tqdm(range(mask_training_iterations), desc="3DGS Selected Mask Training progress")

    if args.all_mask_path is None:
        for iteration in range(mask_training_iterations):
            if iteration % 10 == 0:
                progress_bar.update(10)

            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()

            viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack) - 1)]
            if dataset.load2gpu_on_the_fly:
                viewpoint_cam.load2device()
            fid = viewpoint_cam.fid

            total_frame = len(scene.getTrainCameras())
            time_interval = 1 / total_frame


            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration - opt.iterations)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

            # Render
            tau = 3*(1-iteration/mask_training_iterations) + 0.001
            render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, mask=True, tau=tau)
            image, viewspace_point_tensor, visibility_filter, radii, mask_index = render_pkg_re["render"], render_pkg_re[
                "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["mask"]

            masked_image = viewpoint_cam.original_image * viewpoint_cam.original_mask


            image_un_masked = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, mask=False)["render"]

            if loss_selection == 'two':
                Ll1 = l1_loss(image * viewpoint_cam.original_mask, image_un_masked * viewpoint_cam.original_mask)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image * viewpoint_cam.original_mask, image_un_masked * viewpoint_cam.original_mask)) # + 0.1 * torch.mean(mask_index)
                Ll1_background = l1_loss(image * (1 - viewpoint_cam.original_mask), torch.zeros_like(image))
                loss += 0.3 * ((1.0 - opt.lambda_dssim) * Ll1_background)
            
            elif loss_selection == "wo_ssim":
                Ll1 = l1_loss(image * viewpoint_cam.original_mask, image_un_masked * viewpoint_cam.original_mask)
                loss = Ll1
                Ll1_background = l1_loss(image * (1 - viewpoint_cam.original_mask), torch.zeros_like(image))
                loss += 0.3 * Ll1_background

            elif loss_selection == "constrastive":
                Ll1 = l1_loss(image * viewpoint_cam.original_mask, image_un_masked * viewpoint_cam.original_mask)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image * viewpoint_cam.original_mask, image_un_masked * viewpoint_cam.original_mask)) # + 0.1 * torch.mean(mask_index)
                Ll1_background = l1_loss(image * (1 - viewpoint_cam.original_mask), torch.zeros_like(image))
                Ll1_background_constrastive = l1_loss(image * (1 - viewpoint_cam.original_mask), image_un_masked * (1 - viewpoint_cam.original_mask))
                loss += 0.3 * ((1.0 - opt.lambda_dssim) * Ll1_background) - 0.1 * ((1.0 - opt.lambda_dssim) * Ll1_background_constrastive)

            elif loss_selection == "high_wight_constrastive":
                high_weight = 3.0
                Ll1 = l1_loss(image * viewpoint_cam.original_mask, image_un_masked * viewpoint_cam.original_mask)
                loss = high_weight * ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image * viewpoint_cam.original_mask, image_un_masked * viewpoint_cam.original_mask))) # + 0.1 * torch.mean(mask_index)
                Ll1_background = l1_loss(image * (1 - viewpoint_cam.original_mask), torch.zeros_like(image))
                Ll1_background_constrastive = l1_loss(image * (1 - viewpoint_cam.original_mask), image_un_masked * (1 - viewpoint_cam.original_mask))
                loss += 0.5 * ((1.0 - opt.lambda_dssim) * Ll1_background) - 0.3 * ((1.0 - opt.lambda_dssim) * Ll1_background_constrastive)
            loss = loss / acc_step

            if iteration % 100 == 0:
                print(f"l1 loss: {(1.0 - opt.lambda_dssim) * Ll1}, ssim loss: {opt.lambda_dssim * (1.0 - ssim(image, masked_image))}, mask loss: {torch.mean(mask_index)}")
            loss.backward()
            if iteration >= (mask_training_iterations - 20) or iteration % 100 == 0:
                # 保存image
                folder_path = os.path.join(args.model_path, f"mask_render_{loss_selection}_{acc_step}")
                os.makedirs(folder_path, exist_ok=True)
                torchvision.utils.save_image(image, os.path.join(folder_path, \
                                        str(iteration).zfill(6) + '.png'))

                # Render
                gt_folder_path = os.path.join(args.model_path, "unmask_render")
                os.makedirs(gt_folder_path, exist_ok=True)
                torchvision.utils.save_image(image_un_masked, os.path.join(gt_folder_path, \
                                        str(iteration).zfill(6) + '.png'))

            
                
            # Optimizer step
            # if iteration < opt.iterations:
            if iteration % acc_step == 0:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration, mask=True, mask_training_iterations=mask_training_iterations)
                # deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                # deform.optimizer.zero_grad()
                # deform.update_learning_rate(iteration)

        # sys.exit()
        ############ Masked 3DGS End #####################

        ########### Save Masks ################
        with torch.no_grad():

            viewpoint_stack = scene.getTrainCameras().copy()
            # 定义一个全零矩阵，维度为gaussians.get_xyz.shape[0] x 1
            all_mask = torch.zeros(gaussians.get_xyz.shape[0], 1, device='cuda')
            print(len(viewpoint_stack), "viewpoint_stack length")
            for idx, viewpoint_cam in enumerate(viewpoint_stack):
                if dataset.load2gpu_on_the_fly:
                    viewpoint_cam.load2device()
                fid = viewpoint_cam.fid
                xyz = gaussians.get_xyz
                time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                results = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, mask=True, training=False)
                rendering = results["render"]
                depth = results["depth"]
                depth = depth / (depth.max() + 1e-5)

                # 累加所有被认为是object区域的高斯球位置。
                all_mask += results["mask"]

                gt = viewpoint_cam.original_image[0:3, :, :]

                # 保存image
                folder_path = os.path.join(args.model_path, f"mask_render_object_{loss_selection}_{acc_step}")
                os.makedirs(folder_path, exist_ok=True)
                torchvision.utils.save_image(rendering, os.path.join(folder_path, \
                                        str(idx).zfill(6) + '.png'))

                # Render
                image_un_masked = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof, mask=False, training=False)["render"]
                gt_path = os.path.join(args.model_path, "gt")
                os.makedirs(gt_path, exist_ok=True)
                torchvision.utils.save_image(image_un_masked, os.path.join(gt_path, \
                                        str(idx).zfill(6) + '.png'))

            # 将all_mask中大于1的全部置为1，小于0.1的全部置为0
            all_mask = (all_mask > 0.1).float()  # 将大于0.1的置为1，小于等于0.1的置为0
            # 存储 all_mask 到硬盘
            mask_path = os.path.join(args.model_path, "mask_path")
            os.makedirs(mask_path, exist_ok=True)
            all_mask_path = os.path.join(mask_path, 'all_mask.pt')
            torch.save(all_mask, all_mask_path)
            all_mask = all_mask.clone().detach().requires_grad_(True)
            gaussians.all_mask = all_mask

    else:
        # 从 all_mask_path 中读取all_mask
        all_mask = torch.load(args.all_mask_path).cuda()
        all_mask = all_mask.clone().detach().requires_grad_(True)
        gaussians.all_mask = all_mask

    # all_mask = torch.load("/mnt/dongxu-fs2/data-ssd/leiliu/3d_editing/20250629_CTRL_D_w_VGGT/output/haru-sit_1/mask_path/all_mask.pt").cuda()
    # all_mask = all_mask.clone().detach().requires_grad_(True)
    # gaussians.all_mask = all_mask

    # sys.exit()

    gaussians.training_setup(opt)
    ############################################ Editing ############################################

    Editing_module = Editing_Module(diffusion_ckpt = diffusion_ckpt)

    if write_local_log:
        log_folder = os.path.join(args.model_path, 'log', datetime.now().strftime('%Y_%m_%d_%H-%M-%S'))
        os.makedirs(log_folder, exist_ok=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.edit_iterations), desc="3DGS Editing Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    is_first = True
    edit_stack = None
    edit_image_dataset = {}
    for cam in scene.getTrainCameras():
        edit_image_dataset[cam.uid] = cam.original_image

    viewpoint_stack = None
    edited_stack = [scene.getTrainCameras().copy()[first_idx]]

    total_frame = len(scene.getTrainCameras())
    time_interval = 1 / total_frame

    for iteration in range(opt.iterations + 1, opt.iterations + opt.edit_iterations + 1):

        if args.fine_tune:
            if iteration >= (opt.iterations + opt.edit_iterations * (1-args.fine_tune_percent) + 1):
                gaussians.all_mask = None
 
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = edited_stack.copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)
        
        ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration - opt.iterations)
        d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)


        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]

        # edited_image = edit_image_dataset[viewpoint_cam.uid].cuda()

        # only cal different area loss
        edited_image = edit_image_dataset[viewpoint_cam.uid].cuda()
        # image = image * viewpoint_cam.original_mask.cuda()

        # Loss
        # 如果需要进行mask，则仅使用mask区域进行loss计算。
        # if gaussians.all_mask is not None:
        #     Ll1 = l1_loss(image, edited_image * viewpoint_cam.original_mask.cuda())
        #     loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, edited_image * viewpoint_cam.original_mask.cuda()))
        # else:
        Ll1 = l1_loss(image, edited_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, edited_image))

        if iteration % opt.temp_loss_interval == 0 and len(edited_stack) == total_frame and viewpoint_cam.uid > 0:
            N = gaussians.get_xyz.shape[0]
            viewpoint_cam_prev = scene.getTrainCameras()[viewpoint_cam.uid - 1]
            assert viewpoint_cam_prev.uid == viewpoint_cam.uid - 1
            time_input_prev = viewpoint_cam_prev.fid.unsqueeze(0).expand(N, -1)
            d_xyz_prev, d_rotation_prev, d_scaling_prev = deform.step(gaussians.get_xyz.detach(), time_input_prev)
            render_pkg_re_prev = render(viewpoint_cam_prev, gaussians, pipe, background, d_xyz_prev, d_rotation_prev, d_scaling_prev, dataset.is_6dof)
            image_prev, _, _, _ = render_pkg_re_prev["render"], render_pkg_re_prev[
                "viewspace_points"], render_pkg_re_prev["visibility_filter"], render_pkg_re_prev["radii"]

            # for different area loss
            # image_prev = image_prev * viewpoint_cam_prev.original_mask.cuda()
            # if gaussians.all_mask is not None:
            #     loss_temp = opt.lambda_temp * l1_loss(image, image_prev * viewpoint_cam_prev.original_mask.cuda())
            # else:
            loss_temp = opt.lambda_temp * l1_loss(image, image_prev)
            loss += loss_temp

        loss.backward()

        # 进行梯度清零操作
        if gaussians.all_mask is not None:
            gaussians._xyz.grad = gaussians._xyz.grad * gaussians.all_mask
            gaussians._features_dc.grad = gaussians._features_dc.grad * gaussians.all_mask.view(-1,1,1)
            gaussians._features_rest.grad = gaussians._features_rest.grad * gaussians.all_mask.view(-1,1,1)
            gaussians._opacity.grad = gaussians._opacity.grad * gaussians.all_mask
            gaussians._scaling.grad = gaussians._scaling.grad * gaussians.all_mask
            gaussians._rotation.grad = gaussians._rotation.grad * gaussians.all_mask

            # for param in list(deform.deform.parameters()):
            #     print(param)

            # deform 怎么梯度清零回头研究一下，这里边不是[256,256]就是[256]维度，找不到gaussian point啊，没办法执行mask selected


        iter_end.record()

        if iteration < opt.iterations + opt.editing_warmup_iter:
            with torch.no_grad():
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                    radii[visibility_filter])

                viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
                gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

                if iteration % opt.densification_interval == 0:
                    size_threshold = 20
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        if iteration % opt.DU_interval == 0:
            
            if not edit_stack:
                edit_stack = scene.getTrainCameras().copy()
            
            random_index = 0
            if iteration < opt.iterations + opt.editing_warmup_iter:
                random_index = first_idx
                edit_viewpoint_cam = edit_stack[random_index]
            else:
                random_index = randint(0, len(edit_stack) - 1)
                edit_viewpoint_cam = edit_stack.pop(random_index)

            if iteration >= opt.iterations + opt.editing_warmup_iter or is_first:
                edit_fid = edit_viewpoint_cam.fid
                N = gaussians.get_xyz.shape[0]
                edit_time_input = edit_fid.unsqueeze(0).expand(N, -1)
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), edit_time_input)

                render_pkg_re = render(edit_viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re[
                    "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
            
                src_image = edit_viewpoint_cam.original_image.cuda()


                # The rendering image from other frame
                if random_index-1 >= 0:
                    cam_ref_1 = edit_stack[random_index-1]
                else:
                    cam_ref_1 = edit_viewpoint_cam
                render_pkg_re_1 = render(cam_ref_1, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
                image_ref_1 = render_pkg_re_1["render"]

                if random_index-2 >= 0:
                    cam_ref_2 = edit_stack[random_index-2]
                else:
                    cam_ref_2 = edit_viewpoint_cam
                
                render_pkg_re_2 = render(cam_ref_2, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
                image_ref_2 = render_pkg_re_2["render"]

                if random_index-3 >= 0:
                    cam_ref_3 = edit_stack[random_index-3]
                else:
                    cam_ref_3 = edit_viewpoint_cam
                
                render_pkg_re_3 = render(cam_ref_3, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
                image_ref_3 = render_pkg_re_3["render"]


                # The src_image and image need to be expend as processed in the training stage.
                # You should simple concatenate the same image at this time for fast test.
                # 先在宽度方向拼接（从左到右）
                top_row = torch.cat((image, image_ref_1), dim=2)  
                bottom_row = torch.cat((image_ref_2, image_ref_3), dim=2)  

                # 再在高度方向拼接（从上到下）
                image_large = torch.cat((top_row, bottom_row), dim=1) 

                top_raw = torch.cat((src_image, cam_ref_1.original_image.cuda()), dim=2)
                bottom_raw = torch.cat((cam_ref_2.original_image.cuda(), cam_ref_3.original_image.cuda()), dim=2)
                src_image_large = torch.cat((top_raw, bottom_raw), dim=1)

                # [C, H, W]
                edited_image = Editing_module.editing(image_large.permute(1, 2, 0), prompt, src_image = src_image_large.permute(1, 2, 0) 
                                                      , resolution = 1024, diffusion_steps = 20, guidance_scale = 7.5, image_guidance_scale = 1.5).to(image)[0]

                # The edited_image will be split into 4 parts, each part is the same as the image.
                _, H_, W_ = edited_image.shape
                edited_image = edited_image[:, :H_ // 2, :W_ // 2]
                
                # 把edited_image 和 src image 结合。
                # if iteration < opt.iterations + opt.edit_iterations:
                edited_image_train_percent = args.edited_image_train_percent
                if args.edited_image_train and iteration >= (opt.iterations + opt.edit_iterations * (1-args.fine_tune_percent) \
                                                             + opt.edit_iterations * args.fine_tune_percent * (1 - edited_image_train_percent) + 1):
                    edited_image = edited_image
                else: 
                    edited_image = edited_image * edit_viewpoint_cam.original_mask.cuda() + src_image * (1 - edit_viewpoint_cam.original_mask.cuda())
                # else:
                #     print(f"Without mask for finally saving, iteration {iteration}")
                
                edit_image_dataset[edit_viewpoint_cam.uid] = edited_image.detach().cpu()

                if write_local_log and iteration % opt.local_vis_log_interval == 0:
                    vis_combined = torch.cat([src_image[None], edited_image[None], image[None]], dim = 0).float()
                    torchvision.utils.save_image(vis_combined, os.path.join(log_folder, str(iteration).zfill(6) + '.png'), nrow=vis_combined.shape[0])
                is_first = False
                
            if edit_viewpoint_cam not in edited_stack:
                edited_stack.append(edit_viewpoint_cam)

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.update(10)

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Optimizer step
            if iteration < opt.iterations + opt.edit_iterations:

                if iteration < opt.iterations + opt.editing_warmup_iter:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)    
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)
                    deform.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    deform.optimizer.zero_grad()
                    deform.update_learning_rate(iteration)
    
    print('Editing Finished!')

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False, is_edit=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)

                    if not is_edit:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.edit_image.to("cuda"), 0.0, 1.0)

                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6999)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 60001, 5000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--load_checkpoint', type=int, default=None)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--diffusion_model_checkpoint', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--write_local_log', action='store_true', default=False)
    parser.add_argument('--loss', type=str, default='two')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--mask_iter', type=int, default=5000)
    parser.add_argument('--all_mask_path', type=str, default=None, help="Path to save the all mask tensor")
    parser.add_argument('--fine_tune', type=bool, default=False)
    parser.add_argument('--fine_tune_percent', type=float, default=0.5, help="The percentage of fine-tuning iterations")
    parser.add_argument('--edited_image_train', type=bool, default=False)
    parser.add_argument('--edited_image_train_percent', type=float, default=0.3, help="The percentage of edited image training iterations")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    # print(args)
    # sys.exit()

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, \
             args.save_iterations, load_checkpoint=args.load_checkpoint, diffusion_ckpt=args.diffusion_model_checkpoint, \
            prompt=args.prompt, first_idx=args.idx, write_local_log=args.write_local_log, loss_selection=args.loss, acc_step=args.batch, args=args)

    # All done
    print("\nTraining complete.")
