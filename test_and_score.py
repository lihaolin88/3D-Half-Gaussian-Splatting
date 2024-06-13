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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
import lpips
loss_fn_alex = lpips.LPIPS(net='vgg')
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_normal")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)

    psnr_score = 0
    ssim_score = 0
    lpip_score = 0

    loss_fn_alex.to(gaussians.get_features.device)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        image = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]

        psnr_score += psnr(image, gt).mean().double()
        ssim_score += ssim(image, gt)
        img1 = image.unsqueeze(0)
        img2 = gt.unsqueeze(0)
        lpip_score += loss_fn_alex.forward(img1, img2).squeeze()  # .to('cpu')#.cpu()

        torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    print("PSNR=", psnr_score / len(views))
    print("SSIM=", ssim_score / len(views))
    print("LPIP=", lpip_score / len(views))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, ignore_points=True)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)