## 3D-HGS: 3D Half-Gaussian Splatting <br><sub>Official PyTorch Implementation</sub> 

### [[Paper]](https://arxiv.org/abs/2406.02720)  [[Scaffold_HGS]](https://drive.google.com/file/d/1YeyAV2D9E3zGmxkCQwV42FrIFxsew_bE/view?usp=sharing)

This repo contains the official implementation for the paper "3D-HGS: 3D-HGS: 3D Half-Gaussian Splatting". Our work proposes to employ 3D Half-Gaussian(3D-HGS) kernels, which can be used as a plug-and-play kernel for Gaussian Splatting-related works. Our experiments demonstrate their capability to improve the performance of current 3D-GS related methods and achieve state-of-the-art rendering performance on various datasets without compromising rendering speed.

<img width="1200" alt="3DHGS" src="https://github.com/lihaolin88/3D-Half-Gaussian-Splatting/assets/50398783/66948147-5ef4-49b8-bd30-01082702e39f">

## Update
07/31/2024: there is a bug in ./scene/gaussian_splatting.py line 193, it will switch the network to finetune mode and decrease the performance if you train from scratch. We fix it today.

## Step-by-step Tutorial
Video may comimg in the future

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# HTTPS
git clone https://github.com/lihaolin88/3D-Half-Gaussian-Splatting.git
```

## Overview

The codebase has 2 main components:
- A PyTorch-based optimizer, this component generates a 3D half Gaussian model from Structure from Motion (SfM) inputs, most parameter settings align with those used in [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting).
- A half gaussain rasterizer, The half Gaussian rasterizer can be used as a plug-and-play core for Gaussian splatting tasks by adjusting the opacity to two values for each Gaussian and activating the normal in your Python code.

## Setup
We provide conda install instructions in this repo:
```shell
conda env create --file environment.yml
conda activate half_gaussian_splatting
```
This environment will automatically use the half Gaussian rasterizer. If the rasterizer is not installed correctly, please manually install the half Gaussian rasterizer:
```shell
conda activate half_gaussian_splatting
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## Running
To Train with our code (you can directly use the dataset download from the link, but don't forget to split the train and test set by yourself):
```shell
python train.py -s /the/path/of/dataset
```
To test with our code and generate score:
```shell
python test_and_score.py -s /the/path/of/test_data -m /the/path/of/trained_result_folder
```
The inference will save the render and ground truth images, the code will also show the PSNR, SSIM and LPIPS for each scene

## BibTeX
If you find our paper/project useful, please consider citing our paper:
```bibtex
@article{li20243d,
  title={3D-HGS: 3D Half-Gaussian Splatting},
  author={Li, Haolin and Liu, Jinyang and Sznaier, Mario and Camps, Octavia},
  journal={arXiv preprint arXiv:2406.02720},
  year={2024}
}
```
