## 3D-HGS: 3D Half-Gaussian Splatting <br><sub>Official PyTorch Implementation</sub> 

### [[Arxiv]](https://arxiv.org/abs/2406.02720) 

This repo contains the official implementation for the paper "3D-HGS: 3D-HGS: 3D Half-Gaussian Splatting". Our work proposes to employ 3D Half-Gaussian(3D-HGS) kernels, which can be used as a plug-and-play kernel for Gaussian Splatting-related works. Our experiments demonstrate their capability to improve the performance of current 3D-GS related methods and achieve state-of-the-art rendering performance on various datasets without compromising rendering speed.

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

## Step-by-step Tutorial
Video may comimg in the future

## Colab
May coming in the future

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# HTTPS
git clone https://github.com/lihaolin88/3D-Half-Gaussian-Splatting.git
```

## Overview

The codebase has 2 main components:
- A PyTorch-based optimizer to produce a 3D half Gaussian model from SfM inputs, and almost all parameter settings will keep same as [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting).
- A half gaussain rasterizer, which can also been used as a plug-and-play core for other gaussian splatting task; to use it as a plug and play, the only thing you need to do is modify the opacity to two values for each gaussian and active the normal in your python code then use our half-gaussian rasterizer.
- A rendering method can help you generate the final rendering result and show the score.

## setup
We provide conda install instructions in this repo:
```shell
conda env create --file environment.yml
conda activate half_gaussian_splatting
```
This environment will automatically use the half gaussian rasterizer. if the rasterizer not install right, please manually install the half gaussian rasterizer:
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
