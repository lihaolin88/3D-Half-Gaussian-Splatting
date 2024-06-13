The dataset can be download through:
https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
https://jonbarron.info/mipnerf360/
http://visual.cs.ucl.ac.uk/pubs/deepblending/

Set up the environment
1. conda install with environment.yml
2. if the submodule not install correctly, to install submodels, please use: 
	pip install ./submodules/diff-gaussian-rasterization
	pip install ./submodules/simple-knn

To run the code
For Train:
with command: 
	python train.py -s /the/path/of/dataset

For inference:
	python test_and_score.py -s /the/path/of/test_data -m /the/path/of/trained_result_folder
The inference will save the render and ground truth images, the code will also show the PSNR, SSIM and LPIPS for each scene