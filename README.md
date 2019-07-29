# Silhouette-Guided-3D
PyTorch implementation for [paper](): "Silhouette Guided Point Cloud Reconstruction beyond Occlusion"

<img src='figs/fig-overview.jpg' width=400>

## Requirements
- Python 3
- PyTorch >= 0.4.0
- numpy
- scipy
- pickle
- skimage
- random
- re
- torchvision
- Matlab (will be transferred to python later)

## Installation
- Install [mve](https://github.com/simonfuhrmann/mve) by following the instructions. This is for FSSR based point cloud refinement.
- [Optional] Install [Pix3D evaluation toolkit](https://github.com/xingyuansun/pix3d) under the current folder. Note that this requires Tensorflow.
- [Optional] Install [PCN evaluation toolkit](https://github.com/TonythePlaneswalker/pcn) under the current folder. Note that this requires Tensorflow. PCN toolkit is for object-centered evaluation.
- [Optional] Install [Mask-RCNN benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) under the current folder. This is for getting visible silhouette for completion in Pix3D (We've included pre-processed results as below).

## Download Data and Pre-trained Model
- Download [pre-trained models](https://drive.google.com/file/d/1KjmNb1TuIALyiKXNsEQCbp7ow9kP_ENB/view?usp=sharing) and put them under the ./model/ folder.
- Download [pre-processed DYCE dataset](https://drive.google.com/file/d/14sa6p3f-wT1SFL1tZlOPMe63N2dntHEG/view?usp=sharing) and put them under the ./data/ folder.
- Download [pre-processed Pix3D dataset](https://drive.google.com/file/d/1DdcDpePJ-t19SBLRuu0LSK5mNCeB1iUJ/view?usp=sharing) and put them under the ./data/ folder. This includes pre-computed complete silhouette and ground truth point clouds rotated w.r.t. camera position.
- Download [ShapeNet dataset](https://drive.google.com/drive/folders/131dH36qXCabym1JjSmEpSQZg4dmZVQid) and put them under the ./data/ folder.
- Download [pre-processed LSUN dataset](https://drive.google.com/file/d/1L7MrNuwYo7-e-adCHJ-S4d4u-_-4JMpS/view?usp=sharing) and put them under the ./data/ folder
- Download [pre-computed result](). and put them under the current folder. This includes point clouds prediction on ShapeNet and Pix3D after FSSR refinement.s

## Training
- Point cloud reconstruction
- Silhouette completion
- Silhouette guidede point cloud reconstruction

## Evaluation




