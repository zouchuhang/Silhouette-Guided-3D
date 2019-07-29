# Silhouette-Guided-3D
PyTorch implementation for [paper](): "Silhouette Guided Point Cloud Reconstruction beyond Occlusion"

<img src='figs/fig-overview.jpg' width=400>

Network architecture:

<img src='figs/fig-network.jpg' width=700>

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
- Under ./matlab folder, install [gptoolbox](https://github.com/alecjacobson/gptoolbox) for Matlab based Poisson-Disc Sampling

- [Optional] Install [Pix3D evaluation toolkit](https://github.com/xingyuansun/pix3d) under the current folder. Note that this requires Tensorflow.
- [Optional] Install [PCN evaluation toolkit](https://github.com/TonythePlaneswalker/pcn) under the current folder. Note that this requires Tensorflow. PCN toolkit is for object-centered evaluation.
- [Optional] Install [Mask-RCNN benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) under the current folder. This is for getting visible silhouette for completion in Pix3D (We've included pre-processed results as below).

## Download Data and Pre-trained Model
- Download [pre-trained models](https://drive.google.com/file/d/1KjmNb1TuIALyiKXNsEQCbp7ow9kP_ENB/view?usp=sharing) and put them under the ./model/ folder.
- Download [pre-processed DYCE dataset](https://drive.google.com/file/d/14sa6p3f-wT1SFL1tZlOPMe63N2dntHEG/view?usp=sharing) and put them under the ./data/ folder.
- Download [pre-processed Pix3D dataset](https://drive.google.com/file/d/1DdcDpePJ-t19SBLRuu0LSK5mNCeB1iUJ/view?usp=sharing) and put them under the ./data/ folder. This includes pre-computed complete silhouette and ground truth point clouds rotated w.r.t. camera position.
- Download [ShapeNet dataset](https://drive.google.com/drive/folders/131dH36qXCabym1JjSmEpSQZg4dmZVQid) and put them under the ./data/ folder.
- Download [pre-processed LSUN dataset](https://drive.google.com/file/d/1L7MrNuwYo7-e-adCHJ-S4d4u-_-4JMpS/view?usp=sharing) and put them under the ./data/ folder
- Download [pre-computed result](). and put them under the ./result/ folder. This includes point clouds prediction on ShapeNet and Pix3D after FSSR refinement.

## Training
- Point cloud reconstruction
    ```
    python train.py
    ```
    - FSSR post-refinement:

- Silhouette completion
    First train on DYCE dataset:
    ```
    python train_sc.py
    ```
    - Then finetune on Pix3D dataset, using 5-fold cross validation ( you will need to run it 5 times by changing the fold number in L32-35 ):
    ```
    python train_sc_ft.py
    ```

- Silhouette guidede point cloud reconstruction
    ```
    python train_occ.py
    ```
    - FSSR post-refinement:

## Evaluation
- ShapeNet
    - You need to use TensorFlow 3.0+ to run the evaluation:
    ```
    cd pcn
    python eval_shapenet.py
    ```
- Pix3D
    - You need to use TensorFlow 3.0+ to run the evaluation:
    ```
    cd pix3d/eval/
    python eval_pix3d.py
    ```

