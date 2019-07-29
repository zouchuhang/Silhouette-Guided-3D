import argparse
import glob, os
import random
import re
import sys
import time

from os.path import join

import cv2
import numpy as np
import tensorflow as tf

from utils.metrics_utils import *

random.seed(1024)
np.random.seed(1024)
tf.set_random_seed(1024)

from utils.icp_zou import icp
from tqdm import tqdm
import scipy.io as sio

# Create Placeholders
xyz1=tf.placeholder(tf.float32,shape=(None, 3))
xyz2=tf.placeholder(tf.float32,shape=(None, 3))
xyz3 = tf.expand_dims(xyz1, 0)
xyz4 = tf.expand_dims(xyz2, 0)
xyz3_scaled, xyz4_scaled = scale(xyz3, xyz4)
xyz3_scaleds = tf.squeeze(xyz3_scaled)
xyz4_scaleds = tf.squeeze(xyz4_scaled)

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# load predictions
# data_path
category = 'chair'
data_dir = '/data/czou4/ShapeNet/'
eval_path = '/data/czou4/pix3d/result_pix3d_predfseg_ft_ply_out_smooth/'+category+'/'
gt_path = '../data/pix3d/'+category+'_proc/gt/'
# Load data

iters = 0
f_sum = 0.0
cd_sum = 0.0
emd_sum = 0.0

namelist = next(os.walk(gt_path))[2]

for file_list in namelist:
    if file_list.endswith(".npy"):
        continue
    iters += 1

    pcl_pred = np.loadtxt(eval_path+file_list[:-4]+'-clean.xyz')
    print(eval_path+file_list)

    # load gt
    pcl_gt = np.loadtxt(gt_path+file_list)

    # Perform Scaling
    pcl_gt_scaled, pcl_pred_scaled = sess.run([xyz3_scaleds, xyz4_scaleds], feed_dict={xyz1:pcl_gt,xyz2:pcl_pred})
    T, _, _ = icp(pcl_gt_scaled,pcl_pred_scaled, tolerance=1e-10, max_iterations=1000)

    # no translation
    pcl_pred_icp = pcl_pred_scaled - T[:3, 3]

    # save
    np.savetxt(eval_path+file_list[:-4]+'_pred.npy', pcl_pred_icp)
    np.savetxt(gt_path+file_list[:-4]+'_gt.npy', pcl_gt_scaled)
