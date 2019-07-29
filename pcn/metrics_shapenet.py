import argparse
import os
import random
import re
import sys
import time

from os.path import join

import cv2
import numpy as np
import tensorflow as tf

import sklearn

from utils.metrics_utils import *

random.seed(1024)
np.random.seed(1024)
tf.set_random_seed(1024)

from utils.icp import icp
from tqdm import tqdm

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
data_dir = '../../data/ShapeNet/'
test_datapath = '../../data/test_list.txt'
eval_path = '../../result/result_shapenet_ply_out_smooth_pt1024/'
rendering_path = '../../data/ShapeNet/ShapeNetRendering/'
# Load data
namelist = []
with open(test_datapath, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        namelist.append(line)

class_name = {'02828884':'bench','03001627':'chair','03636649':'lamp','03691459':'speaker','04090263':'firearm','04379243':'table','04530566':'watercraft','02691156':'plane','02933112':'cabinet','02958343':'car','03211117':'monitor','04256520':'couch','04401088':'cellphone'}
model_number = {i:0 for i in class_name}
sum_f = {i:0 for i in class_name}
sum_cd = {i:0 for i in class_name}
sum_emd = {i:0 for i in class_name}
iters = 0
f_sum = 0.0
cd_sum = 0.0
emd_sum = 0.0

def camera_info(param):
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])

    camY = param[3]*np.sin(phi)
    temp = param[3]*np.cos(phi)
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0,1,0])
    axisX = np.cross(axisY, axisZ)
    axisY = np.cross(axisZ, axisX)

    cam_mat = np.array([axisX, axisY, axisZ])
    cam_mat = sklearn.preprocessing.normalize(cam_mat, axis=1)
    return cam_mat, cam_pos

for file_list in namelist:
    iters += 1
    print(iters)

    if os.path.isfile(eval_path+file_list[19:-4]+'_pred.npy'):
        continue
    pcl_pred = np.loadtxt(eval_path+file_list[19:-4]+'-clean.xyz')
    pcl_pred = np.concatenate((np.expand_dims(pcl_pred[:,0], axis=1), np.expand_dims(pcl_pred[:,2], axis=1), np.expand_dims(-1*pcl_pred[:,1], axis=1)), axis=1)
    print(eval_path+file_list)

    file_list_sub = file_list.split("_")
    class_id = file_list_sub[0][19:]
    # rotate back
    view_path = rendering_path+class_id+'/'+file_list_sub[1]+'/rendering/rendering_metadata.txt'
    cam_params = np.loadtxt(view_path)
    idx = int(float(file_list_sub[2][:-4]))
    cam_params = cam_params[idx]
    cam_mat, cam_pos = camera_info(cam_params)
    pcl_pred = np.dot(pcl_pred, cam_mat)+cam_pos
    x = np.expand_dims(pcl_pred[:,0], axis=1)
    y = np.expand_dims(pcl_pred[:,1], axis=1)
    z = np.expand_dims(pcl_pred[:,2], axis=1)
    pcl_pred = np.concatenate((-y, -z, x), axis=1)
    pcl_pred = pcl_pred/0.57
    # load gt
    gt_path = data_dir+'ShapeNet_pointclouds/'+class_id+'/'+file_list_sub[1]+'/pointcloud_1024.npy'
    pcl_gt = np.load(gt_path)
    
    # Perform Scaling
    pcl_gt_scaled, pcl_pred_scaled = sess.run([xyz3_scaleds, xyz4_scaleds], feed_dict={xyz1:pcl_gt,xyz2:pcl_pred})

    T, _, _ = icp(pcl_gt_scaled,pcl_pred_scaled, tolerance=1e-10, max_iterations=1000)
    pcl_pred_icp = np.matmul(pcl_pred_scaled, T[:3,:3]) - T[:3, 3]

    np.savetxt(eval_path+file_list[19:-4]+'_pred.npy', pcl_pred_icp)
    np.savetxt(eval_path+file_list[19:-4]+'_gt.npy', pcl_gt_scaled)
