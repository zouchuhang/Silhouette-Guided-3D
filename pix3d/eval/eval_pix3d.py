import argparse
import os, sys
import numpy as np
import tensorflow as tf

from emd import tf_auctionmatch
from cd import tf_nndistance
import time
import scipy.io as sio

def f_score(label, predict, dist_label, dist_pred, threshold):
    num_label = label.shape[0]
    num_predict = predict.shape[0]

    f_scores = []
    for i in range(len(threshold)):
        num = len(np.where(dist_label <= threshold[i])[0])
        recall = 100.0 * num / num_label
        num = len(np.where(dist_pred <= threshold[i])[0])
        precision = 100.0 * num / num_predict

        f_scores.append((2*precision*recall)/(precision+recall+1e-8))
    return np.array(f_scores)

category = 'chair'
# data_path
eval_path = '/data/czou4/pix3d/result_pix3d_predfseg_ft_ply_out_smooth/'+category+'/'
#eval_path = '../../data/pix3d/result_pix3d_predvseg_ply_out_smooth/'+category+'/'
#eval_path = '../../data/pix3d/result_pix3d_gtfseg_ply_out_smooth/'+category+'/'
#eval_path = '../../data/pix3d/result_pix3d_noseg_ply_out_smooth/'+category+'/'
gt_path = '../../data/pix3d/'+category+'_proc/gt/'

test_list = '../../data/pix3d/list/'+category+'_occ_all.txt'
occlist = []
with open(test_list, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        occlist.append(line)

test_list = '../../data/pix3d/list/'+category+'_socc_all.txt'
trunclist = []
with open(test_list, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        trunclist.append(line)

# Load data
namelist = next(os.walk(gt_path))[2]

def get_chamfer_metrics(pcl_gt, pred):
    '''
    Calculate chamfer, forward, backward distance between ground truth and predicted
    point clouds. They may or may not be scaled.
    Args:
        pcl_gt: tf placeholder of shape (B,N,3) corresponding to GT point cloud
        pred: tensor of shape (B,N,3) corresponding to predicted point cloud
    Returns:
        Fwd, Bwd, Chamfer: (B,)
    '''
    #(B, NUM_POINTS) ==> ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) for nn pair of points (x1,y1,z1) <--> (x2, y2, z2)
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pcl_gt, pred)
    dists_forward = tf.reduce_mean(tf.sqrt(dists_forward), axis=1) # (B, )
    dists_backward = tf.reduce_mean(tf.sqrt(dists_backward), axis=1) # (B, )
    chamfer_distance = dists_backward + dists_forward
    return dists_forward, dists_backward, chamfer_distance

def get_emd_metrics(pcl_gt, pred, batch_size, num_points):
    '''
    Calculate emd between ground truth and predicted point clouds. 
    They may or may not be scaled. GT and pred need to be of the same dimension.
    Args:
        pcl_gt: tf placeholder of shape (B,N,3) corresponding to GT point cloud
        pred: tensor of shape (B,N,3) corresponding to predicted point cloud
    Returns:
        emd: (B,)
    '''
    X,_ = tf.meshgrid(tf.range(batch_size), tf.range(num_points), indexing='ij')
    ind, _ = tf_auctionmatch.auction_match(pred, pcl_gt) # Ind corresponds to points in pcl_gt
    ind = tf.stack((X, ind), -1)
    emd = tf.reduce_mean(tf.sqrt(tf.reduce_sum((tf.gather_nd(pcl_gt, ind) - pred)**2, axis=-1)), axis=1) # (B, )
    return emd

BATCH_SIZE = 1
NUM_EVAL_POINTS = 1024

# Initialize session
# xyz1:dataset_points * 3, xyz2:query_points * 3
xyz1=tf.placeholder(tf.float32,shape=(None, 3))
xyz2=tf.placeholder(tf.float32,shape=(None, 3))
xyz3 = tf.expand_dims(xyz1, 0)
xyz4 = tf.expand_dims(xyz2, 0)
# chamfer distance
dists_forward, dists_backward, chamfer_distance = get_chamfer_metrics(xyz3, xyz4)
emd_dist = get_emd_metrics(xyz3, xyz4, BATCH_SIZE, NUM_EVAL_POINTS)
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
config.allow_soft_placement=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

###
iters = 0
cnt_trunc = 0
cnt_occ = 0
cnt = 0
cd_sum = 0.0
cd_occ_sum = 0.0
cd_trunc_sum = 0.0
emd_sum = 0.0
emd_occ_sum = 0.0
emd_trunc_sum = 0.0

for file_list in namelist:
    if file_list.endswith(".xyz"):
        continue
    iters += 1
    print(file_list)

    predict =  np.loadtxt(eval_path+file_list[:-7]+'_pred.npy')
    label = np.loadtxt(gt_path+file_list)

    cd, emd = sess.run([chamfer_distance, emd_dist], feed_dict={xyz1:label,xyz2:predict})

    if file_list[:-7]+'.mat' in occlist:
        cnt_occ+=1
        cd_occ_sum += cd
        emd_occ_sum += emd
    elif file_list[:-7]+'.mat' in trunclist:
        cnt_trunc+=1
        cd_trunc_sum += cd
        emd_trunc_sum += emd

    cnt+=1
    cd_sum += cd
    emd_sum += emd
    
print(cnt, cd_sum/cnt*100, emd_sum/cnt*100)
print(cnt_occ, cd_occ_sum/cnt_occ*100, emd_occ_sum/cnt_occ*100)
print(cnt_trunc, cd_trunc_sum/cnt_trunc*100, emd_trunc_sum/cnt_trunc*100)

print('Testing Finished!')
