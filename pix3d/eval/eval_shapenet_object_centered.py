import argparse
import os, sys
import numpy as np
import tensorflow as tf
from emd import tf_auctionmatch
from cd import tf_nndistance
import time

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

# data_path
data_dir = '../../data/ShapeNet/'
test_datapath = '../../data/test_list.txt'
# Load data
namelist = []
with open(test_datapath, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        namelist.append(line)
# eval_path
eval_path = '../../result/result_shapenet_ply_out_smooth_pt1024/'
eval_path2 = eval_path

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
class_name = {'02828884':'bench','03001627':'chair','03636649':'lamp','03691459':'speaker','04090263':'firearm','04379243':'table','04530566':'watercraft','02691156':'plane','02933112':'cabinet','02958343':'car','03211117':'monitor','04256520':'couch','04401088':'cellphone'}
model_number = {i:0 for i in class_name}
sum_f = {i:0 for i in class_name}
sum_cd = {i:0 for i in class_name}
sum_emd = {i:0 for i in class_name}
iters = 0
f_sum = 0.0
cd_sum = 0.0
emd_sum = 0.0
for file_list in namelist:

    iters += 1

    predict =  np.loadtxt(eval_path+file_list[19:-4]+'_pred.npy')
    label = np.loadtxt(eval_path2+file_list[19:-4]+'_gt.npy')

    file_list_sub = file_list.split("_")
    class_id = file_list_sub[0][19:]

    cd, emd = sess.run([chamfer_distance, emd_dist], feed_dict={xyz1:label,xyz2:predict})

    model_number[class_id] += 1.0
    sum_cd[class_id] += cd # cd is the mean of all distance
    sum_emd[class_id] += emd #emd[0] # emd is the sum of all distance
    cd_sum += cd
    emd_sum += emd #emd[0]

    print(iters, cd_sum/iters*10, emd_sum/iters*10)

for item in model_number:
    number = model_number[item] + 1e-8
    cd = (sum_cd[item] / number) *100#* 1000 #cd is the mean of all distance, cd is L2
    emd = (sum_emd[item] / number) *100 #* 0.01 #emd is the sum of all distance, emd is L1
    print(class_name[item], int(number), cd, emd)#, f)

print('Testing Finished!')
