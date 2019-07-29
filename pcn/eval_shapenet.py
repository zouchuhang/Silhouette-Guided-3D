import os, sys
import numpy as np
import tensorflow as tf
from pc_distance import tf_nndistance, tf_approxmatch
import pickle

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
data_dir = '../data/ShapeNet/'
test_datapath = '../data/test_list.txt'
# Load data
namelist = []
with open(test_datapath, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        namelist.append(line)
# eval_path
eval_path = '../result/result_shapenet_ply_out_smooth_pt2466/'

# Initialize session
# xyz1:dataset_points * 3, xyz2:query_points * 3
xyz1=tf.placeholder(tf.float32,shape=(None, 3))
xyz2=tf.placeholder(tf.float32,shape=(None, 3))
xyz3 = tf.expand_dims(xyz1, 0)
xyz4 = tf.expand_dims(xyz2, 0)
# chamfer distance
dist1,idx1,dist2,idx2 = tf_nndistance.nn_distance(xyz3, xyz4)
# earth mover distance, notice that emd_dist return the sum of all distance
match = tf_approxmatch.approx_match(xyz3, xyz4)
emd_dist = tf_approxmatch.match_cost(xyz3, xyz4, match)

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
res = np.zeros((len(namelist),1))
for file_list in namelist:
    
    # load predictions
    if not os.path.isfile(eval_path+file_list[19:-4]+'-clean.xyz'):
        continue
    iters += 1
    print(file_list)
     
    predict = np.loadtxt(eval_path+file_list[19:-4]+'-clean.xyz')
    # load gt
    pkl_path = data_dir+file_list
    pkl = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
     
    label = pkl[1][:,:3]

    file_list_sub = file_list.split("_")
    class_id = file_list_sub[0][19:]

    predict = np.concatenate((np.expand_dims(predict[:,0], axis=1), np.expand_dims(predict[:,2], axis=1), np.expand_dims(-1*predict[:,1], axis=1)), axis=1)
    
    # chamfer distance
    d1,i1,d2,i2,emd = sess.run([dist1,idx1,dist2,idx2, emd_dist], feed_dict={xyz1:label,xyz2:predict})
    cd = np.mean(d1) + np.mean(d2) 

    fs = f_score(label,predict,d1,d2,[0.0001, 0.0002])
    model_number[class_id] += 1.0

    sum_f[class_id] += fs #f_score(label,predict,d1,d2,[0.0001, 0.0002])
    sum_cd[class_id] += cd # cd is the mean of all distance
    sum_emd[class_id] += emd[0] # emd is the sum of all distance

    f_sum += fs
    cd_sum += cd
    emd_sum += emd[0]

    print(iters, f_sum/iters, cd_sum/iters, emd_sum/iters)

for item in model_number:
    number = model_number[item] + 1e-8
    f = sum_f[item] / number
    cd = (sum_cd[item] / number) * 1000 #cd is the mean of all distance, cd is L2
    emd = (sum_emd[item] / number) * 0.01 #emd is the sum of all distance, emd is L1
    print(class_name[item], int(number), cd, emd, f)

print('Testing Finished!')
