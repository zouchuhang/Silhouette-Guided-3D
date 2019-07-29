from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
print("PyTorch Version: ",torch.__version__)
import pickle
import os

from model import *

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "./data/ShapeNet/"

# model path
weight_path = './model/PointCloud_Reconstruction_resnet50_ShapeNet.pth' # this one

test_datapath = './data/test_list.txt'
save_path = './result/'

# Pre-trained models to choose from [resnet18, resnet34, resnet50]
model_name = "resnet50"

num_classes = 1024
grid_size = 2

print("Load Models...")
# Define the encoder
encoder = initialize_encoder(model_name, num_classes,use_pretrained=True)
# Full model
model_ft = OccNet(encoder, num_classes, grid_size)
model_ft.load_state_dict(torch.load(weight_path))

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

# evaluation mode
model_ft.eval()

# Load data
namelist = []
with open(test_datapath, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        namelist.append(line)

def pairwise_distances(a, b):
    # single batch case
    x,y = a,b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind_x = torch.arange(0, x.shape[1]).type(torch.cuda.LongTensor)
    diag_ind_y = torch.arange(0, y.shape[1]).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1)
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1)
    P = (rx.transpose(2,1).expand_as(zz) + ry.expand_as(zz) - 2*zz)
    return P.min(1)[0], P.min(2)[0]

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

class_name = {'02828884':'bench','03001627':'chair','03636649':'lamp','03691459':'speaker','04090263':'firearm','04379243':'table','04530566':'watercraft','02691156':'plane','02933112':'cabinet','02958343':'car','03211117':'monitor','04256520':'couch','04401088':'cellphone'}
sum_cd = {i:0 for i in class_name}
sum_fs = {i:0 for i in class_name}
model_number = {i:0 for i in class_name}

cnt = 0
loss_sum = 0.0
f_sum = 0.0
for file_list in namelist:

    pkl_path = data_dir+file_list
    file_list_sub = file_list.split("_")
    clsid = file_list_sub[0][19:]
    
    pkl = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
    img = pkl[0].astype('float32')/255.0
    labels = pkl[1][:,:3]
    img = torch.tensor(img).to(device).float()
    
    labels = torch.tensor(labels).to(device).float()
    inputs = img.permute(2,0,1)
    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)
        
    _, outputs_f = model_ft(inputs)
        
    dist1, dist2 = pairwise_distances(outputs_f, labels)

    loss = (torch.mean(dist1))+(torch.mean(dist2))
    fs = f_score(labels[0],outputs_f[0],dist1,dist2,[0.0001, 0.0002])

    loss_sum += loss.item()
    f_sum += fs
    # Note that the CD here is not the actuall CD since we do not sample 2466 points as in Pixel2Mesh
    # just for sanity check
    sum_cd[clsid] += loss.item()
    sum_fs[clsid] += fs
    model_number[clsid] += 1.0

    outputs_f = outputs_f.squeeze(0)
    
    # save prediction
    np.savetxt(save_path+file_list[19:-3]+'xyz', outputs_f.data.cpu().numpy())

    cnt += 1
    print(f_sum/cnt)
    print('No. {}, CD Loss: {:.6f}'.format(cnt,loss_sum/cnt))


print('Total number: {}, Avg CD Loss: {:.6f}'.format(cnt,loss_sum/cnt))
#each class:
for item in model_number:
    number = model_number[item] + 1e-8
    cd = (sum_cd[item] / number) * 1000
    f = sum_fs[item] / number
    print(class_name[item], int(number), cd, f)
