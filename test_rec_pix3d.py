from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
print("PyTorch Version: ",torch.__version__)

from model_occ import *
import cv2
import scipy.io as sio

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
category = 'table' # chair, table, sofa

data_dir = './data/pix3d/'+category+'_proc/'

weight_path = './model/PointCloud_Reconstruction_resnet50_Complete_Silhouette_Guidance.pth'

save_path = './result_pix3d_fseg_ft/'+category+'/'

# Pre-trained models to choose from [resnet18, resnet34, resnet50]
model_name = "resnet50"

num_classes = 1024
grid_size = 2

print("Load Models...")
# Define the encoder
encoder = initialize_encoder(model_name, num_classes,use_pretrained=True)
# Full model
model_ft = OccNet(encoder, num_classes, grid_size)

save_dict = torch.load(weight_path)
model_ft.load_state_dict(save_dict['model_state_dict'])


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

# evaluation mode
model_ft.eval()

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

namelist = next(os.walk(data_dir+'/mask_gt/'))[2]
cnt = 0
loss_sum = 0.0
for file_list in namelist:
    
    #file_list = np.random.choice(namelist, 1)  
    #file_list = file_list[0]
    print(file_list)
    
    img = cv2.imread(data_dir+'/img/'+file_list)
    img = img.astype('float32')/255.0
    # visible silhouette
    #mask =cv2.imread(data_dir+'/mask/'+file_list)
    # ground truth complete silhouette
    #mask =cv2.imread(data_dir+'/mask_gt/'+file_list)
    # predicted complet silhoutte
    mask = sio.loadmat(data_dir+'/mask_pred_ft/'+file_list[:-4]+'.mat')
    mask = mask['pred']
    
    #mask = mask.astype('float32')/255.0
    #mask = mask[:,:,0]
    mask = np.expand_dims(mask,axis=2)

    mask = np.float32(mask>0.5)
    mask = torch.tensor(mask).to(device).float()
    img = torch.tensor(img).to(device).float()
    
    img = torch.cat((img, mask),2)
    
    inputs = img.permute(2,0,1)
    inputs = inputs.unsqueeze(0)
    
    outputs_r, outputs_f = model_ft(inputs)
    
    inputs = inputs.squeeze(0)
    inputs = inputs.permute(1,2,0)

    outputs_f = outputs_f.squeeze(0)
    outputs_r = outputs_r.squeeze(0)

    # save
    np.savetxt(save_path+file_list[:-3]+'xyz', outputs_f.data.cpu().numpy())
    
    cnt += 1






