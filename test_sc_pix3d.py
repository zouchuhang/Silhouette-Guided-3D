from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
print("PyTorch Version: ",torch.__version__)

import pickle
import os
import scipy.io as sio

import cv2

from model_sc import *

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure

category = 'table'

#weight_path = "./model/Silhouette_Completion_DYCE_resnet50.pth"
weight_path = "./model/Silhouette_Completion_Pix3D_fold1.pth"

test_datapath = './data/pix3d/'+category+'_proc/'

save_path = test_datapath+'mask_pred_ft/'

test_list = './data/pix3d/list/'+category+'_test_fold1.txt'

occlist = []
with open(test_list, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        occlist.append(line)

# Pre-trained models to choose from [resnet18, resnet34, resnet50]
model_name = "resnet50"

num_classes = 1024

print("Load Models...")
# Define the encoder
encoder = initialize_encoder(model_name, num_classes,use_pretrained=True)
# Full model
model_ft = SegNet(encoder, num_classes)
model_ft.load_state_dict(torch.load(weight_path))

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

# evaluation mode
model_ft.eval()

# Load data
namelist = next(os.walk(test_datapath+'/mask_gt/'))[2]

criterion = nn.BCELoss()

cnt = 0
cnt_occ = 0
loss_f_sum = 0.0
loss_f_occ_sum = 0.0
imsize = 256

for file_list in namelist:
    
    #file_list = np.random.choice(namelist, 1)  
    #file_list = file_list[0]
    print(file_list)
    im_path = test_datapath+'/img/'+file_list
    img = cv2.imread(im_path)
    img = img.astype('float32')/255.0
    mask_path = test_datapath+'/mask/'+file_list
    mask = cv2.imread(mask_path)
    mask = mask.astype('float32')/255.0
    mask = mask[:,:,0]
    label_path = test_datapath+'/mask_gt/'+file_list
    label = cv2.imread(label_path)
    label = label.astype('float32')/255.0
    label = label[:,:,0]
    box_path = test_datapath+'/box/'+file_list
    box = cv2.imread(box_path)
    box = box.astype('float32')/255.0
    box = box[:,:,0]
    
    # reshape
    img = cv2.resize(img, (224, 224))
    np.clip(img, 0.0, 1.0 , out=img)
    label = cv2.resize(label, (imsize, imsize))
    mask_v = cv2.resize(mask, (imsize, imsize))
    mask_v = np.float32(mask_v>0.5)
    mask = cv2.resize(mask, (224, 224))
    mask = np.float32(mask>0.5)
    label = np.float32(label>0.5)
    box = cv2.resize(box, (imsize, imsize))
    box = np.float32(box>0.5)

    image = torch.tensor(img).to(device).float()
    masks = torch.tensor(mask).to(device).float()     

    inputs = image.permute(2,0,1)
    inputs = inputs.unsqueeze(0)

    masks = masks.unsqueeze(0)
    masks = masks.unsqueeze(1)
    inputs = torch.cat((inputs,masks),1)
    
    outputs = model_ft(inputs)
    outputs = outputs.squeeze(0).squeeze(0)
    outputs_f = outputs.data.cpu().numpy()
    outputs_f = outputs_f>0.5
    outputs = cv2.resize(outputs.data.cpu().numpy(), (imsize, imsize))
    
    outputs = outputs*box
    label = label*box

    # uncomment if test on mask rcnn
    #outputs = mask_v

    # full loss
    loss_f = np.sum((outputs>0.5)&(label>0.5))/(np.sum((outputs>0.5)|(label>0.5))+np.finfo(float).eps)
    
    if file_list[:-3]+'mat' in occlist:
        loss_f_occ_sum += loss_f
        cnt_occ +=1
        # save
        sio.savemat(save_path+file_list[:-3]+'mat',{'image':inputs.data.cpu().numpy(), 'pred':outputs_f})

    loss_f_sum += loss_f
    cnt+=1

    inputs = inputs.squeeze(0)
    inputs = inputs.permute(1,2,0)
    
    print('No. {}, Loss f: {:.6f}, Loss f occ: {:.6f}'.format(cnt+cnt_occ,loss_f_sum/(cnt+np.finfo(float).eps), loss_f_occ_sum/(cnt_occ+np.finfo(float).eps)))


print(cnt)
print(cnt_occ)
print('Total number: {}, Avg CD Loss: {:.6f}'.format(cnt,loss_f_sum/cnt))
