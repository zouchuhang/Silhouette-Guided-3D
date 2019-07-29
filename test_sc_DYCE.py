from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
print("PyTorch Version: ",torch.__version__)

import os
import scipy.io as sio

import cv2

from model_sc import *

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure

weight_path = "./model/Silhouette_Completion_DYCE_resnet50.pth"
test_datapath = './data/DYCE/test/'
save_path = './result_sc_DYCE/'

# Pre-trained models to choose from [resnet18, resnet34, resnet50]
model_name = "resnet50"
#model_name = "resnet18"

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
namelist = next(os.walk(test_datapath+'/image/'))[2]

criterion = nn.BCELoss()

cnt = 0
loss_f_sum = 0.0
loss_v_sum = 0.0
loss_i_sum = 0.0
imsize = 256

for file_list in namelist:
    
    #file_list = np.random.choice(namelist, 1)  
    #file_list = file_list[0]

    im_path = test_datapath+'/image/'+file_list
    img = cv2.imread(im_path)
    img = img.astype('float32')/255.0
    mask_path = test_datapath+'/modal/'+file_list
    mask = cv2.imread(mask_path)
    mask = mask.astype('float32')/255.0
    mask = mask[:,:,0]
    label_path = test_datapath+'/amodal/'+file_list
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
    outputs = cv2.resize(outputs.data.cpu().numpy(), (imsize, imsize))
    outputs = outputs*box
    label = label*box
    
    # full loss
    loss_f = np.sum((outputs>0.5)&(label>0.5))/(np.sum((outputs>0.5)|(label>0.5))+np.finfo(float).eps)
    mask_i = (label-mask_v)>0.5
    # visible
    loss_v = np.sum((outputs>0.5)&(1-mask_i)&(mask_v>0.5))/(np.sum(((outputs>0.5)&(1-mask_i))|(mask_v>0.5))+np.finfo(float).eps)
    # invisible
    loss_i = np.sum((outputs>0.5)&(1-(mask_v>0.5))&mask_i)/(np.sum((outputs>0.5)&(1-(mask_v>0.5))|mask_i)+np.finfo(float).eps)
    
    loss_f_sum += loss_f
    loss_v_sum += loss_v
    loss_i_sum += loss_i
    
    # save
#    sio.savemat(save_path+file_list[:-3]+'mat',{'image':inputs.data.cpu().numpy(), 'gt':label, 'pred':outputs, 'box': box, 'mask': mask_v})

    cnt += 1
    print('No. {}, Loss f: {:.6f}, Loss v: {:.6f}, Loss i: {:.6f}'.format(cnt,loss_f_sum/cnt, loss_v_sum/cnt, loss_i_sum/cnt))


print('Total number: {}, Avg CD Loss: {:.6f}'.format(cnt,loss_sum/cnt))
