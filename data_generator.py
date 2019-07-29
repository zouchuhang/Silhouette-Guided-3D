from __future__ import print_function
from __future__ import division
import numpy as np
#import torch
from torchvision import transforms
import time
import os
import pickle
from torch.utils import data
import scipy.ndimage

import random
from skimage import exposure

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}

# Data generator
class ShapeNetDataset(data.Dataset):
    def __init__(self, file_list, root_dir, train_type, grid_size=2, transform=None):
        self.namelist = []
        with open(file_list, 'r') as f:
            while(True):
                line = f.readline().strip()
                if not line:
                    break
                self.namelist.append(line)
        self.root_dir = root_dir
        self.transform = transform
        self.train_type = train_type
        self.refine_size = 2048

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        pkl_path = os.path.join(self.root_dir,self.namelist[idx])
        pkl = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
        img = pkl[0].astype('float32')/255.0
        label = pkl[1][:,:3]
        
        # re-sample ground truth, ShapeNet point cloud ground truth by Wang et al. is not of the same number across images
        if label.shape[0]<self.refine_size:
            # re-sample
            sub_iter = self.refine_size // label.shape[0]
            sub_num =  self.refine_size - label.shape[0]*sub_iter
            label_n = label.copy()
            for i in range(sub_iter-1):
                label = np.concatenate((label, label_n), axis=0)
            subidx = np.random.permutation(label_n.shape[0])
            subidx = subidx[:sub_num]
            label = np.concatenate((label, label_n[subidx]), axis=0)

        # load mask
        mask_path = self.root_dir+self.namelist[idx][:5]+'mask/'+self.namelist[idx][19:-3]+'png'
        mask = scipy.ndimage.imread(mask_path)
        mask = np.expand_dims(mask,axis=2)

        subidx = np.random.permutation(label.shape[0])
        subidx = subidx[:self.refine_size]
        label_f = label[subidx]
        label_f = np.float32(label_f)

        # data augmentation
        if self.train_type == 'train':
            # gamma
            random.seed()
            g_prob = np.random.random()*1+0.5
            img = exposure.adjust_gamma(img, g_prob)
            # intensity
            random.seed()
            g_prob = np.random.random()*127
            img = exposure.rescale_intensity(img*255.0, in_range=(g_prob, 255))
            # color channel
            random.seed()
            g_prob = np.random.random()*0.4+0.8
            img[:,:,0] = img[:,:,0]*g_prob
            random.seed()
            g_prob = np.random.random()*0.4+0.8
            img[:,:,1] = img[:,:,1]*g_prob
            random.seed()
            g_prob = np.random.random()*0.4+0.8
            img[:,:,2] = img[:,:,2]*g_prob
            np.clip(img, 0.0, 1.0 , out=img)

        # permute dim
        if self.transform:
            if self.train_type == 'train':
                img = data_transforms['train'](img).float()
                mask = data_transforms['train'](mask).float() 
            else:
                img = data_transforms['val'](img).float()
                mask = data_transforms['val'](mask).float()

        return img, label_f,  mask
