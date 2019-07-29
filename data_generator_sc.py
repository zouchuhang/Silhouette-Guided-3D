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
import cv2
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
    def __init__(self, root_dir, train_type, transform=None):
        print(root_dir+'/image/')
        self.namelist = next(os.walk(root_dir+'/image/'))[2]
        self.root_dir = root_dir
        self.transform = transform
        self.train_type = train_type
        self.maxcrop = 360

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        im_path = self.root_dir+'/image/'+self.namelist[idx]
        img = cv2.imread(im_path)
        img = img.astype('float32')/255.0
        mask_path = self.root_dir+'/modal/'+self.namelist[idx]
        mask = cv2.imread(mask_path)
        mask = mask.astype('float32')/255.0
        mask = mask[:,:,0]
        label_path = self.root_dir+'/amodal/'+self.namelist[idx]
        label = cv2.imread(label_path)
        label = label.astype('float32')/255.0
        label = label[:,:,0]
        
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
            if random.uniform(0, 1) > 0.5:
                img = np.fliplr(img).copy()
                label = np.fliplr(label).copy()
                mask = np.fliplr(mask).copy()

            # random rotation
            random.seed()
            rot = np.random.random()*10-5
            h = img.shape[0]
            w = img.shape[1]
            img = scipy.ndimage.interpolation.rotate(img, rot, mode='nearest')
            label = scipy.ndimage.interpolation.rotate(label, rot, mode='nearest')
            mask = scipy.ndimage.interpolation.rotate(mask, rot, mode='nearest')
            h2 = img.shape[0]
            w2 = img.shape[1]
            h_t = (h2-h)//2
            w_l = (w2-w)//2
            img = img[h_t:h_t+h, w_l:w_l+w,:]
            label = label[h_t:h_t+h, w_l:w_l+w]
            mask = mask[h_t:h_t+h, w_l:w_l+w]
        
            # random (center) cropping
            cropr = np.random.random()/8+1/4
            croph = int(round(self.maxcrop - cropr*self.maxcrop))
            h_s = (self.maxcrop - croph)//2
            img = img[h_s:h_s+croph, h_s:h_s+croph]
            label = label[h_s:h_s+croph, h_s:h_s+croph]
            mask  = mask[h_s:h_s+croph, h_s:h_s+croph]

        # reshape
        img = cv2.resize(img, (224, 224))
        np.clip(img, 0.0, 1.0 , out=img)
        label = cv2.resize(label, (224, 224))
        mask = cv2.resize(mask, (224, 224))
        mask = np.float32(mask>0.5)
        label = np.float32(label>0.5)

        label = np.expand_dims(label, axis=2)
        mask = np.expand_dims(mask, axis=2)
        img = np.concatenate((img, mask), axis=2)

        # permute dim
        if self.transform:
            if self.train_type == 'train':
                img = data_transforms['train'](img).float()
                label = data_transforms['train'](label).float()
            else:
                img = data_transforms['val'](img).float()
                label = data_transforms['val'](label).float()

        return img, label
