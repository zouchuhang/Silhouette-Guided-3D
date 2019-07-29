from __future__ import print_function
from __future__ import division
import numpy as np
from torchvision import transforms
import os
import pickle
from torch.utils import data
import scipy.io as sio
import scipy.ndimage
import random
from skimage import exposure
import cv2

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

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def occlude_with_objects(image, mask, bbox1, image0, mask2, bbox2, bg):
    image1 = image.copy()
    mask1 = mask.copy()
    image2 = image0.copy()

    # cut mask2 and paste on mask1
    # random select pasting position
    h1 = bbox1[0, 2]-bbox1[0, 0]
    w1 = bbox1[0, 3]-bbox1[0, 1]
    h2 = bbox2[0, 2]-bbox2[0, 0]
    w2 = bbox2[0, 3]-bbox2[0, 1]
    random.seed()
    y1 = random.randint(max(bbox1[0, 0] - h2,0), bbox1[0, 2])
    random.seed()
    x1 = random.randint(max(bbox1[0, 1] - w2,0), bbox1[0, 3])
    x2 = min(x1+w2, image1.shape[1])
    x2_shrink = x1+w2-x2
    y2 = min(y1+h2, image1.shape[0])
    y2_shrink = y1+h2 - y2
    patch = image2[bbox2[0, 0]:(bbox2[0, 2]-y2_shrink), bbox2[0, 1]:(bbox2[0, 3]-x2_shrink),:]
    patch_msk = mask2[bbox2[0, 0]:(bbox2[0, 2]-y2_shrink), bbox2[0, 1]:(bbox2[0, 3]-x2_shrink)]
    patch_msk = np.repeat(patch_msk[:, :, np.newaxis], 3, axis=2)
    patch = patch*patch_msk
    image1[y1:y2, x1:x2,:] = image1[y1:y2, x1:x2,:]*(1-patch_msk)+patch
    patch_msk = np.float32(patch_msk[:,:,0])
    mask1 = np.float32(mask1)
    mask1[y1:y2, x1:x2,:] = mask1[y1:y2, x1:x2,:]-patch_msk[:, :, np.newaxis]
    mask1 = mask1>0

    # add background
    mask3 = mask1.copy()
    mask3[y1:y2, x1:x2,:] = mask3[y1:y2, x1:x2,:]+patch_msk[:, :, np.newaxis]
    image1 = image1*mask3+bg*(1-mask3)

    return image1, mask1

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
        if train_type == 'train':
            self.LSUN_dir = '/data/czou4/LSUN/train/'
        else:
            self.LSUN_dir = '/data/czou4/LSUN/val/'
        self.LSUNlist = next(os.walk(self.LSUN_dir))[2]
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
        if label.shape[0]<self.refine_size:#1024:
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

        # load LSUN background
        LSUN_path = os.path.join(self.LSUN_dir,np.random.choice(self.LSUNlist, 1)[0])
        LSUN_im = scipy.ndimage.imread(LSUN_path)
        LSUN_im = LSUN_im.astype('float32')/255.0
        LSUN_im = cv2.resize(LSUN_im, (224, 224))
        
        random.seed()
        if random.uniform(0, 1) > 0.5:
            LSUN_im = np.ones((224, 224, 3)).astype('float32')
        # get occlusion
        random.seed()
        #if True:
        if random.uniform(0, 1) > 0.5:
            while True:
                mask = mask>0
                mask = 1-scipy.ndimage.binary_dilation(1-mask)
                mask1 = mask.copy()
                bbox1 = extract_bboxes(mask1)
                # occ from copy&paste
                namelist_sub = np.random.choice(self.namelist, 1)[0]
                pkl_path = os.path.join(self.root_dir,namelist_sub)
                pkl = pickle.load(open(pkl_path, 'rb'), encoding='bytes')
                img_occ = pkl[0].astype('float32')/255.0
                # load mask
                mask_path = self.root_dir+namelist_sub[:5]+'mask/'+namelist_sub[19:-3]+'png'
                mask_occ = scipy.ndimage.imread(mask_path)
                mask_occ = np.expand_dims(mask_occ,axis=2)
                mask1_occ = mask_occ>0
                mask1_occ = scipy.ndimage.binary_dilation(1-mask1_occ)
                mask1_occ = 1-mask1_occ
                bbox2 = extract_bboxes(mask1_occ)
                img_new, mask_new = occlude_with_objects(img, mask1, bbox1, img_occ, np.squeeze(mask1_occ,axis=2), bbox2, LSUN_im)
                # skip largely occluded sample
                occ_ratio = np.sum(mask_new&mask1)/(np.sum(mask1)+1e-8)
                if occ_ratio > 0.5:
                    img = img_new
                    # uncomment below for visible silhouette
                    #mask = mask_new
                    break

        mask = np.float32(mask)

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
