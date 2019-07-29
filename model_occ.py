
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import resnet_seg as resnet
import torch.nn.functional as F

# Initialize and Reshape the Encoders
def initialize_encoder(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet18":
        """ Resnet18
        """
        #model_ft = models.resnet18(pretrained=use_pretrained)
        model_ft = resnet.resnet18(pretrained=use_pretrained, num_classes=1000)
        #set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet34":
        """ Resnet34
        """
        #model_ft = models.resnet34(pretrained=use_pretrained)
        model_ft = resnet.resnet34(pretrained=use_pretrained, num_classes=1000)
        #set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        #model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = resnet.resnet50(pretrained=use_pretrained, num_classes=1000)
        #set_parameter_requires_grad(model_ft)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

# full model
class OccNet(nn.Module):
    def __init__(self, encoder, num_classes, grid_size):
        super(OccNet, self).__init__()
        self.resnet = encoder
        self.conv1 = nn.Conv2d(num_classes+5, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 3, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.num_coarse = 1024
        self.grid_size = grid_size
        self.num_fine = self.grid_size*self.grid_size*self.num_coarse

        self.linear1 = nn.Linear(num_classes, 1024)
        self.linear2 = nn.Linear(1024, self.num_coarse*3)

    def forward(self, images):
        fea = self.resnet(images)
        # coarse
        x = self.linear1(fea)
        x = self.relu(x)
        x = self.linear2(x)
        x = x.view(x.size()[0], -1, 3)
        
        # refined
        dx = torch.linspace(-0.05, 0.05, steps = self.grid_size).cuda()
        dy = torch.linspace(-0.05, 0.05, steps = self.grid_size).cuda()
        dx = dx.repeat(self.grid_size,1)
        dy = dy.repeat(self.grid_size,1).t()
        grid = torch.stack([dx, dy], dim=2).view(-1,2).unsqueeze(0)
        grid_feat = grid.repeat(fea.shape[0], self.num_coarse, 1)
       
        point_feat = x.unsqueeze(2).repeat(1,1,self.grid_size*self.grid_size, 1)
        point_feat = point_feat.view(-1, self.num_fine, 3)

        global_feat = fea.unsqueeze(1).repeat(1,self.num_fine,1)

        feat = torch.cat((grid_feat, point_feat, global_feat),2)
        
        center = x.unsqueeze(2).repeat(1,1,self.grid_size*self.grid_size, 1)
        center = center.view(-1, self.num_fine, 3)

        feat = feat.unsqueeze(3).permute(0,2,3,1)
        fine = self.relu(self.conv1(feat))
        fine = self.relu(self.conv2(fine))
        fine = self.conv3(fine)
        fine = fine.squeeze(2).permute(0,2,1)
        fine = fine+center

        return x , fine

# 3d point-wise loss
def distChamfer_bn(a,b):
    x,y = a,b
    # random subsample to save memory
    seg_idx = torch.randperm(x.shape[1])
    seg_idx = seg_idx[:2048] 
    x = x[:,seg_idx,:]

    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)

    return P.min(1)[0], P.min(2)[0]

# 3d point-wise loss
def distChamfer(a,b):
    x,y = a,b

    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return P.min(1)[0], P.min(2)[0]

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
    return P.min(2)[0] #P.min(1)[0], P.min(2)[0]

# 2d loss:
def distChamfer_2d(a, m):
    pt_x, pt_y, pt_z = torch.chunk(a, 3, dim=2)
    # projection
    F = 248
    h = (-pt_y)/(-pt_z)*F + 224/2.0
    w = pt_x/(-pt_z)*F + 224/2.0
    
    pred = torch.cat((h,w), 2)

    # segment indices
    dist = 0.0
    for i in range(m.shape[0]):
        #print(i)
        mask = m[i][0]
        pt = pred[i].unsqueeze(0)
        seg=(mask>0.5).nonzero().float().unsqueeze(0)
        # subsample gt to save memory
        if seg.shape[1]>25000:
            seg_idx = torch.randperm(seg.shape[1])
            seg_idx = seg_idx[:25000]
            seg = seg[:,seg_idx,:]
        dist1 = pairwise_distances(seg, pt)
        dist += torch.mean(dist1)
    dist = dist/m.shape[0]
    
    return dist

def distChamfer_2d_t(m, n):
    a,b = m,n
    bs = b.shape[0]

    seg_idx = torch.randperm(a.shape[1])
    seg_idx = seg_idx[:2048]
    a = a[:,seg_idx,:]
    
    a_cen = torch.mean(a, 1, True)
    b_cen = torch.mean(b, 1, True)    

    # projection
    F = 248
    
    # y transform
    aa = a - a_cen
    bb = b - b_cen
    Ry = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]).cuda()
    Ry = Ry.unsqueeze(0).repeat(bs, 1, 1)
    aa = torch.bmm(aa, Ry)+a_cen
    bb = torch.bmm(bb, Ry)+b_cen
    pt_x, pt_y, pt_z = torch.chunk(aa, 3, dim=2)
    gt_x, gt_y, gt_z = torch.chunk(bb, 3, dim=2)
    h = (-pt_y)/(-pt_z)*F + 224/2.0
    w = pt_x/(-pt_z)*F + 224/2.0
    pred = torch.cat((h,w), 2)
    h_gt = (-gt_y)/(-gt_z)*F + 224/2.0
    w_gt = gt_x/(-gt_z)*F + 224/2.0
    gt = torch.cat((h_gt,w_gt), 2)
    
    disty, _ = distChamfer(pred,gt)

    # x transform
    aa = a - a_cen
    bb = b - b_cen
    Ry = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]).cuda()
    Ry = Ry.unsqueeze(0).repeat(bs, 1, 1)
    aa = torch.bmm(aa, Ry)+a_cen
    bb = torch.bmm(bb, Ry)+b_cen
    pt_x, pt_y, pt_z = torch.chunk(aa, 3, dim=2)
    gt_x, gt_y, gt_z = torch.chunk(bb, 3, dim=2)
    h = (-pt_y)/(-pt_z)*F + 224/2.0
    w = pt_x/(-pt_z)*F + 224/2.0
    pred = torch.cat((h,w), 2)
    h_gt = (-gt_y)/(-gt_z)*F + 224/2.0
    w_gt = gt_x/(-gt_z)*F + 224/2.0
    gt = torch.cat((h_gt,w_gt), 2)
    distx, _ = distChamfer(pred,gt)
    
    dist = torch.mean(disty)+torch.mean(distx)

    return dist
    

# Set Model Parameters, requires_grad attribute
def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = True
