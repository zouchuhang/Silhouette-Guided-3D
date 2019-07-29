"""
Chuhang Zou
07.2019

Code Revised from:

Finetuning Torchvision Models
=============================

**Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__

"""

from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
import copy
print("PyTorch Version: ",torch.__version__)

from torch.utils import data
from model_sc import *
from data_generator_sc import *

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
model_path = "./model/resnet50_ae_seg_crossentropy_skip4_rot_maskrcnn_adam.pth"
train_datapath = '/data/czou4/DYCE/train_det/'
val_datapath = '/data/czou4/DYCE/val_det/'

# Pre-trained models to choose from [resnet18, resnet34, resnet50]
model_name = "resnet50"

# if load pretrained model
Flag_loadweights = False
weight_path = "./model/resnet50_ae_seg_crossentropy_skip4_rot_adam-2.pth"

# Number of classes in the dataset
num_classes = 1024

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for 
num_epochs = 100000
steps_per_epoch = 20


# Model Training and Validation Code
def train_model(model, train_generator, val_generator, optimizer, criterion, steps=100, num_epochs=25):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = np.Inf

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloaders = train_generator
            else:
                model.eval()   # Set model to evaluate mode
                dataloaders = val_generator

            loss_sum = 0.0
            step = 0

            # Iterate over data.
            for input in dataloaders:
                
                inputs = input[0]
                labels = input[1]
                
                # gpu mode
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    
                    # loss
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()

                # statistics
                loss_sum += loss.item()*inputs.size(0)/steps
                
                # Break after 'steps' steps
                if step==steps-1:
                    break
                step += 1

            print('{} Loss: {:.6f}'.format(phase, loss_sum))

            # deep copy the model
            if phase == 'val' and loss_sum < best_acc:
                best_acc = loss_sum
                best_model_wts = copy.deepcopy(model.state_dict())
                # save model
                torch.save(best_model_wts, model_path)
                print("Model saved ...")
            if phase == 'val':
                val_acc_history.append(loss_sum)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:6f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


print("Load Models...")
# Define the encoder
encoder = initialize_encoder(model_name, num_classes,use_pretrained=True)

# Full model
model_ft = SegNet(encoder, num_classes)

# Model initialization
set_parameter_requires_grad(model_ft)
# Print the model we just instantiated
#print(model_ft) 

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model_ft = model_ft.to(device)

# if load weights
if Flag_loadweights:
    pretrained_dict = torch.load(weight_path)
    model_dict = model_ft.state_dict()
    pretrained_dict['resnet.conv1.weight'] = torch.cat((pretrained_dict['resnet.conv1.weight'], model_dict['resnet.conv1.weight'][:,3:,:,:]), 1)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_ft.load_state_dict(model_dict)

# Gather the parameters to be optimized/updated in this run.
params_to_update = model_ft.parameters()
#print("Params to learn:")
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        print("\t",name)

# Create the Optimizer
optimizer_ft = optim.Adam(params_to_update, lr = 1e-4, eps = 1e-6)

# Setup the loss
criterion = nn.BCELoss()

# Load Data
print("Initializing Datasets and Dataloaders...")
train_set = ShapeNetDataset(train_datapath, 'train', transform=True)
train_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_set = ShapeNetDataset(val_datapath, 'val', transform=True)
val_generator = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

# Train and evaluate
model_ft, hist = train_model(model_ft, train_generator, val_generator, optimizer_ft, criterion, steps_per_epoch, num_epochs=num_epochs)

print('training done')
