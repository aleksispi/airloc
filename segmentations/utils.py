from config import CONFIG

import math
import json
import os
import time
import zipfile
import gc
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split,Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import urllib
from dateutil.parser import parse

import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import cross_entropy

# Run on init, setup colors for segmentation classes
CLASS_COLORS_LIST = ['#3C1098', '#8429F6', '#6EC1E4', '#FEDD3A', '#E2A929', '#9B9B9B','#000000' ]
CLASS_COLORS_TENSOR = torch.tensor([ to_rgb(c) for c in CLASS_COLORS_LIST])

"""
labels = [int(to_rgb(l)[0]*255) for l in labels]
cls = range(1,len(labels)+1)
labels = dict(zip(labels,cls))
"""

def map_ohe_to_color(seg_mask):
    """ Maps a ohe feature map with C channels to a RGB image with specific colors for all classes"""

    if len(seg_mask.shape) > 3:
        # Get seg mask details
        num_images, num_classes, im_H, im_W = seg_mask.shape
        
        #num_classes = torch.max(seg_mask).item() + 1

        # Convert class values to RGB image, permute to put channel dim infront of width and height
        rgb_images = CLASS_COLORS_TENSOR[torch.argmax(seg_mask , dim = 1)].permute(0, 3, 1 , 2) 
    else:
        num_images, im_H, im_W = seg_mask.shape

        rgb_images = CLASS_COLORS_TENSOR[seg_mask].permute(0,3,1,2) 

    return rgb_images
 

def visualize_segmentation_est(val_seg , est_seg , vis_name = "visualization" , transform = None, max_images = 5):

    val_seg = val_seg[0:max_images].cpu()
    est_seg = est_seg[0:max_images].cpu()
    max_images = min(max_images, val_seg.shape[0]) 

    # DUe to ndindex being very weird and bad we need a litle hack
    if max_images == 1:
        max_images = 2
        turn_off_last_row = True
    else:
        turn_off_last_row = False

    fig, axes = plt.subplots(max_images, 2)

    # Transform images from channel wise ohe to RGB images
    rgb_val_seg = map_ohe_to_color(val_seg)

    # Check if still ohe 
    if len(est_seg.shape) > 3: 
        est_seg = torch.argmax(est_seg, dim = 1)

    rgb_est_seg = map_ohe_to_color(est_seg)

    for (i, axis_inds) in enumerate(np.ndindex(axes.shape)):
        ix, iy = axis_inds 

        if turn_off_last_row and ix == 1:
            axes[ix,iy].axis('off')
            continue

        curr_img = rgb_val_seg[ix,:] if iy == 0 else rgb_est_seg[ix,:]

        axes[ix,iy].imshow(curr_img.permute(1,2,0))

        if (ix,iy) == (0,0):
            axes[ix,iy].set_title('Ground Truth')
        if (ix,iy) == (0,1):
            axes[ix,iy].set_title('Estimate Segmentation')

    plt.savefig(os.path.join(CONFIG.STATS_SEG_log_dir , vis_name) + '.png')

    plt.cla()
    plt.clf()
    plt.close('all')
    gc.collect()


def format_labels_for_loss(labels):

    # If input is single channel, transform to ohe
    if labels.shape[1] == 1:
        n_classes = int(torch.max(labels).item()) + 1
        label_out = torch.zeros((labels.shape[0] , n_classes , labels.shape[2],labels.shape[3]))
        for i in range(n_classes):
            
            label_out[:,i,:,:] = ((labels == i) * 1.0).squeeze(1)
        
        label_out = label_out.long()
    else:
        label_out = labels

    return label_out.to(CONFIG.device)


def compute_loss(est_mask  , label_mask):
    
    # For now only compute cross_entropy loss
    loss = cross_entropy(est_mask , label_mask)

    # TODO - Add boundary separation loss

    return loss


def mask_image(image_batch, grid_size = 8): 
    p = 0.5
    masked_batch = image_batch.clone().detach()
    n_up = int(CONFIG.MISC_im_size[0]/grid_size)
    
    
    for i_img in range(masked_batch.shape[0]):
        masked = (np.random.uniform(size = (8,8)) > p )
        #masked = np.resize(masked,CONFIG.MISC_im_size)
        masked = masked.repeat(n_up, axis=0).repeat(n_up, axis=1)
        masked_batch[i_img] = masked_batch[i_img] * torch.tensor(masked)

    return masked_batch


# DICE Loss
class DiceLoss(nn.Module):

    def __init__(self, class_weights = None):
        super(DiceLoss, self).__init__()
      
        self.class_weights = class_weights
        if isinstance( self.class_weights, list):
            self.class_weights = torch.tensor(self.class_weights)

        self.class_weights = self.class_weights.to(CONFIG.device)

    def forward(self, inputs , targets):
        """
            Calculate Dice loss between inputs and targets assumed to be BxCxHxW
        """
        if inputs.max() > 1 or inputs.min() < -1:
            inputs = torch.sigmoid(inputs)

        numerator = 2 * (inputs * targets).flatten( start_dim = 2 ).sum( dim = 2)
        
        denominator = (inputs.sum(dim = (2,3)) + targets.sum(dim = (2,3)))

        dice_loss = (numerator + 1.0) / (denominator + 1.0) 

        if not self.class_weights is None:
            dice_loss = self.class_weights * dice_loss

        return -dice_loss.mean()

def debug_plot(im): 
    matplotlib.use('TkAgg') 
    plt.imshow(im.permute(1,2,0))
    plt.show()

