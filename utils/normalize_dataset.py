#!/bin/env python3
import os
import numpy as np
import imageio as iio
import time
import torch
import json
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split,Subset
from utils.dataset_utils import CustomDataset, Dubai, Masa, MasaFilt, MasaFull,MasaSeven, DubaiSeven, ImagesPre
from utils.utils import load_normalize_data
from config import CONFIG

import argparse

"""
Training set:
RGB-means [128.65051054 118.45636216 130.87956071]
RGB-stds [42.03129609 39.19244565 41.53636231]
"""

parser = argparse.ArgumentParser()
parser.add_argument("--dataset" , type = str, default = CONFIG.MISC_dataset, help = "Select which dataset to normalize.")


args = parser.parse_args()
# Set dataset
dataset = args.dataset

print(f"Calculating the mean and std for {dataset}")
# Global vars
BASE_PATH_DATA = CONFIG.MISC_dataset_path
SEED = 0
batch_size = 1
n_chan = 3
transform = None
download = False
if dataset == 'masa':
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = Masa(CONFIG.MISC_dataset_path,split = 'train',transform = transform)
elif dataset == 'masa_filt':
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = MasaFilt(CONFIG.MISC_dataset_path,split = 'train',transform = transform)
elif dataset == 'dubai':
    CONFIG.RL_priv_use_seg = True
    trainset = Dubai(CONFIG.MISC_dataset_path,split = 'train',transform = transform)
    n_chan  = 6
elif dataset.startswith('custom_'):
    trainset = CustomDataset(CONFIG.MISC_dataset_path, CONFIG.MISC_dataset[7:], split='train', transform = transform)
elif dataset == 'masa_full':
    trainset = MasaFull(CONFIG.MISC_dataset_path , split = 'train' , transform = transform)
    n_chan = 3
elif dataset == 'masa_seven':
    trainset = MasaSeven(CONFIG.MISC_dataset_path, split = 'train', transform = transform)
    n_chan = 3
elif dataset == 'dubai_seven':
    trainset = DubaiSeven(CONFIG.MISC_dataset_path, split = 'train', transform = transform)
    n_chan = 3
elif dataset == 'images_pre':
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = ImagesPre(CONFIG.MISC_dataset_path, split = 'train', transform = transform)
    n_chan = 3
elif dataset == 'images_post':
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = ImagesPre(CONFIG.MISC_dataset_path, split = 'train', transform = transform, post_instead=True)
    n_chan = 3
else:
    raise(Exception("Unknown dataset"))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=download, num_workers=2 )
# Calculate mean and std of pixel values in dataset
# See this page for how to do it incrementally with respect to std
# https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
img_means = np.zeros(n_chan)
img_s0s = np.zeros(n_chan)
img_s1s = np.zeros(n_chan)
img_s2s = np.zeros(n_chan)
for it, frame in enumerate(trainloader):

    frame = frame[0].detach().numpy()
    IM_H ,IM_W = frame.shape[-2:]
    # Update running means
    for chan_idx in range(n_chan):
        img_means[chan_idx] = (it *  img_means[chan_idx] + np.mean(frame[:, chan_idx, :,:])) / (it + 1)
        img_s0s[chan_idx] += IM_H * IM_W
        img_s1s[chan_idx] += np.sum(frame[:, chan_idx, :,:])
        img_s2s[chan_idx] += np.sum(frame[:, chan_idx, :,:] * frame[:, chan_idx, :,:])
    # Display
    if it % 100 == 0:
        img_stds = np.sqrt(np.abs(((img_s0s * img_s2s) - img_s1s * img_s1s) / (img_s0s * (img_s0s - 1))))
        print(f"Iter {it}/{len(trainloader)}")
        print("RGB-means", img_means)
        print("RGB-stds", img_stds)

dataset_path = dataset if not dataset.startswith('custom_') else os.path.join("Custom", dataset[7:])

img_stds = np.sqrt(np.abs(((img_s0s * img_s2s) - img_s1s * img_s1s) / (img_s0s * (img_s0s - 1))))
stat_path = os.path.join(CONFIG.MISC_dataset_path,dataset_path,'stats.json')
stats = {"means":list(img_means),"stds":list(img_stds)}

with open(stat_path, 'w') as fp:
    json.dump(stats,fp,indent = 4)

print(f"Done!! \nThe mean and std for {dataset} is:")
print("RGB-means", img_means)
print("RGB-stds", img_stds)
