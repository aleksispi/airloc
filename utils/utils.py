import math
import json
import os
import time
import zipfile
import signal
import pdb
import gc
from shutil import copyfile
import numpy as np
import torch
from glob import glob
# Might not be available on RISE
import seaborn as sns

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split,Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import urllib
from utils.dataset_utils import CustomDataset,Dubai,Masa, MasaSeven, MasaFilt, MasaFull, DubaiSeven
from dateutil.parser import parse

# from torchviz import make_dot
from config import CONFIG


def move_crop(crop, offset_xy):
    """
    crop has 4 columns in (h, w)-coordinate system:
        1. Top-coordinate of the crop
        2. Left-coordinate of the crop
        3. Height of the crop
        4. Width of the crop

    offset_xy has 2 columns in (x, y)-coordinate system:
        1. Movement in x-direction
        2. Movement in y-direction

    The output is given by moving crop along offset_xy
    """

    # Translate offset_xy from (x,y)- to (h,w)-coordinate system
    if isinstance(offset_xy, np.ndarray):
        offset_hw = np.zeros_like(offset_xy)
    else:
        offset_hw = torch.zeros_like(offset_xy)
    offset_hw[:, 0] = -offset_xy[:, 1]  # H_img - offset_xy[:, 1]
    offset_hw[:, 1] = offset_xy[:, 0]

    # Perform the translation
    if isinstance(offset_xy, np.ndarray):
        moved_crop = np.zeros_like(crop)
    else:
        moved_crop = torch.zeros_like(crop)
    moved_crop[:, :2] = crop[:, :2] + offset_hw
    moved_crop[:, 2:] = crop[:, 2:]
    return moved_crop

def compute_iou(crop1, crop2):
    """
    Any given row of any of the two crops has the following format:
        1. Top-coordinate of the crop
        2. Left-coordinate of the crop
        3. Height of the crop
        4. Width of the crop

    The output is the intersection-over-unions (IoUs) between the rows of
    crop1 and crop2
    """

    # Ensure correct data types and dims
    if isinstance(crop1, tuple) or isinstance(crop1, list):
        crop1 = np.array(crop1)[np.newaxis, :]
    if isinstance(crop2, tuple) or isinstance(crop2, list):
        crop2 = np.array(crop2)[np.newaxis, :]
    if crop1.ndim == 1:
        crop1 = crop1[np.newaxis, :]
    if crop2.ndim == 1:
        crop2 = crop2[np.newaxis, :]

    # Get the coordinates of the intersection bounding box
    try:
        ihmins = np.maximum(crop1[:, 0], crop2[:, 0])
    except:
        print(crop1, crop2)
    ihmaxs = np.minimum(crop1[:, 0] + crop1[:, 2], crop2[:, 0] + crop2[:, 2])
    iwmins = np.maximum(crop1[:, 1], crop2[:, 1])
    iwmaxs = np.minimum(crop1[:, 1] + crop1[:, 3], crop2[:, 1] + crop2[:, 3])
    # TODO: Find out why this plus one was here
    iws = np.maximum(iwmaxs - iwmins, 0)
    ihs = np.maximum(ihmaxs - ihmins, 0)

    # Calculate the area of the intersection
    inters = iws * ihs

    # Calculate the area of union
    unis = crop1[:, 2] * crop1[:, 3] + crop2[:, 2] * crop2[:, 3] - inters

    # Calculate and return the IoUs between crop1 and crop2
    return inters / unis

def get_frac_outside(crop):
    """
    Any given row of crop has the following format
        1. Top-coordinate of the crop
        2. Left-coordinate of the crop
        3. Height of the crop
        4. Width of the crop

    The output is the percentage (fraction) of crops in crop (i.e. rows in
    crop) that fall at least partially outside the full image
    """

    # Get size of the full image
    H_img, W_img = CONFIG.MISC_im_size

    # Check for out-of-image
    hmins_outside = crop[:, 0] < 0
    hmaxs_outside = crop[:, 0] + crop[:, 2] >= H_img
    wmins_outside = crop[:, 1] < 0
    wmaxs_outside = crop[:, 1] + crop[:, 3] >= W_img

    # Compute fraction of outside
    outsides = np.logical_or(np.logical_or(hmins_outside, hmaxs_outside),
                             np.logical_or(wmins_outside, wmaxs_outside))
    return np.count_nonzero(outsides) / len(outsides)

def normalize_coords(coords_xy, crop_locs_start, crop_locs_goal, unnormalize=False):

    # Get size of the full image
    H_img, W_img = CONFIG.MISC_im_size

    # Start and goal bbox widths
    heights_start = crop_locs_start[:, 2]
    widths_start = crop_locs_start[:, 3]
    heights_goal = crop_locs_goal[:, 2]
    widths_goal = crop_locs_goal[:, 3]

    # Perform the unnormalization
    if isinstance(coords_xy, np.ndarray):
        coords_xy_out = np.copy(coords_xy)
    else:
        coords_xy_out = torch.clone(coords_xy)
    if unnormalize:
        coords_xy_out[:, 0] *= (W_img - widths_start / 2 - widths_goal / 2)
        coords_xy_out[:, 1] *= (H_img - heights_start / 2 - heights_goal / 2)
    else:
        coords_xy_out[:, 0] /= (W_img - widths_start / 2 - widths_goal / 2)
        coords_xy_out[:, 1] /= (H_img - heights_start / 2 - heights_goal / 2)
    return coords_xy_out


def _setup_fixed_games(n_games = 5):
    """ Randomly sample n_games number of fixed games."""

    # Get some images to be able to use get_random_crops
    images = torch.zeros((n_games, 3, CONFIG.MISC_im_size[0] , CONFIG.MISC_im_size[1]))

    _ , start_crop_locs = get_random_crops(images)

    # Then sample the goal locs

    _ , goal_crop_locs = get_random_crops( images, other_crop_locs = start_crop_locs , max_dist = CONFIG.RL_max_start_goal_dist)

    return start_crop_locs , goal_crop_locs

def get_random_crops(images, other_crop_locs=None, max_dist=None, min_iou=None):
    """
    Note that if max_dist and min_iou are both provided, then only max_dist
    will be used. Hence, for min_iou to take effect, max_dist has to be None.
    """
    # Define some useful constants
    H, W = CONFIG.MISC_patch_size
    step_sz = int(CONFIG.RL_softmax_step_size*H)
    im_H,im_W = CONFIG.MISC_im_size
    crop_locations = torch.zeros((images.shape[0], 4))
    n_chan = images.shape[1]
    n_imgs = images.shape[0]

    # Initialize memory for the crops size = (batch, n_chan, H_p, W_p)
    # Keep the number of channels at a constant
    crops = torch.zeros(size=(n_imgs, n_chan, H, W))

    for i in range(images.shape[0]):
        if CONFIG.MISC_grid_game:
            # Image is divided into a static uniform grid. Patches are sampled from this grid
            upper_H , upper_W = int(im_H / H) , int(im_W / W)
            lower_H , lower_W = ( 0 , 0)

            target_pos = np.array([-1,-1])
            if max_dist is not None:
                target_pos = (other_crop_locs[i,0:2].numpy() / np.array(CONFIG.MISC_patch_size)).astype('int64')
                upper_H , upper_W = (min(target_pos[0] + max_dist + 1, upper_H),min(target_pos[1] + max_dist + 1, upper_W)) #Has to be in int
                lower_H , lower_W = (max(target_pos[0] - (max_dist ), 0) , max(target_pos[1] - max_dist, 0))

            grid_loc = np.floor(np.random.uniform( low = [lower_H ,lower_W] , high = [upper_H , upper_W]))
            while (grid_loc == target_pos).all():
                grid_loc = np.floor(np.random.uniform( low = [lower_H ,lower_W] , high = [upper_H , upper_W]))
            crop_loc = np.concatenate(((grid_loc ) *
                                       np.array(step_sz) , np.array(CONFIG.MISC_patch_size)) ).astype('int64')

        else:
            # Sample the patches entirly at random
            break_while = False
            while not break_while:
                crop_loc = transforms.RandomCrop.get_params(images[i, :, :, :][np.newaxis, :],
                                                            output_size=CONFIG.MISC_patch_size)
                break_while = other_crop_locs is None or (max_dist is None and min_iou is None)
                if not break_while:
                    # At this stage we may want to ensure that the sampled crop
                    # is not too far away from other_crop_locs, or that they
                    # do not have too little IoU-overlap
                    if max_dist is not None:
                        offset = get_regression_targets(crop_loc, other_crop_locs[i, :][np.newaxis, :],
                                                        normalize=False)
                        break_while = np.linalg.norm(offset) <= max_dist
                    elif min_iou is not None:
                        iou = compute_iou(crop_loc, other_crop_locs[i, :][np.newaxis, :])
                        break_while = iou >= min_iou

        crop_locations[i, :] = torch.Tensor(np.array(crop_loc, dtype = int))
        crops[i, :n_chan, :, :] = transforms.functional.crop(images[i, :, :, :], top=crop_loc[0],
                                                       left=crop_loc[1], height=crop_loc[2],
                                                       width=crop_loc[3])
    return crops, crop_locations




def get_deterministic_crops(images,coords = [0,0]):
    """
    Allows for extraction of deterministic crops in the image
    """
    # Define some useful constants
    H, W = CONFIG.MISC_patch_size
    im_H,im_W = CONFIG.MISC_im_size
    crop_locations = torch.zeros((images.shape[0], 4))
    n_chan = images.shape[1]
    n_imgs = images.shape[0]

    # Initialize memory for the crops size = (batch, n_chan, H_p, W_p)
    # Keep the number of channels at a constant
    crops = torch.zeros(size=(n_imgs, n_chan, H, W))

    # Coords can be supplied as list or tensor but needs to be in correct numpy format
    if isinstance(coords , torch.Tensor):
        coords = coords.detach().numpy()
    if isinstance(coords , list):
        coords = np.array(coords)
    if len(coords.shape) == 1:
        coords = np.expand_dims(coords, 0)
        coords = np.repeat(coords , n_imgs , 0)

    # Iterate over the images getting the correct patches
    for i in range(n_imgs):
        h = int(coords[i][0])
        w = int(coords[i][1])
        crop_loc = [h , w , H,W]

        if not CONFIG.RL_agent_allowed_outside and  check_outside(torch.as_tensor(crop_loc)[None,:]):
            # if the agent is not allowed outside and is going to end up outside
            # move it in again
            crop_loc = project_into_image(crop_loc)

        # Sample the crop
        crops[i, :n_chan, :, :] = transforms.functional.crop(images[i, :, :, :], top= crop_loc[0], left=crop_loc[1], height=H, width=W)[:,:48,:48]
        crop_locations[i, :] = torch.Tensor(np.array(crop_loc))

    return crops, crop_locations

def get_regression_targets(crop_locs_start, crop_locs_goal, normalize=True):

    # Get size of the full image
    H_img, W_img = CONFIG.MISC_im_size

    # Ensure correct data types and dims
    if isinstance(crop_locs_start, tuple) or isinstance(crop_locs_start, list):
        crop_locs_start = np.array(crop_locs_start)[np.newaxis, :]
    if isinstance(crop_locs_goal, tuple) or isinstance(crop_locs_goal, list):
        crop_locs_goal = np.array(crop_locs_goal)[np.newaxis, :]
    if isinstance(crop_locs_start, np.ndarray) and crop_locs_start.ndim == 1:
        crop_locs_start = crop_locs_start[np.newaxis, :]
    if isinstance(crop_locs_goal, np.ndarray) and crop_locs_goal.ndim == 1:
        crop_locs_goal = crop_locs_goal[np.newaxis, :]

    # Start
    tops_start = crop_locs_start[:, 0]
    lefts_start = crop_locs_start[:, 1]
    heights_start = crop_locs_start[:, 2]
    widths_start = crop_locs_start[:, 3]
    # Go from (h,w)- to (x,y)-coordinate system
    xs_start = lefts_start + widths_start / 2
    ys_start = H_img - (tops_start + heights_start / 2)

    # Goal

    tops_goal = crop_locs_goal[:, 0]
    lefts_goal = crop_locs_goal[:, 1]
    heights_goal = crop_locs_goal[:, 2]
    widths_goal = crop_locs_goal[:, 3]
    # Go from (h,w)- to (x,y)-coordinate system
    xs_goal = lefts_goal + widths_goal / 2
    ys_goal = H_img - (tops_goal + heights_goal / 2)

    # Offsets
    xs_offset = xs_goal - xs_start
    ys_offset = ys_goal - ys_start

    # Concatenate
    if isinstance(xs_offset, np.ndarray):
        regression_targets = np.concatenate([xs_offset[:, np.newaxis], ys_offset[:, np.newaxis]], 1)
    else:
        regression_targets = torch.cat([torch.unsqueeze(xs_offset, 1), torch.unsqueeze(ys_offset, 1)], 1)

    # Potentially normalize regression targets to the [-1, 1]-range
    if normalize:
        regression_targets = normalize_coords(regression_targets, crop_locs_start, crop_locs_goal)

    # Return the regression targets
    return regression_targets



def load_normalize_data(download=False, batch_size=16,
                        multiply_images = None, split='val', use_eval_split = False):

    if not CONFIG.MISC_dataset in ['masa_filt', 'masa_seven'] and split=='test':
        raise(Exception("Testing mode only implemented for the masa filt dataset"))
    if CONFIG.MISC_dataset.startswith('custom_'):
        stat_path = os.path.join(CONFIG.MISC_dataset_path,"Custom", CONFIG.MISC_dataset[7:],'stats.json')
    else:
        stat_path = os.path.join(CONFIG.MISC_dataset_path,CONFIG.MISC_dataset,'stats.json')
    if os.path.exists(stat_path):
        with open(stat_path) as json_file:
            stats = json.load(json_file)
    else:
        print("Unable to find calculated mean and std for this dataset.\nUse utils/normalize_dataset.py")
        exit(1)
    # to control the behaviour of the function
    CONFIG.RUNTIME_multiply_images = multiply_images
    def collate_fn(batch):
        # Transform each returned batch to desirerd format
        images , labels = tuple(zip(*batch))

        # If enabled make several training examples of each image
        if CONFIG.RUNTIME_multiply_images is not None and CONFIG.RUNTIME_multiply_images != 1:

           images = torch.stack(images)
           images = torch.repeat_interleave(images,CONFIG.RUNTIME_multiply_images , dim = 0)

           temp = np.asarray(labels)
           start_crops = np.repeat(temp[:,0,:] , CONFIG.RUNTIME_multiply_images , axis =0)
           goal_crops = np.repeat(temp[:,1,:] , CONFIG.RUNTIME_multiply_images, axis = 0)
           labels = (start_crops , goal_crops)
        else:
            temp = np.asarray(labels)
            labels = (temp[:,0,:] , temp[:,1,:])
            images = torch.stack(images)
        return ( images , labels)

    # Select which interpolation to be used
    interpolation = 'bilinear'

    if interpolation == 'nearest':
        # Works for label masks but totally destroys the images.
        # Unable to train models on images that have been resize with nearest
        interpolation_mode = transforms.InterpolationMode.NEAREST
    elif interpolation == 'bilinear':
        # BILINEAR interpolation ruin label masks
        interpolation_mode = transforms.InterpolationMode.BILINEAR
    elif interpolation == 'bicubic':
        interpolation_mode = transforms.InterpolationMode.BICUBIC
    else:
        raise(Exception("Unkonwn interpolation mode"))

    transforms_train = [transforms.Resize([CONFIG.MISC_im_size[0]+4,CONFIG.MISC_im_size[1]+4], interpolation = interpolation_mode),
                        transforms.CenterCrop(CONFIG.MISC_im_size)
                       ]
    transforms_val = [transforms.Resize([CONFIG.MISC_im_size[0]+4,CONFIG.MISC_im_size[1]+4], interpolation = interpolation_mode),
                      transforms.CenterCrop(CONFIG.MISC_im_size)]

    # Data augmentation
    transforms_train += [
        #transforms.RandomResizedCrop(CONFIG.MISC_im_size, scale = (0.8,0.8),ratio = (1,1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()]

    # If we use seg add a dummy variable to make the transforms work properly
    if CONFIG.RL_priv_use_seg:
        if CONFIG.MISC_dataset in ['masa', 'masa_filt']:
            stats['means'] += [0]
            stats['stds'] += [1]
    else:
        stats['means'] = stats['means'][:3]
        stats['stds'] = stats['stds'][:3]
        transforms_train = [transforms.ToTensor()] + transforms_train
        transforms_val = [transforms.ToTensor()] + transforms_val

    transforms_train += [transforms.Normalize(stats["means"], stats["stds"])]
    transforms_val += [transforms.Normalize(stats["means"], stats["stds"])]
    transform_train = transforms.Compose(transforms_train)
    transform_val = transforms.Compose(transforms_val)

    if CONFIG.MISC_dataset == 'dubai':
        trainset = Dubai(CONFIG.MISC_dataset_path,split = 'train',transform
                             = transform_train)
        valset = Dubai(CONFIG.MISC_dataset_path ,split = 'val',transform =
                           transform_val, use_eval_split = use_eval_split)
    elif CONFIG.MISC_dataset == 'masa':
        trainset = Masa(CONFIG.MISC_dataset_path,split = 'train',transform
                             = transform_train)
        valset = Masa(CONFIG.MISC_dataset_path ,split = 'val',transform =
                           transform_val)
    elif CONFIG.MISC_dataset == 'masa_filt':
        trainset = MasaFilt(CONFIG.MISC_dataset_path,split = 'train',transform
                             = transform_train,use_eval_split = False)
        valset = MasaFilt(CONFIG.MISC_dataset_path ,split=split, transform =
                           transform_val,use_eval_split = use_eval_split)
    elif CONFIG.MISC_dataset == 'masa_seven':
        trainset = MasaSeven(CONFIG.MISC_dataset_path,split = 'train',transform
                             = transform_train,use_eval_split = False)
        valset = MasaSeven(CONFIG.MISC_dataset_path ,split=split, transform =
                           transform_val,use_eval_split = use_eval_split)
    elif CONFIG.MISC_dataset == 'dubai_seven':
        trainset = DubaiSeven(CONFIG.MISC_dataset_path,split = 'train',transform
                             = transform_train,use_eval_split = False)
        valset = DubaiSeven(CONFIG.MISC_dataset_path ,split=split, transform =
                           transform_val,use_eval_split = use_eval_split)
    elif CONFIG.MISC_dataset.startswith('custom_'):
        trainset = CustomDataset(CONFIG.MISC_dataset_path, CONFIG.MISC_dataset[7:],split = 'train' , transform = transform_train, custom_split_file = CONFIG.MISC_dataset_split_file)
        valset = CustomDataset(CONFIG.MISC_dataset_path,CONFIG.MISC_dataset[7:],split ='val', transform = transform_val, custom_split_file = CONFIG.MISC_dataset_split_file)
    elif CONFIG.MISC_dataset == 'masa_full':
        trainset = MasaFull(CONFIG.MISC_dataset_path, split='train', transform=transform_train, randomRotation=True)
        valset = MasaFull(CONFIG.MISC_dataset_path, split='val', transform=transform_val, randomRotation=True)
    else:
        raise(Exception("Unknown dataset"))
    def worker_init(*args):
        signal.signal(signal.SIGINT , lambda x,y: None)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=download, num_workers=2, collate_fn=collate_fn , worker_init_fn=worker_init)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=download, num_workers=2, collate_fn=collate_fn, worker_init_fn = worker_init)

    return trainloader, valloader


def visualize_batch(batch , save_name = 'vis_batch' , verbose = False , PATH = CONFIG.STATS_log_dir,
        transform = None, prefix = 'agent'):
    """ Visualizes all episodes in one batch. """

    # Make save directory
    dir_path = os.path.join(PATH , save_name)
    if save_name == 'eval_0' and prefix == 'Det':
        dir_path = os.path.join(PATH , 'eval')
        os.makedirs(dir_path,exist_ok= True)
    elif 'eval' in save_name:
        dir_path = os.path.join(PATH , 'eval')

    elif prefix == 'agent':
        os.makedirs(dir_path)

    # Limit number of episodes to visualize per batch
    if CONFIG.MISC_vis_per_batch:
        nbr_vis = min(CONFIG.MISC_vis_per_batch , batch.idx)
    else:
        nbr_vis = batch.idx

    # Loop through each episode and visualize it
    for i in range( nbr_vis  ):
        if 'eval' in save_name:
            name = save_name + '_'+ prefix
        else:
            name = str(i)+ "_vis_" + prefix

        episode , _ , weights= batch.get_episode(i, batch.steps[i].int() + 1)
        visualize_trajectory(episode , save_name = name,verbose =True,
            transform = transform , PATH = dir_path)


def visualize_trajectory(episode,
                         save_name='visualization', verbose=False, PATH = CONFIG.STATS_log_dir ,
                         transform = None):
    """
    start_patch, goal_patch and agent_patches all have the format
    (h_topleft, w_topleft, height, width)

    - The patches need to be of numpy format
    - To plot a sequence of agent patches just insert the sequence like (len(seq),4)
    - To plot only one it is fine to insert of the shape (4)
    """


    start_patch = episode.locs[0]
    goal_patch = episode.loc_goal[0]
    full_img = episode.image.squeeze(0).squeeze(0)[0:3,:,:].cpu()
    agent_patches = episode.locs[1:, :]
    rewards = episode.weights.cpu()

    # Get size of the full image and patches
    H_img, W_img = CONFIG.MISC_im_size
    H_patches, W_patches = CONFIG.MISC_patch_size
    step_sz = CONFIG.MISC_step_sz

    # If we recieved a transform -> use it
    if transform is not None:
        full_img = transform(full_img)

    # Add image to plotting
    fig, ax = plt.subplots( figsize = (14, 8))
    ax.imshow(np.transpose(full_img, (1, 2, 0)))


    # Draw all the possible boxes
    for i in range(25):
        y, x = divmod(i,5)
        rect = patches.Rectangle((x * step_sz,y * step_sz),
                                 W_patches - 1,
                                 H_patches - 1,
                                 edgecolor = 'w',
                                 facecolor = 'none')
        ax.add_patch(rect)



    # Draw all the possible boxes
    for i in range(25):
        y, x = divmod(i,5)
        rect = patches.Rectangle((x * step_sz,y * step_sz),
                                 W_patches - 1,
                                 H_patches - 1,
                                 edgecolor = 'w',
                                 facecolor = 'none')
        ax.add_patch(rect)




    # Super important comment

    # Add start_patch and goal_patch
    # NOTE: patches.Rectangle's xy refers to y going from top to bottom, i.e. it
    # is "reversed" relative to the mathematical y-axis which would go from
    # bottom to top

    text_offset_x = CONFIG.MISC_patch_size[0] // 2
    text_offset_y = CONFIG.MISC_patch_size[0] // 2


    # Check if goal and final loc is same plot success as yellow rectangle instead
    final_iou = compute_iou(agent_patches[-1,:] , goal_patch)
    if final_iou > CONFIG.RL_done_iou:
        rect_goal_color = 'y'
    else:
        rect_goal_color = 'g'

    # rect_goal = patches.Rectangle(xy=(goal_patch[1], goal_patch[0]), width=goal_patch[3],
    #                               height=goal_patch[2], linewidth=1.5,
    #                               joinstyle = 'round', edgecolor=rect_goal_color, facecolor='none')
    # rect_start = patches.Rectangle(xy=(start_patch[1], start_patch[0]), width=start_patch[3],
    #                                height=start_patch[2], linewidth=1.5,
    #                                joinstyle = 'round', edgecolor='b', facecolor='none')

    rect_goal = patches.FancyBboxPatch(xy=(goal_patch[1], goal_patch[0]),
                                       width=goal_patch[3],
                                       height=goal_patch[2],
                                       linewidth=1.5,
                                       boxstyle = patches.BoxStyle("Round",
                                                                   pad=-13,
                                                                   rounding_size=5),
                                       edgecolor='none', facecolor='#97D077')
    rect_start = patches.FancyBboxPatch(xy=(start_patch[1], start_patch[0]),
                                       width=start_patch[3],
                                       height=start_patch[2],
                                       linewidth=1.5,
                                       boxstyle = patches.BoxStyle("Round",
                                                                   pad=-13,
                                                                   rounding_size=5),
                                       edgecolor='none', facecolor='#D0CEE2')

    ax.add_patch(rect_start)
    ax.text(start_patch[1] + text_offset_x, start_patch[0] + text_offset_y,
            "S", fontsize=23, color='w',rotation = 0,rotation_mode = 'anchor',
            horizontalalignment='center',verticalalignment='center')

    # Make sure that the agent-selected patches are of the corrrect dimensions
    #add one so that the loop is correct
    if len(agent_patches.shape) == 1:
        agent_patches = agent_patches[np.newaxis, :]

    # Also super importatnt comment

    # Add agent-selected patch(es)
    for i in range(agent_patches.shape[0]):
        agent_patch = agent_patches[i,:]
        # agent_rect = patches.Rectangle(xy=(agent_patch[1], agent_patch[0]), width=agent_patch[3],
        #                                height=agent_patch[2], linewidth=1.5, edgecolor='r', facecolor='none')
        # if final_iou < CONFIG.RL_done_iou or i < (agent_patches.shape[0]-1):
        #     ax.add_patch(agent_rect)
        ax.text(agent_patch[1] + 4*i + 4, agent_patch[0] + 6  , str(i + 1),
                horizontalalignment='left',verticalalignment='center',
                bbox=dict(boxstyle='circle',fc='#7EA6E0',ec='none'), fontsize=11, color='w')

        # Show IoU only in the last iteration which should be the convergent one
        if verbose and i == (agent_patches.shape[0]-1):
            dist = np.linalg.norm(get_regression_targets(agent_patch[None,:], goal_patch[None,:], normalize=False))
            iou = compute_iou(agent_patch, goal_patch)
            # Set text in top right corner for easier readability
            ax.text(W_img + 2, 2 , 'Final dist = '+str(round(dist,1)), fontsize=9, color='b')
            ax.text(W_img + 2, 12  , 'Final IoU = '+str(round(iou.item() ,  3)), fontsize=9, color='b')
        # Add option to show rewards for each step
        if verbose and rewards is not None:
            ax.text(W_img + 2 , 20 + 10*i , 'Reward %d: %2.1f'% (i, rewards[i]), fontsize = 8 , color='b')


    ax.add_patch(rect_goal)
    ax.text(goal_patch[1] + text_offset_x, goal_patch[0] + text_offset_y, "G",
            horizontalalignment='center',verticalalignment='center',
            fontsize=23, color='w',rotation = 0,rotation_mode = 'anchor')


    if save_name is not None:
        # If save name is none show image instead
        fig.savefig(os.path.join(PATH, save_name + '.png'))
        plt.cla()
        plt.clf()
        plt.close('all')
        gc.collect()
    else:
        plt.show()


def get_crop_distance(crop_loc1 , crop_loc2):
    dist = math.sqrt((crop_loc1[0] - crop_loc2[0])**2 + (crop_loc1[1] - crop_loc2[1])**2)
    return dist

def check_outside(crops_loc):
    H, W = CONFIG.MISC_patch_size
    im_H,im_W = CONFIG.MISC_im_size
    bools = torch.zeros(crops_loc.shape[0])
    for i,crop_loc in enumerate(crops_loc):
        bools[i] = (crop_loc[0] < -H) or (crop_loc[0] > im_H) or (crop_loc[1] < -W) or (crop_loc[1] > im_W)

    return bools




def project_into_image(crop_loc):
        H, W = CONFIG.MISC_patch_size
        im_H,im_W = CONFIG.MISC_im_size

        if crop_loc[0] < -H: crop_loc[0] = 0
        elif crop_loc[0] > im_H-H: crop_loc[0] = im_H-H

        if crop_loc[1] < -W: crop_loc[1] = 0
        elif crop_loc[1] > im_W-W: crop_loc[1] = im_W-W


        return crop_loc




# Used to select the latest results folder
def selectLatest(input):
    # Transform to correct datettime format
    dt = ''.join([input[0:10],input[10:19].replace('-',':')])
    dt = dt.replace('_','T')

    # Convert to date time and then to unix time
    dt = parse(dt)
    return time.mktime(dt.timetuple())



"""
Finds the latest directory in a folder sorted by the directory name format used in the logs.
"""
def find_latest_log(log_folder_path , n_latest = 0):
    folders = os.listdir(log_folder_path)

    # Add filtering for non directory filenames
    # Should not be any non-dirs here so might be unnecessary
    folders = list(filter( lambda d: os.path.isdir(os.path.join(log_folder_path , d)) , folders))

    # Sort it based on name
    try:
        folders.sort(key = selectLatest , reverse = True)
    except:
        print("Error when trying to sort the logs in find_latest_log")

    if n_latest >= len(folders):
        # Not that many logs available
        return None
    else:
        return folders[n_latest]



"""
Reads the contents of the config file and overwrites the current settings with the ones in the config file.
"""
def read_and_overwrite_config(config_file_path  , same_data = False):

    with open(config_file_path , 'r') as local_config:
        # Iterate over file lines
        for line in local_config:
            # If it does not start with CONFIG not relevant
            if line[0] != 'C' or "CONFIG." not in line:
                continue

            split = line.split('.')[1].strip(' #=').split('=')
            setting = split[0].strip(' #')

            # Check which setting
            if "RL_priv" in setting:
                # Get that setting
                value = eval(split[1].strip(' #'))
                CONFIG[setting] = value


            # We also need the patch size and image size to match
            if "MISC_patch_size" in setting or "MISC_im_size" in setting:
                value = eval(split[1].strip(' #'))
                CONFIG[setting] = value
            # If enabled will also read settings that make the trainset contain exact same
            # data as the model was trained on.
            if same_data and "MISC_random_seed" in setting or "MISC_dataset" == setting:
                value = eval(split[1].strip(' #'))
                CONFIG[setting] = value




def cosine_sim_heatmap(embeddings , grid_size = 16, pos = [1,2]):

    grid_size = grid_size - 1

    # Generate positional embeddings for all positions
    comb = torch.combinations( torch.arange(0, grid_size) , r = 2)
    # Add flipped
    comb = torch.cat((comb , comb.flip(dims=(1,))))
    comb = torch.cat((comb , torch.cat((torch.arange(0,grid_size).unsqueeze(1),torch.arange(0,grid_size).unsqueeze(1)),dim = 1))).long()

    pos_embedding = torch.flatten(embeddings[0,comb ] , start_dim = 1, end_dim = 2)

    # Plot heat map of one positional embedding compared to the rest
    selected_pos = torch.tensor(pos)
    selected_pos_emb = pos_embedding[ ( selected_pos == comb).all(dim = 1) , :]


    cosine_sim = torch.nn.CosineSimilarity( dim = 1)

    # Calculate
    cosine_similarities = cosine_sim( selected_pos_emb , pos_embedding)

    # Due to torch's inability to index with tensors properly we will have to inser
    # the values manually

    heat_map_tensor = torch.ones((grid_size , grid_size  ))

    for (i, sim) in enumerate(cosine_similarities):
        heat_map_tensor[comb[i,0] , comb[i,1]] = sim

    # Convert to numpy
    heat_map_numpy = heat_map_tensor.cpu().numpy()

    ax = sns.heatmap(heat_map_numpy)
    ax.set_title('Positional Encoding: (%d,%d)' % (pos[0] , pos[1]))
    plt.savefig(os.path.join(CONFIG.STATS_log_dir , "positional_embeddings" , "embedding_%d_%d.jpg" % (pos[0] , pos[1])))


    plt.cla()
    plt.clf()
    plt.close('all')
    gc.collect()



def calculate_cnn_output_size(layers, input_size, max_pool):
    """ OLD_NOT_USED : layer = [kernel , out_chan, padding, stride, in_chan]"""
    """ layer = [in_chan , out_chan, kernel, stride, padding]"""


    H , W = input_size
    C = layers[0][4]
    for layer in layers:
        H = (H + 2 * layer[4] - (layer[2] - 1) - 1 ) / layer[3] + 1
        W = (W + 2 * layer[4] - (layer[2] - 1) - 1 ) / layer[3] + 1
        C = layer[1]

        # If max_pool avaialble assume there is such a layer after each conv
        if max_pool is not None:
            H = (H - (max_pool[0] - 1) - 1) / max_pool[1] + 1
            W = (W - (max_pool[0] - 1) - 1) / max_pool[1] + 1

        H = math.floor(H)
        W = math.floor(W)

    return [H,W,C]



def crop_s(crop1 , crop2 ,T = None, DEBUG = True):
    """
        Simply plots two crops side by side. Usefull for debugging.
    """
    if DEBUG:
        # Assume in debug mode and switch to TkAgg backend
        matplotlib.use('TkAgg')

    # If a transform is supplied use it
    if T is not None:
        crop1 , crop2 = T(crop1) , T(crop2)

    fig, axes = plt.subplots(nrows = 1 , ncols = 2)

    axes[0].imshow(crop1.permute(1,2,0))
    axes[1].imshow(crop2.permute(1,2,0))

    axes[0].set_title("Crop 1")
    axes[1].set_title("Crop 2")

    # TODO

    if DEBUG:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close('all')
    gc.collect()


def get_correct_act(loc_start , loc_goal):
    """ Calculate which action (in ohe form) is the correct."""

    grid_correct_move = ( loc_goal[0,0:2] - loc_start[0,0:2]) / ( CONFIG.RL_softmax_step_size * torch.tensor(CONFIG.MISC_patch_size)).long()

    # Due to flooring of negative integers
    grid_correct_move = grid_correct_move.round()

    # remember first coordinate is y (positive downwards)
    dx, dy = grid_correct_move[1].item() , grid_correct_move[0].item()

    # now map this move to ohe
    if dx == 0 and dy == -1:
        i = 0 # Move up
    elif dx == 1 and dy == -1:
        i = 1 # Move up right
    elif dx == 1 and dy == 0:
        i = 2 # Move right
    elif dx == 1 and dy == 1:
        i = 3 # move right down
    elif dx == 0 and dy == 1:
        i = 4 # move down
    elif dx == -1 and dy == 1:
        i = 5 # Move down left
    elif dx == -1 and dy == 0:
        i = 6 # Move left
    elif dx == -1 and dy == -1:
        i = 7 # Move up left
    else:
        raise(Exception("Unknown move in get_correct_act. Is maximum distance really 1?"))

    correct_act = torch.zeros((1,8))
    correct_act[0,i] = 1

    return correct_act


def setupLogDir( log_dir_path):
    """
        Create log directory and save all current files. If pretrained network is enabled
        move the config files from that log directory to this one as well.
    """

    if not os.path.exists(CONFIG.STATS_log_dir_base):
        os.makedirs(CONFIG.STATS_log_dir_base)
    os.makedirs(CONFIG.STATS_log_dir, exist_ok = False)

    CONFIG.STATS_vis_dir = os.path.join(CONFIG.STATS_log_dir, "visualizations")
    os.makedirs(CONFIG.STATS_vis_dir)

    CONFIG.STATS_scripts_dir = os.path.join(CONFIG.STATS_log_dir, "saved_scripts")
    os.makedirs(CONFIG.STATS_scripts_dir)

    copyfile("training/train_agent.py" , os.path.join(CONFIG.STATS_scripts_dir, "train_agent.py"))

    copyfile("config.py" , os.path.join(CONFIG.STATS_scripts_dir , ".." , "config.py"))

    # Save Network files
    copyfile("networks/early_rl_agents.py" , os.path.join(CONFIG.STATS_scripts_dir , "early_rl_agents.py"))
    copyfile("networks/resnets.py" , os.path.join(CONFIG.STATS_scripts_dir, "resnets.py"))
    copyfile("networks/rnn_agents.py", os.path.join(CONFIG.STATS_scripts_dir, "rnn_agents.py"))

    # Save Utils files
    copyfile("utils/utils.py", os.path.join(CONFIG.STATS_scripts_dir, "utils.py"))
    copyfile("utils/training_utils.py" , os.path.join(CONFIG.STATS_scripts_dir , "training_utils.py"))
    copyfile("utils/agent_utils.py" , os.path.join(CONFIG.STATS_scripts_dir, "agent_utils.py"))

    # Create folder for saving intermediate models
    #os.makedirs(os.path.join(CONFIG.STATS_log_dir, "models"))

    CONFIG.STATS_metrics_dir = os.path.join(CONFIG.STATS_log_dir, "metrics")
    os.makedirs(CONFIG.STATS_metrics_dir)

    # If pretrained network is to be loaded save the config files from that folder as well
    if CONFIG.RL_priv_pretrained:

        CONFIG.STATS_pretrained_log_dir = os.path.join(CONFIG.STATS_log_dir, "pretrained")
        os.makedirs(CONFIG.STATS_pretrained_log_dir)

        embedder_type_mapping = dict([
            ("Doerch" , 'RL_pretrained_doerch_net'),
            ("ShareNet" , 'RL_pretrained_doerch_net'),
            ("Segmentations", 'RL_pretrained_segmentation_net')
            ])

        pre_trained_dir = CONFIG[embedder_type_mapping[CONFIG.RL_patch_embedder]]

        # Check if pretrained directory really exists
        if not os.path.exists(pre_trained_dir):
            raise(Exception("Unable to find pretrained directory:\t%s" , pre_trained_dir))

        pretrained_files = glob(pre_trained_dir + "/*.py")

        pretrained_files += glob(pre_trained_dir + "/*.json")

        for file in pretrained_files:
            dst_file = file[file.rfind("/")+1:]
            copyfile(file, os.path.join(CONFIG.STATS_pretrained_log_dir, dst_file))


def get_pos_goal_loc(loc , grid_size , distance):
    """
        Get all possible grid positions at a given distance from loc.
    """

    pos_grid = np.ones((grid_size,grid_size)) * -1
    try:
        pos_grid[loc[0],loc[1]] = 0
    except:
        pdb.set_trace()
    for d in range(1, distance + 1 ):
        # Get index of previous locations
        inds = np.asarray(np.where(pos_grid == (d - 1))).T

        # For each direction take a step and write new distance (if possible)
        for i in range(0,8):
            if i == 0:
                dir = np.array([[-1,0]])
            elif i == 1:
                dir = np.array([[-1,1]])
            elif i == 2:
                dir = np.array([[0,1]])
            elif i == 3:
                dir = np.array([[1,1]])
            elif i == 4:
                dir = np.array([[1,0]])
            elif i == 5:
                dir = np.array([[1,-1]])
            elif i == 6:
                dir = np.array([[0,-1]])
            elif i == 7:
                dir = np.array([[-1,-1]])

            inds_dir = inds + dir
            # Filter out all new locations that are outside grid
            inds_dir = inds_dir[((inds_dir >= 0) & (inds_dir < grid_size)).all(axis= 1),:]

            # Also filter out any new location already visited
            inds_dir = inds_dir[ (pos_grid[inds_dir[:,0] , inds_dir[:,1]] == -1), :]

            # Write new distance
            if len(inds_dir) > 0:
                pos_grid[inds_dir[:,0] , inds_dir[:,1]] = d

    # When completed find indicies of positions with distance
    arrays_distance = np.stack(np.where( pos_grid == distance)).T

    if len(arrays_distance) == 0:
        raise(Exception("Did not find any valid locations"))

    return arrays_distance

def sample_grid_games_fixed_distance( distance, n_games):
    """
        Samples n_games with fixed distance between start goal.
        Returns: n_games x 4 with start and goal loc
    """

    # Calculate grid size
    grid_size = CONFIG.MISC_im_size[0] // CONFIG.MISC_step_sz + 1

    # Determine if bounds are needed for sampling start
    if distance > grid_size // 2:
        # TODO
        start_bounds = 0
        # sample start locations

        # Due to distance being so large that not all start locations are possible
        # create a grid of valid start positions to sample from
        grid = np.reshape( np.arange(0,grid_size**2) , (grid_size, grid_size))

        grid_pos_dist = (grid_size - distance)

        # Check that grid_pos_dist is positive
        if grid_pos_dist <= 0:
            raise(ValueError("Distance equal to or larger than grid size."))

        grid[grid_pos_dist:grid_size - grid_pos_dist , grid_pos_dist:grid_size - grid_pos_dist] = -1
        grid = grid.flatten()
        grid = np.delete( grid, np.where( grid == -1))

        # Grid now contains 1D array of allowed start positions
        start_pos = np.random.choice( grid, size = n_games)

        start_locs = np.vstack( ( start_pos // grid_size, start_pos % grid_size)).T

    else:
        # No need for bounds sample start location
        start_locs = np.random.randint(0, grid_size, size = (n_games, 2))

    goal_locs = np.zeros_like(start_locs)

    # For each unique start location generate
    unique_start_locs = np.unique(start_locs , axis = 0)

    pos_goal_locs = {}

    for loc in unique_start_locs:
        # Calculate number of locs with this position
        locs_ind = (( start_locs == loc).all(axis = 1))
        pos_goal_locs = get_pos_goal_loc(loc, grid_size , distance)
        rand_inds = np.random.randint( pos_goal_locs.shape[0] , size = locs_ind.sum())

        # These are the sampled goal locations for this start positio
        local_goal_locs = pos_goal_locs[rand_inds, :]

        # take these goal locs and put them in the ot
        goal_locs[ np.where( locs_ind) , : ] = local_goal_locs

    return np.hstack( (start_locs,  goal_locs))
