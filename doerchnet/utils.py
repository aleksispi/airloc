import torch
import numpy as np
import torchvision.transforms as transforms
import os
import gc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import urllib
import matplotlib
import random

from torch.nn.functional import one_hot
from config import CONFIG

from utils.utils import get_deterministic_crops


def sample_doerch_crops(images):
    """
    note that if max_dist and min_iou are both provided, then only max_dist
    will be used. hence, for min_iou to take effect, max_dist has to be none.
    """

    # NEW IDEA! Only sample n/2 pairs of crops and for the next n/2 pairs pick the opposite
    # correct action as in the previous.

    # When below variable is enabled this new approach is used. False gives old
    NEW_SAMPLING_APPROACH = False

    # When enabled exactly same number of locs in the different positions
    EQUAL_GOAL_LOC_DISTRIBUTION = False

    if EQUAL_GOAL_LOC_DISTRIBUTION and NEW_SAMPLING_APPROACH:
        raise(Exception("EQUAL_GOAL_LOC_DISTRIBUTION does not work with NEW_SAMPLING_APPROACH"))

    # define some useful constants
    h, w = CONFIG.MISC_patch_size
    im_h, im_w = CONFIG.MISC_im_size
    actions_one_hot = torch.zeros((images.shape[0],8))
    n_chan = images.shape[1]
    n_imgs = images.shape[0]
    N = images.shape[0]


    # initialize memory for the crops size = (batch, n_chan, h_p, w_p)
    # keep the number of channels at a constant
    crops_goal = torch.zeros(size=(n_imgs, n_chan, h, w))
    crops_start = torch.zeros(size=(n_imgs, n_chan, h, w))

    loc_crops_goal = torch.zeros(size = ( n_imgs , 4))
    loc_crops_start = torch.zeros(size = (n_imgs , 4))

    if EQUAL_GOAL_LOC_DISTRIBUTION:
        for i in range(0, 8):
            actions_one_hot[(N//8 * (i)):(N//8) *(i+1), i] = 1
        # Randomize if any left
        for i in range(N//8 * 8, N):
            actions_one_hot[i , random.randint(0,7)] = 1

    if NEW_SAMPLING_APPROACH:

        N = images.shape[0]
        for i in range(N // 2 + N % 2): # If odd sample one more
            # image is divided into a static uniform grid. patches are sampled from this grid
            upper_h , upper_w = int(im_h / h) - 1  , int(im_w / w) - 1
            lower_h , lower_w = ( 0 , 0)

            grid_loc = np.floor(np.random.uniform( low = [lower_h ,lower_w] , high = [upper_h , upper_w]))
            goal_loc, action = sample_goal(grid_loc,upper_h,upper_w)
            actions_one_hot[i,action] = 1

            locs_start = np.concatenate(((grid_loc + 1/2) * np.array(CONFIG.MISC_patch_size) , np.array(CONFIG.MISC_patch_size)) ).astype('int64')
            locs_goal = np.concatenate(((goal_loc + 1/2) * np.array(CONFIG.MISC_patch_size) , np.array(CONFIG.MISC_patch_size)) ).astype('int64')
            loc_crops_start[i, :] = torch.tensor(np.array(locs_start, dtype = int))
            loc_crops_goal[i, :] = torch.tensor(np.array(locs_goal, dtype = int))

            crops_start[i, :n_chan, :, :] = transforms.functional.crop(images[i, :, :, :], top=locs_start[0], left=locs_start[1], height=locs_start[2], width=locs_start[3])
            crops_goal[i, :n_chan, :, :] = transforms.functional.crop(images[i, :, :, :], top=locs_goal[0], left=locs_goal[1], height=locs_goal[2], width=locs_goal[3])

        # Now we have sampled half the pairs we need. The next half should be sampled start crops
        # But inverse of previous correct action
        for i in range(N//2 + N % 2 , N):
            upper_h , upper_w = int(im_h / h) - 1  , int(im_w / w) - 1
            lower_h , lower_w = ( 0 , 0)


            grid_loc = np.floor(np.random.uniform( low = [lower_h ,lower_w] , high = [upper_h , upper_w]))

            # The following line is only difference
            goal_loc, action = opposite_goal(grid_loc,torch.argmax(actions_one_hot[i - N//2, :]).item())
            actions_one_hot[i,action] = 1

            locs_start = np.concatenate(((grid_loc + 1/2) * np.array(CONFIG.MISC_patch_size) , np.array(CONFIG.MISC_patch_size)) ).astype('int64')
            locs_goal = np.concatenate(((goal_loc + 1/2) * np.array(CONFIG.MISC_patch_size) , np.array(CONFIG.MISC_patch_size)) ).astype('int64')
            loc_crops_start[i, :] = torch.tensor(np.array(locs_start, dtype = int))
            loc_crops_goal[i, :] = torch.tensor(np.array(locs_goal, dtype = int))

            crops_start[i, :n_chan, :, :] = transforms.functional.crop(images[i, :, :, :], top=locs_start[0], left=locs_start[1], height=locs_start[2], width=locs_start[3])
            crops_goal[i, :n_chan, :, :] = transforms.functional.crop(images[i, :, :, :], top=locs_goal[0], left=locs_goal[1], height=locs_goal[2], width=locs_goal[3])
    else:
        # The old approach, all samples are uniform
        for i in range(images.shape[0]):
            # image is divided into a static uniform grid. patches are sampled from this grid
            upper_h , upper_w = int(im_h / h) - 2  , int(im_w / w) - 2
            lower_h , lower_w = ( 1 , 1)


            grid_loc = np.floor(np.random.uniform( low = [lower_h ,lower_w] , high = [upper_h , upper_w]))
            if EQUAL_GOAL_LOC_DISTRIBUTION:
                goal_loc = map_grid_action_to_goal(grid_loc, torch.argmax(actions_one_hot[i, :]))
            else:
                goal_loc, action = sample_goal(grid_loc,upper_h,upper_w)
                actions_one_hot[i,action] = 1

            locs_start = np.concatenate(((grid_loc + 1/2) * np.array(CONFIG.MISC_patch_size) , np.array(CONFIG.MISC_patch_size)) ).astype('int64')
            locs_goal = np.concatenate(((goal_loc + 1/2) * np.array(CONFIG.MISC_patch_size) , np.array(CONFIG.MISC_patch_size)) ).astype('int64')
            loc_crops_start[i, :] = torch.tensor(np.array(locs_start, dtype = int))
            loc_crops_goal[i, :] = torch.tensor(np.array(locs_goal, dtype = int))

            crops_start[i, :n_chan, :, :] = transforms.functional.crop(images[i, :, :, :], top=locs_start[0], left=locs_start[1], height=locs_start[2], width=locs_start[3])
            crops_goal[i, :n_chan, :, :] = transforms.functional.crop(images[i, :, :, :], top=locs_goal[0], left=locs_goal[1], height=locs_goal[2], width=locs_goal[3])

    return crops_start,crops_goal,actions_one_hot ,loc_crops_start , loc_crops_goal


def get_label(loc_start,loc_goal, dim = 8):
    """
        Given start and goal locations outputs the label for the doerchnet to predict.

        Dimensions:
            dim = 8: Eight possible locations close to start patch
            dim = 25: Absolute prediction of location of goal patch
            dim = 80: Relative prediction of location of goal patch compared to start patch.

    """

    H, W = CONFIG.MISC_patch_size
    step_sz = int(CONFIG.RL_softmax_step_size*H)

    if dim == 8:
        diff = (loc_goal[:,:2]- loc_start[:,:2] )/step_sz
        diff += 1
        inds = diff[:,0]*3 + diff[:,1]
        actions = torch.zeros_like(inds)
        for i,inds in enumerate(inds):
            if inds == 0: actions[i] = 7
            elif inds == 1: actions[i] = 0
            elif inds == 2: actions[i] = 1
            elif inds == 3: actions[i] = 6
            elif inds == 4: raise(Exception("Same start and goal loc"))
            elif inds == 5: actions[i] = 2
            elif inds == 6: actions[i] = 5
            elif inds == 7: actions[i] = 4
            elif inds == 8: actions[i] = 3
        actions = one_hot(actions.long(), dim).float()
    elif dim == 25:
       diff = loc_goal[:,:2]/step_sz
       actions = diff[:,0]*5 + diff[:,1]
       actions = one_hot(actions.long(), dim).float()
    elif dim == 80:
        # 0 - 80 from top left to lower right
        # TODO Why not signed?
        move = ((loc_goal[:,:2] - loc_start[:,:2]) / step_sz )
        actions = torch.zeros((move.shape[0] , 9,9 ), dtype = torch.long) # NOTE - Current position still here
        actions[torch.arange(move.shape[0]) , (4 + move[:,0]).long() , (4 + move[:,1]).long()] = 1

        # Reshape to one hot encoding
        actions = torch.flatten(actions , start_dim = 1)

        # Check if any start and goal is at same position
        if (actions[:, 4 * 9 + 4] == 1).any():
            raise(Exception("Same start and goal location in get_label"))
        else:
            # Remove current position from label space
            actions = torch.cat((actions[:,0:40] , actions[:,41:]), dim = 1).float()
    else:
        raise(Exception("UNexpected dimension in 'get_label':\t%d" % dim))

    return actions

def opposite_goal(grid_loc, prev_action):
    """ Select the opposing location."""

    # 8 possible directions. Add four do modulo to find opposing side
    action_idx = (prev_action + 4) % 8

    goal_loc = map_grid_action_to_goal(grid_loc, action_idx)

    return goal_loc, action_idx

def map_grid_action_to_goal(grid_loc , action):

    step = CONFIG.RL_softmax_step_size

    goal_loc = grid_loc.copy()

    if   action  == 0: goal_loc += [-step,0]
    elif action  == 1: goal_loc += [-step,step]
    elif action  == 2: goal_loc += [0,step]
    elif action  == 3: goal_loc += [step,step]
    elif action  == 4: goal_loc += [step,0]
    elif action == 5: goal_loc += [step,-step]
    elif action == 6: goal_loc += [0,-step]
    elif action == 7: goal_loc += [-step,-step]

    return goal_loc

def sample_goal(grid_loc,upper_h,upper_w):

    probs = np.ones((8))
    # ensure no stepping outside the image
    # TODO:Note Limit is dependent on upperh and upperw
    if grid_loc[0] <= 0:
        probs[-1] = 0
        probs[0:2] = 0
        raise(Exception("Setting action probabilites to 0 in sample_goal"))
    if grid_loc[1] <= 0:
        probs[ 5:] = 0
        raise(Exception("Setting action probabilites to 0 in sample_goal"))
    if grid_loc[0] >= upper_h:
        probs[ 3:6] = 0
        raise(Exception("Setting action probabilites to 0 in sample_goal"))
    if grid_loc[1] >= upper_w:
        probs[ 1:4] = 0
        raise(Exception("Setting action probabilites to 0 in sample_goal"))

    probs = probs/np.sum(probs)
    action_idx  = np.random.choice(range(8), p=probs)

    goal_loc = map_grid_action_to_goal(grid_loc , action_idx)

    return goal_loc, action_idx
def visualize_batch_doerch( imgs , locs_start , locs_goal , actions, transform = None , PATH = "vis" , save_name = 'vis', max_imgs = 8):
    """
        Visualizes results from an entire batch during DoerchNet pretraining.
    """

    n_imgs = min(imgs.shape[0] , max_imgs)

    # Create new directory to save this batch visualization in
    save_dir = os.path.join(PATH , save_name)
    os.makedirs(save_dir)

    imgs , locs_start, locs_goal , actions = imgs.cpu() , locs_start.cpu() , locs_goal.cpu() , actions.cpu()


    # For each image make visualization
    for i in range(n_imgs):

        visualize_doerch(imgs[i,:] , locs_start[i,:], locs_goal[i,:], actions[i,:], transform = transform , PATH = save_dir, save_name = "vis_%d" % i)


def visualize_doerch(img ,  loc_start , loc_goal , action , transform = None, PATH = '.' , save_name = 'vis'):

    """
    # Select first image:
    action = actions[0,:]
    loc_start = locs_start[0,:].cpu()
    loc_goal = locs_goal[0,:].cpu()

    img = imgs[0,:].detach().cpu()
    """



    if transform is not None:
        img = transform(img)

    patch_size_tensor = torch.tensor(CONFIG.MISC_patch_size).cpu()

    action_idx = torch.argmax(action).item()
    loc_action = loc_start.detach().clone().cpu()
    # given the action find the choosen location
    if action.shape[0] == 8:
        if   action_idx == 0: loc_action[0:2] += torch.tensor([-1.1,0] ) * patch_size_tensor
        elif action_idx == 1: loc_action[0:2] += torch.tensor([-1.1,1.1] ) * patch_size_tensor
        elif action_idx == 2: loc_action[0:2] += torch.tensor([0,1.1]) * patch_size_tensor
        elif action_idx == 3: loc_action[0:2] += torch.tensor([1.1,1.1] ) * patch_size_tensor
        elif action_idx == 4: loc_action[0:2] += torch.tensor([1.1,0]) * patch_size_tensor
        elif action_idx == 5: loc_action[0:2] += torch.tensor([1.1,-1.1]  ) * patch_size_tensor
        elif action_idx == 6: loc_action[0:2] += torch.tensor([0,-1.1] )* patch_size_tensor
        elif action_idx == 7: loc_action[0:2] += torch.tensor([-1.1,-1.1] ) * patch_size_tensor
    elif action.shape[0] == 25:
        x,y = divmod(action_idx,5)
        loc_action[0:2] = torch.tensor([x,y] ) * int(CONFIG.RL_softmax_step_size*CONFIG.MISC_patch_size[0])
    elif action.shape[0] == 80:
        if action_idx >= 40: action_idx += 1
        x,y = divmod(action_idx,9)
        x -= 4
        y -= 4
        loc_action[0:2] += torch.tensor([x,y] ) * int(CONFIG.RL_softmax_step_size*CONFIG.MISC_patch_size[0])
    else:
        raise(Exception("Unknown action space"))

    # make sure integer value
    loc_action = loc_action.long()

    fig = plt.figure(figsize = (10, 6))

    subfigs = fig.subfigures(1, 2)

    ax = subfigs[0].subplots( )
    ax.imshow(np.transpose(img, (1, 2, 0)))

    rect_start = patches.Rectangle(xy=(loc_start[1], loc_start[0]), width=loc_start[3],
                                   height=loc_start[2], linewidth=2, edgecolor='b', facecolor='none')
    rect_goal = patches.Rectangle(xy=(loc_goal[1], loc_goal[0]), width=loc_goal[3],
                                  height=loc_goal[2], linewidth=2, edgecolor='g', facecolor='none')
    rec_action = patches.Rectangle(xy=(loc_action[1], loc_action[0]), width=loc_action[3],
                                  height=loc_action[2], linewidth=1, edgecolor='y', facecolor='none')

    # Add rectangles
    ax.add_patch(rect_start)
    ax.add_patch(rect_goal)
    ax.add_patch(rec_action)

    offset = CONFIG.MISC_patch_size[0] // 4
    # TODO - Add text ???
    ax.text(loc_start[1] + offset, loc_start[0] + offset + 5, f"Start", fontsize=18, color='w',rotation = 315,rotation_mode = 'anchor')
    ax.text(loc_goal[1] + offset, loc_goal[0] + offset + 0 , f"Target", fontsize=18, color='w',rotation = 315,rotation_mode = 'anchor')
    ax.text(loc_action[1] + offset, loc_action[0] + offset + 5, f"Agent", fontsize=18, color='w',rotation = 315,rotation_mode = 'anchor')

    # Plot start and goal patch to the right
    right_axes = subfigs[1].subplots(nrows = 2 , ncols = 1)

    # get starat and goal crops
    start_crop , _ = get_deterministic_crops(img.unsqueeze(0) , loc_start)
    goal_crop , _ = get_deterministic_crops(img.unsqueeze(0) , loc_goal)

    right_axes[0].imshow(start_crop.squeeze(0).permute(1,2,0))
    right_axes[1].imshow(goal_crop.squeeze(0).permute(1,2,0))

    right_axes[0].set_title("Start Crop")
    right_axes[1].set_title("Goal Crop")

    # Save figure
    fig.savefig(os.path.join(PATH, save_name + '.png'))

    # Close and clear up
    plt.cla()
    #plt.clf()
    plt.close('all')
    gc.collect()

def calculate_precision( outputs , labels):
    """
        Calcualte accuracy(precision) of model for current batch for corners, boundaries and middle area.c:w
    """

    corners = [0,4,20,24]
    boundaries = [1,2,3,5,10,15,9,14,19,21,22,23]
    middle = [6,7,8,11,12,13,16,17,18]

    # Calcualte corner precision
    corners_ind = labels[:,corners].any(dim = 1)
    prec_corner = ( torch.argmax(outputs[corners_ind , :] , dim = 1) == torch.argmax(labels[corners_ind,:], dim = 1)).float().mean()

    # Calcualte boundary precision
    boundaries_ind = labels[:,boundaries].any(dim = 1)
    prec_boundaries = ( torch.argmax(outputs[boundaries_ind , :] , dim = 1) == torch.argmax(labels[boundaries_ind,:], dim = 1)).float().mean()

    # Calcualte corner precision
    middle_ind = labels[:,middle].any(dim = 1)
    prec_middle = ( torch.argmax(outputs[middle_ind , :] , dim = 1) == torch.argmax(labels[middle_ind,:], dim = 1)).float().mean()

    return prec_corner.item() , prec_boundaries.item() , prec_middle.item()


if __name__ == '__main__':
    # DEBUGGING get_label
    H = 48
    step_sz = int(CONFIG.RL_softmax_step_size*H)
    loc1 = torch.tensor([[0,0]])
    loc2 = torch.tensor([[0,1]]) * step_sz
    loc3 = torch.tensor([[1,1]]) * step_sz
    loc4 = torch.tensor([[0,3]]) * step_sz

    comb1 = torch.cat((loc1,loc2))
    comb2 = torch.cat((loc1,loc1))
    comb3 = torch.cat((loc2 , loc3))

    label1 = get_label(loc1, loc2, dim = 80)
    label2 = get_label(loc3, loc1, dim = 80)
    label3 = get_label(loc1, loc3, dim = 80)

    label4 = get_label(comb1 , comb3, dim = 80)
    label5 = get_label(comb2 , comb3 , dim = 80)
    label6 = get_label(comb1, comb3 , dim = 80)

    assert(label1[0,4*9 + 4] == 1)
    assert(label2[0,3*9 + 3] == 1)
    assert(label3[0,5*9 + 4] == 1)

    pass
    label7 = get_label(comb1, comb2)
