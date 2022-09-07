
import math
import os
import time
import zipfile
import warnings
import gc
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.distributions import MultivariateNormal, OneHotCategorical

import matplotlib
import matplotlib.pyplot as plt

from torch.nn import CrossEntropyLoss


from config import CONFIG
from utils.utils import move_crop , get_deterministic_crops , compute_iou ,\
                        get_random_crops , get_frac_outside , visualize_trajectory , \
                        get_crop_distance , check_outside , project_into_image

def normalize_batch_weights(batch_weights , batch_dists):
    if CONFIG.RL_batch_size != 1 or CONFIG.RL_multiply_images != 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            batch_weights = normalize_grid(batch_weights, batch_dists)
    return batch_weights

def normalize_grid(batch_weights,batch_dists):
    # Due to pytorch lacking nanstd we have to convert to numpy to do te tings
    batch_dists_np = batch_dists.cpu().detach().numpy()
    batch_weights_np = batch_weights.cpu().detach().numpy()
    n_groups = 5
    step = 1
    lwr,upr = 0,step
    for i in range(n_groups):
        idx = np.all([(batch_dists_np <= upr) , (batch_dists_np >lwr)],axis =
                     0)

        # Calculate nanstd separatly to make sure that it is'nt zero anywhere
        nanstd = np.nanstd(batch_weights_np[idx] )
        if nanstd == 0.0:
            nanstd = 1

        # Normalize weights for each step of the agent separatly
        batch_weights_np[idx] = (batch_weights_np[idx] - np.nanmean(batch_weights_np[idx])) / nanstd

        # Move to the next set of distances
        lwr += step
        upr += step

    # Handle the largest as one group
    idx = batch_dists_np >lwr

    # Calculate nanstd separatly to make sure that it is'nt zero anywhere
    nanstd = np.nanstd(batch_weights_np[idx] , axis = 0)
    if nanstd == 0.0:nanstd = 1

    # Normalize weights for each step of the agent separatly
    batch_weights_np[idx] = (batch_weights_np[idx] - np.nanmean(batch_weights_np[idx])) / nanstd

    # Convert back to tensor and send to device
    batch_weights = torch.from_numpy(batch_weights_np).to(CONFIG.device)
    return batch_weights

def get_policy(agent_net, episode):

    # If softmax agent is enabled the polich is now a distribution over 8 different
    # directions in which the agent can move.

    # Get the output of the agent
    output, softmax_embedding = agent_net(episode)

    # Create policy distribution
    policy = OneHotCategorical(probs = output)

    return policy, softmax_embedding

def get_action(agent_net , episode, deterministic = False):
    if deterministic:
        action , softmax_embedding = agent_net(episode)
        return action, softmax_embedding
    else:
        policy , softmax_embedding = get_policy(agent_net, episode)
        samp = policy.sample()
        if not CONFIG.RL_agent_allowed_outside:
            outside = get_outside(episode).to(CONFIG.device)
            if (samp * outside).sum() == 1 :
                samp = policy.sample()

        return samp, softmax_embedding

def get_outside(episode):
    outside = torch.zeros([1,8])
    x,y = episode.locs[episode.step,:2]/CONFIG.MISC_step_sz
    if x == 0:
        outside[0,7] = 1
        outside[0,:2] = 1
    if y == 0:
        outside[0,5:] = 1
    if x == 4:
        outside[0,3:6] = 1
    if y == 4:
        outside[0,1:4] = 1
    return outside


def map_action_to_move(action):
    """ Maps the action which is a one hot encoded vector to a move in pixels."""

    # This will be the move in pixels
    c = torch.argmax(action).item()
    step_sz = int(CONFIG.RL_softmax_step_size * CONFIG.MISC_patch_size[0])
    # Translate selected action to a pixelwise move.
    # Remeber, increasing y coordinate means moving down in image
    if c == 0:
        dx , dy = 0,-1 # Move up
    elif c == 1:
        dx , dy = 1 , -1 # Move up right
    elif c == 2:
        dx , dy = 1 , 0 # Move right
    elif c == 3:
        dx , dy = 1 , 1 # Move down right
    elif c == 4:
        dx , dy = 0 , 1 # Move down
    elif c == 5:
        dx , dy = -1 , 1 # Move down left
    elif c == 6:
        dx , dy = -1 , 0 # Move left
    elif c == 7:
        dx , dy = -1 , -1 # Move up left
    else:
        raise(Exception("Invalid action:\t%d" % c))

    move = torch.tensor([dy , dx])

    # Now we have direction, multiply with patch size to get correct distance
    # Also hyperparameter to control step size
    move = step_sz * move

    return move

def take_step(action , episode, softmax_embedding=None):

    # Calculate the new location
    action_in = action
    # The action is a oneHotEncoding of in which direction the agent should move
    # Map the action to a move in (dx,dy) and add to previous position
    move = map_action_to_move(action)[None,:]

    loc_next = episode.loc_current.clone().detach()
    loc_next[0,0:2] += move[0,:]

    # Calculate the reward for this action
    reward = get_reward(loc_next, episode, action_in)

    # Check if the episode has been completed
    done = check_if_done(loc_next, episode)

    return loc_next, reward, done

def check_if_done(loc_next , episode):

    # If overlap with goal is significant we are done
    iou = compute_iou(loc_next, episode.loc_goal ).item()

    done = iou >= CONFIG.RL_done_iou

    # If we have reached the maximum number of steps the episode has ended
    return done or (episode.step + 1 >= CONFIG.RL_max_episode_length)

def get_reward(loc_next, episode, action):

    # Rewards are partially based on distances
    prev_dist = get_crop_distance(episode.loc_current[0], episode.loc_goal[0])
    next_dist = get_crop_distance(loc_next[0], episode.loc_goal[0])
    # TODO: Add max dist which is in regard to the goal and start patches
    max_dist = np.sqrt(np.prod(np.array(CONFIG.MISC_im_size) - np.array(CONFIG.MISC_patch_size)))

    iou = compute_iou(loc_next , episode.loc_goal).item()

    if iou > 0.2:
        reward = CONFIG.RL_reward_step + CONFIG.RL_reward_iou_scale * iou
    else:
        reward = CONFIG.RL_reward_step

    if iou > CONFIG.RL_done_iou:
        reward += CONFIG.RL_reward_goal
    elif episode.step + 1 >= CONFIG.RL_max_episode_length:
        reward += CONFIG.RL_reward_failed
    if ( prev_dist > next_dist):
        reward += CONFIG.RL_reward_closer
    if CONFIG.RL_reward_distance:
        reward += CONFIG.RL_reward_goal*(max_dist - next_dist)/ max_dist

    return reward

def update_net(batch , agent_net, optimizer, entropy_bonus = None):

    loss = 0

    # Log the entropy of taken action
    entropy_taken_actions = torch.zeros(CONFIG.RL_batch_size * CONFIG.RL_multiply_images * CONFIG.RL_max_episode_length)
    action_counter = 0
    eps_counter = 0

    # Get one trajectory, calculate loss for each time step and add to global loss
    for ep_id in range(batch.idx):
        eps_counter += 1
        for step_id in range(1 , batch.steps[ep_id].int() + 1):

            # Get the episode, the action and the weight
            ep , action , weight = batch.get_episode(ep_id , step_id)

            # Get the corresponding policy
            policy , softmax_embedding = get_policy(agent_net , ep)

            # Get log probability of taken action
            logp = policy.log_prob(action)

            # Add to loss with weight
            loss -= logp * weight

            # Calculate entropy for logging (and possibly for entropy bonus)
            # entropy = - policy.probs * policy.logits
            entropy = policy.entropy()

            entropy_taken_actions[action_counter] = entropy

            if entropy_bonus is not None and entropy_bonus != 0:
                loss -= entropy_bonus * entropy

            action_counter += 1

        # If the agent is of type RNN reset the hidden states
        if agent_net.AGENT_TYPE == 'RNN':
            agent_net.reset()

        # Update the network with the correct frequency
        if CONFIG.RL_nbr_eps_update == eps_counter:
            loss = loss / action_counter

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch.sc.s('Loss').collect( loss.item() )
            batch.sc.s('Entropy').collect(entropy_taken_actions[0:action_counter].mean().item())

            loss = 0
            action_counter = 0
            eps_counter = 0

    if  (CONFIG.RL_nbr_eps_update //2) <= eps_counter or (batch.idx == eps_counter ):

        loss = loss / action_counter

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch.sc.s('Loss').collect( loss.item() )
        batch.sc.s('Entropy').collect(entropy_taken_actions[0:action_counter].mean().item())

        loss = 0
        action_counter = 0
    else:
        pass
        #print("Skipping batch with %d episodes" % eps_counter)




def compute_loss(batch , agent_net, entropy_bonus = None):

    loss = 0

    # Log the entropy of taken action
    entropy_taken_actions = torch.zeros(CONFIG.RL_batch_size * CONFIG.RL_multiply_images * CONFIG.RL_max_episode_length)
    action_counter = 0

    # Get one trajectory, calculate loss for each time step and add to global loss
    for ep_id in range(batch.idx):
        for step_id in range(1 , batch.steps[ep_id].int() + 1):

            # Get the episode, the action and the weight
            ep , action , weight = batch.get_episode(ep_id , step_id)

            # Get the corresponding policy
            policy , softmax_embedding = get_policy(agent_net , ep)

            # Get log probability of taken action
            logp = policy.log_prob(action)

            # Add to loss with weight
            loss -= logp * weight

            # Calculate entropy for logging (and possibly for entropy bonus)
            # entropy = - policy.probs * policy.logits
            entropy = policy.entropy()

            entropy_taken_actions[action_counter] = entropy

            if entropy_bonus is not None and entropy_bonus != 0:
                loss -= entropy_bonus * entropy

            action_counter += 1

        # If the agent is of type RNN reset the hidden states
        if agent_net.AGENT_TYPE == 'RNN':
            agent_net.reset()

    # Log the entropy
    batch.sc.s('Entropy').collect(entropy_taken_actions[0:action_counter].mean().item())

    loss = loss / action_counter

    batch.sc.s('Loss').collect( loss.item() )

    return loss

def map_grid_dist_to_ohe( grid_dist):

    ohe = torch.zeros((1, 8))

    c = torch.zeros((1))

    #dist_diag = grid_dist * torch.tensor([[1],[1]]) / 1.4142
    #dist_diag_2 = grid_dist * torch.tensor([1,-1]) / 1.4142
    # For now correct step is diagonal if possible

    # grid_dist = dy , dx


    if grid_dist[0] < 0 and grid_dist[1] == 0:
        c[0] = 0 # up
    elif grid_dist[0] < 0 and grid_dist[1] > 0:
        c[0] = 1 # right up
    elif grid_dist[0] == 0 and grid_dist[1] > 0:
        c[0] = 2 # right
    elif grid_dist[0] > 0 and grid_dist[1] > 0:
        c[0] = 3 # right down
    elif grid_dist[0] > 0 and grid_dist[1] == 0:
        c[0] = 4 # down
    elif grid_dist[0] > 0 and grid_dist[1] < 0:
        c[0] = 5 # left down
    elif grid_dist[0] == 0 and grid_dist[1] < 0:
        c[0] = 6 # left
    elif grid_dist[0] < 0 and grid_dist[1] < 0:
        c[0] = 7
    else:
        raise(Exception("Invalid action:\t%s" % grid_dist))

    return c.long()


"""
def compute_loss(batch , agent_net, entropy_bonus = None):
    loss = 0

    # Log the entropy of taken action
    entropy_taken_actions = torch.zeros(CONFIG.RL_batch_size * CONFIG.RL_multiply_images * CONFIG.RL_max_episode_length)
    action_counter = 0

    # Get one trajectory, calculate loss for each time step and add to global loss
    for ep_id in range(batch.idx):
        for step_id in range(1 , batch.steps[ep_id].int() + 1):

            # Get the episode, the action and the weight
            ep , action , weight = batch.get_episode(ep_id , step_id)

            # Get the corresponding policy
            policy = get_policy(agent_net , ep)

            # Get log probability of taken action
            logp = policy.log_prob(action)

            # Add to loss with weight
            loss -= logp * weight

            # Calculate entropy for logging (and possibly for entropy bonus)
            # entropy = - policy.probs * policy.logits
            entropy = policy.entropy()

            entropy_taken_actions[action_counter] = entropy

            if entropy_bonus is not None and entropy_bonus != 0:
                loss -= entropy_bonus * entropy

            action_counter += 1

        # If the agent is of type RNN reset the hidden states
        if agent_net.AGENT_TYPE == 'RNN':
            agent_net.reset()


    # Log the entropy
    batch.sc.s('Entropy').collect(entropy_taken_actions[0:action_counter].mean().item())

    return loss/batch.idx
"""



"""
Calculates the rewards from step until finish given the reward of the trajectory.
"""

def rewards_to_go(rewards):

    rtg = torch.zeros_like(rewards).to(CONFIG.device)
    for i in range(len(rewards)):
        # First get gamma
        discount = torch.pow(CONFIG.RL_discount_factor , torch.arange(0 , len(rewards)-i)).to(CONFIG.device)
        rtg[i] = torch.sum( rewards[i:] * discount)

    # Normalize per action here?
    # Or in main loop?
    return rtg


""" Run a trajectory in a search area """
def run_eval_trajectory(image,episode, agent_net, deterministic = CONFIG.RL_eval_deterministic, loc_start = None, loc_goal = None, probs_diff = None):
    episode.initialize(image = image , loc_start=loc_start, loc_goal = loc_goal, probs_diff = probs_diff)

    # Execute episode
    done = False
    while not done:
        # Get an action from the agent
        action, softmax_embedding = get_action(agent_net, episode, deterministic)

        # Update the environment according to the correct action
        loc_next, reward, done = take_step(action, episode, softmax_embedding)

        # Get the crop at the current location
        crop_current, loc_current = get_deterministic_crops(image, coords=loc_next[0])

        # Update the episode storage
        try:
            tmp = torch.nn.Softmax( dim = 1 )(softmax_embedding).cpu().detach().numpy()
        except:
            tmp = np.zeros((1, 8))

        episode.update(action, reward, loc_current, crop_current, misc=tmp)

    # Episode done return results
    episode.finish()
    return episode


""" Used to freeze or unfreeze parts of the network """
def set_freezed_parts_of_net(net , mode = 'none'):

    # Mode determines which parts should be froozen
    # mode = 'patch' - Freezes patch embedder everything else unfroozen
    # mode = 'policy' - Freezes all that is not the patch embedder
    # mode = 'none' - unfreezes all parts of the network

    for child in net.children():
        if child == self.patch_emb:
            if mode == 'patch':
                for parameter in child.parameters():
                    parameter.requires_grad = False
                else:
                    parameter.requires_grad = True
        else:
            if mode == 'policy':
                for parameter in child.parameters():
                    parameter.requires_grad = False
                else:
                    parameter.requires_grad = True


def visualize_cnn_filter(conv_layer, filter = 0, save_name = 'filter_vis.png', show = True):
    """ Plots the weights of a filter in a convolutional layer."""

    input_channels_ploted = min( 16 , conv_layer.weight.shape[1])

    filter = conv_layer.weight[filter,:]

    filter_ = filter[0:input_channels_ploted,:,:].detach().clone().cpu().permute(1,2,0).numpy()

    n_rows = int(math.sqrt(input_channels_ploted))
    n_cols = int( input_channels_ploted / n_rows) + int( input_channels_ploted % n_rows != 0)

    matplotlib.use('TkAgg') if show else None

    fig, axes = plt.subplots(n_rows , n_cols)

    for (i , ax_inds) in enumerate(np.ndindex(axes.shape)):

        axes[ax_inds].imshow(filter_[:,:,i])
        axes[ax_inds].set_title("Input Channel %d" % i)

    if show:

        plt.show()

    if False: # for now....
        plt.savefig(os.path.join(CONFIG.STATS_log_dir, "filter_visualizations", save_name))

    # Clean up
    plt.cla()
    plt.clf()
    plt.close('all')
    gc.collect()
