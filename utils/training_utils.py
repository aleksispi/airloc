import torch
import sys
import numpy as np
import torchvision.transforms as transforms
from copy import deepcopy
import time
import math
import torchvision.transforms.functional as F
from torchvision.transforms import Resize

from utils.agent_utils import rewards_to_go, normalize_batch_weights

from utils.utils import get_random_crops, get_deterministic_crops, compute_iou,\
                        get_correct_act, sample_grid_games_fixed_distance

from doerchnet.utils import sample_doerch_crops
from config import CONFIG


class BatchStorage():
    """ Class used as storage container for information generated during one batch."""

    def __init__(self,sc, CONFIG = CONFIG ):
        self.CONFIG = CONFIG
        self.device = CONFIG.device
        self.ep_len = self.CONFIG.RL_max_episode_length
        self.n_eps = self.CONFIG.RL_batch_size * self.CONFIG.RL_multiply_images
        self.p_H, self.p_W = self.CONFIG.MISC_patch_size
        self.im_H, self.im_W = self.CONFIG.MISC_im_size
        self.sc = sc
        self.proptime = 0

        self.max_len_batch = self.ep_len * self.n_eps


    def initialize(self, batch_size=None):
        # Allocate memory for all inputs, priv info, trainign tensors
        if batch_size: self.n_eps = batch_size
        # This index is the number of episodes processed
        self.idx = 0

        self.difficulty = torch.zeros(self.n_eps)

        # Allocate for those that are always active
        self.weights = torch.zeros((self.n_eps,self.ep_len))
        # Need this nan stuff to make it work
        self.weights[:,:] = torch.nan
        # TODO: Maybe add an extra dimension if we need duplicate storage
        self.locs = torch.zeros((self.n_eps , self.ep_len + 1, 4))
        self.locs_goal = torch.zeros((self.n_eps, 4))
        # Check if instance segmentation input is enabled if so add one extrac channel
        n_chan_images = 3
        if CONFIG.RL_priv_use_seg:
            if CONFIG.MISC_dataset in ['dubai']:
                n_chan_images = 6
            elif CONFIG.MISC_dataset in ['masa', 'masa_filt']:
                n_chan_images = 4
            else:
                raise(Exception("Define which type of segmentation this dataset has"))

        self.image = torch.zeros((self.n_eps, n_chan_images, self.im_H , self.im_W))

        # Account for the extra channel tracking where the image ends
        n_chan_crops = n_chan_images

        # Add extra one crop for final position after final action
        self.crops = torch.zeros((self.n_eps, self.ep_len + 1, n_chan_crops, self.p_H, self.p_W)).to(self.CONFIG.device)

        # If softmax enabled the action is onehotencoding of 8 possible direcitons
        self.actions = torch.zeros((self.n_eps, self.ep_len , 8)).to(self.CONFIG.device)

        # Distance from current crop to goal for all action-reward-state tuples
        self.dists = torch.zeros((self.n_eps, self.ep_len)).to(self.device)
        self.dists[:,:] = torch.nan
        self.iou = torch.zeros((self.n_eps,))
        self.has_converged = torch.zeros((self.n_eps,))

        self.crops_goal = torch.zeros((self.n_eps , n_chan_crops, self.p_H, self.p_W)).to(self.CONFIG.device)

        if CONFIG.RL_priv_grid_location:
            self.grid_loc = torch.zeros((self.n_eps , self.ep_len , int(self.im_H / self.p_H) , int(self.im_W/self.p_W))).to(CONFIG.device)
            self.grid_curr_loc = torch.zeros((self.n_eps , self.ep_len , int(self.im_H / self.p_H) , int(self.im_W/self.p_W))).to(CONFIG.device)

        # Batch statistics
        self.steps = torch.zeros(( self.n_eps , ))
        self.iou = torch.zeros(( self.n_eps , ))
        self.hasConverged = torch.zeros(( self.n_eps , ))

        self.final_distance = torch.zeros((self.n_eps , ))
        self.final_distance[:] = torch.nan

        self.distance = torch.zeros(( self.n_eps , ))
        self.cumulativeRewardToGo = torch.zeros(( self.n_eps , ))

        # Set all values to nan. Only use episodes which were successfull for mean calc
        self.stepRatio = torch.zeros(( self.n_eps , ))
        self.stepRatio[:] = torch.nan

        self.time = torch.zeros(( self.n_eps , ))

    def append_episode(self, episode):
        """ Add all information from one episode to this batch """

        # First insert state, reward, action tuples
        self.steps[self.idx] = episode.step
        self.crops[self.idx ,:episode.step +1,: ,:,:] = episode.crops
        self.weights[self.idx , :episode.step] = episode.weights
        self.actions[self.idx , :episode.step,:] = episode.actions
        self.dists[self.idx , :episode.step] = episode.dists

        # If agent did not succeed add final dist
        if episode.iou.item() < CONFIG.RL_done_iou:
            self.final_distance[self.idx] = (( episode.loc_goal[0,:2] - episode.locs[-1,:2]) / int(CONFIG.MISC_patch_size[0] * CONFIG.RL_softmax_step_size)).abs().int().max().item()

        self.locs[self.idx , :episode.step + 1, :] = episode.locs
        self.locs_goal[self.idx , :] = episode.loc_goal
        self.iou[self.idx] = episode.iou
        self.hasConverged[self.idx] = 1.0 if episode.iou >= CONFIG.RL_done_iou else 0.0
        self.distance[self.idx] = episode.dists[-1]
        self.cumulativeRewardToGo[self.idx] = episode.weights[0]

        # Only add value for step ratio if agent was successfull this episode
        if episode.iou >= CONFIG.RL_done_iou:
            self.stepRatio[self.idx] = episode.min_steps / episode.step

        self.time[self.idx] = episode.time
        self.image[self.idx, :,:,:] = episode.image

        # The difficulty of the played game
        self.difficulty[self.idx] = episode.dists[0]

        if CONFIG.RL_priv_grid_location:
            self.grid_loc[self.idx , :,:,:] = episode.grid_loc
            self.grid_curr_loc[self.idx ,:,:,:] = episode.grid_curr_loc

        self.crops_goal[self.idx , :,:,:] = episode.crop_goal

        self.idx += 1

    def prepare_for_loss(self):
        # Do any necessary prepartions for loss computationo
        # For now only normalize the weights
        self.weights = normalize_batch_weights(self.weights, self.dists).detach()

    def get_episode(self, ep_id , step_id):
        """ Retrieves a detached episode from the batch."""

        ep = EpisodeStorage(CONFIG = self.CONFIG)
        ep.initialize(self.image[ep_id,:,:,:][None,:].clone())

        ep.crops = self.crops[ep_id , 0:step_id , :,:,:].clone()
        # Set episode values
        ep.crop_goal = self.crops_goal[ep_id,:,:,:].clone().unsqueeze(0)
        ep.locs = self.locs[ep_id , :step_id ,:]
        ep.loc_goal = self.locs_goal[ep_id, :].unsqueeze(0)
        ep.weights = self.weights[ep_id,:step_id]

        # Set correct number of steps taken
        ep.step = step_id-1

        if CONFIG.RL_priv_grid_location:
            ep.grid_loc = self.grid_loc[ep_id]
            ep.grid_curr_loc = self.grid_curr_loc[ep_id]

        if step_id > self.steps[ep_id]:
            # Get entire episode, no action in final state
            action = None
            weight = None
        else:
            action = self.actions[ep_id,step_id-1,:]
            weight = self.weights[ep_id,step_id-1]

        # Return state (episode), the taken action, and the weight for this tuple
        return ep , action, weight

    def store(self, mode = '', eval=False):
        """
        Stores the statistics for the batch
        """
        batch_final_distance = self.final_distance.nanmean()
        if not  math.isnan(batch_final_distance.item()):
            self.sc.s(mode + 'FinalDistanceOnlyFailure').collect(batch_final_distance.item())
            batch_final_distance = torch.tensor([0.0])

        self.sc.s(mode + 'Steps').collect(self.steps.mean().item())
        self.sc.s(mode + 'IoU').collect(self.iou.mean().item())
        self.sc.s(mode + 'CumulativeRewardToGo').collect(self.cumulativeRewardToGo.mean().item())
        self.sc.s(mode + 'HasConverged').collect(self.hasConverged.float().mean().item())

        inds = self.dists[:,0].repeat(6,1).cpu() == torch.as_tensor([[1],[2],[3],[4],[5],[6]])

        temp = np.array([self.steps[inds[0]].mean().item(),
                         self.steps[inds[1]].mean().item(),
                         self.steps[inds[2]].mean().item(),
                         self.steps[inds[3]].mean().item(),
                         self.steps[inds[4]].mean().item(),
                         self.steps[inds[5]].mean().item(),
                        ])
        self.sc.s(mode + 'SeparatedSteps').collect(temp)

        temp = np.array([self.iou[inds[0]].mean().item(),
                         self.iou[inds[1]].mean().item(),
                         self.iou[inds[2]].mean().item(),
                         self.iou[inds[3]].mean().item(),
                         self.iou[inds[4]].mean().item(),
                         self.iou[inds[5]].mean().item(),
                        ])
        self.sc.s(mode + 'SeparatedIoU').collect(temp)

        temp = np.array([self.cumulativeRewardToGo[inds[0]].mean().item(),
                         self.cumulativeRewardToGo[inds[1]].mean().item(),
                         self.cumulativeRewardToGo[inds[2]].mean().item(),
                         self.cumulativeRewardToGo[inds[3]].mean().item(),
                         self.cumulativeRewardToGo[inds[4]].mean().item(),
                         self.cumulativeRewardToGo[inds[5]].mean().item(),
                        ])
        self.sc.s(mode + 'SeparatedCumulativeRewardToGo').collect(temp)

        temp = np.array([self.hasConverged[inds[0]].mean().item(),
                         self.hasConverged[inds[1]].mean().item(),
                         self.hasConverged[inds[2]].mean().item(),
                         self.hasConverged[inds[3]].mean().item(),
                         self.hasConverged[inds[4]].mean().item(),
                         self.hasConverged[inds[5]].mean().item(),
                        ])
        self.sc.s(mode + 'SeparatedHasConverged').collect(temp)

        # Store the relative difficulty of the played games
        relative_diff = np.array([
            ((self.difficulty == 1).sum() / self.n_eps).item() ,
            ((self.difficulty == 2).sum() / self.n_eps).item() ,
            ((self.difficulty == 3).sum() / self.n_eps).item() ,
            ((self.difficulty == 4).sum() / self.n_eps).item() ,
            ])

        self.sc.s(mode + 'Difficulty').collect(relative_diff)

        batch_step_ratio = self.stepRatio.nanmean()
        if not math.isnan(batch_step_ratio.item()):
            self.sc.s(mode + 'StepRatioOnlySuccess').collect(batch_step_ratio.item())
            batch_step_ratio = torch.tensor([0.0]).cpu()
        #self.sc.s(mode + 'StepTime').collect(self.time.float().mean().item())
        #self.sc.s(mode + 'PropTime').collect(self.proptime)

        temp_actions_taken = torch.flatten(self.actions , start_dim = 0, end_dim = 1)
        temp_actions_taken = temp_actions_taken[temp_actions_taken.sum(dim = 1) != 0 , : ].mean(dim =  0)
        self.sc.s(mode + 'ActionsTaken').collect( temp_actions_taken.cpu().numpy())

        #temp_correct_actions = self.correct_act[ self.correct_act.sum(dim = 1) !=  0 , :].mean(dim = 0)
        #self.sc.s(mode + 'CorrectActions').collect(temp_correct_actions.cpu().numpy())
        if eval:
            self.sc.s(mode + 'GoalLoc').collect(self.locs_goal[0].numpy())
            self.sc.s(mode + 'ActionProbs').collect(self.actions[0].cpu().numpy())


class EpisodeStorage():
    """ Class used as storage container for all information generated during run of one episode."""

    def __init__(self, CONFIG=CONFIG):
        self.CONFIG = CONFIG
        self.device = CONFIG.device
        self.ep_len = self.CONFIG.RL_max_episode_length +1
        self.p_H,self.p_W = self.CONFIG.MISC_patch_size
        self.im_H,self.im_W = self.CONFIG.MISC_im_size
        self.misc = []

        # If grid_game is enabled calculate grid_size
        if CONFIG.MISC_grid_game:
            self.grid_size = (int(self.im_H / self.p_H) , int(self.im_W / self.p_W))

    def initialize(self, image, loc_goal=None, loc_start=None, probs_diff=None):
        # if enabled we have received a signal interrupt and should exit
        if CONFIG.TERMINATE:
            sys.exit(1)

        image = image.detach()
        self.image = image
        self.step = 0
        self.time = time.perf_counter()
        self.misc = []

        # Allocate for those that are always active
        self.weights = torch.zeros((self.ep_len-1)).to(self.CONFIG.device)
        # TODO: Maybe add an extra dimension if we need duplicate storage

        # If softmax agent enabled the action is a onehotencoding
        self.actions = torch.zeros((self.ep_len-1 , 8)).to(self.CONFIG.device)

        self.locs = torch.zeros((self.ep_len , 4))

        # Check if instance segmentation input is enabled
        n_chan_images = 3
        if CONFIG.RL_priv_use_seg:
            if CONFIG.MISC_dataset in ['dubai']:
                n_chan_images = 6
            elif CONFIG.MISC_dataset in ['masa', 'masa_filt']:
                n_chan_images = 4
            else:
                raise(Exception("Define which type of segmentation this dataset has"))

        # Take care of extra channel for some of the crops
        n_chan_crops = n_chan_images

        self.dists = torch.zeros((self.ep_len))
        self.crops = torch.zeros((self.ep_len, n_chan_crops, self.p_H, self.p_W)).to(self.CONFIG.device)

        # Sample or load the start and goal crops
        if loc_goal is not None and loc_start is not None:
            self.loc_goal = loc_goal
            self.loc_start = loc_start

            self.crop_goal,self.loc_goal = get_deterministic_crops(image,loc_goal)

            # Either sample or load the start patch
            self.crops[self.step,:,:,:] ,self.loc_start= get_deterministic_crops(image,loc_start)
        else:

            if probs_diff is None:

                self.crops[0,:,:,:] , self.loc_start = get_random_crops(image)

                self.crop_goal, self.loc_goal = get_random_crops(image,self.loc_start.cpu(),
                                 max_dist= self.CONFIG.RL_max_start_goal_dist,
                                 min_iou = self.CONFIG.RL_min_start_goal_iou)
                self.correct_act = torch.zeros(1,8)
                #self.crops[0,:,:,:], self.crop_goal, self.correct_act, self.loc_start,self.loc_goal = sample_doerch_crops(image)
            else:
                # Sample game with random difficulty according to vector
                diff = np.random.choice(4, p = probs_diff.numpy()) + 1

                game = sample_grid_games_fixed_distance( diff, 1) *  CONFIG.MISC_step_sz

                self.crops[0,:,:,:], self.loc_start = get_deterministic_crops(image, game[:,0:2])
                self.crop_goal , self.loc_goal = get_deterministic_crops(image, game[:,2:])
                self.correct_act = torch.zeros(1,8)


        self.crop_goal = self.crop_goal.to(self.CONFIG.device)
        self.min_steps = ((self.loc_goal[0,:2] - self.loc_start[0,:2])/int(self.p_H*CONFIG.RL_softmax_step_size)).abs().int().max()
        if CONFIG.MISC_grid_game:
            self.dists[self.step] = ((self.loc_goal[0,:2] - self.loc_start[0,:2])/int(self.p_H*CONFIG.RL_softmax_step_size)).abs().int().max()
        else:
            self.dists[self.step] = torch.linalg.norm((self.loc_start-self.loc_goal)[:2])
        self.locs[0 ,:] = self.loc_start
        self.loc_current = self.loc_start.clone().detach()

        # If enabled, create a grid of all possible locations in the image
        # fill this grid with ones for where the agent has been
        if CONFIG.RL_priv_grid_location:
            hg, wg = int(self.im_H / self.p_H) , int(self.im_W / self.p_W)
            self.grid_loc = torch.zeros((CONFIG.RL_max_episode_length, hg, wg)).to(CONFIG.device)
            self.grid_curr_loc = torch.zeros((CONFIG.RL_max_episode_length, hg, wg)).to(CONFIG.device)
            grid_loc_start = int(self.loc_start[0,0].item() / self.p_H) , int(self.loc_start[0,1].item() / self.p_W)

            # Fill in initial position
            self.grid_loc[:,grid_loc_start[0] ,grid_loc_start[1]] = 1
            self.grid_curr_loc[0,grid_loc_start[0] , grid_loc_start[1]] = 1

    def update(self, action, reward, loc, crop, misc=None):
        # Add a new step to the trajectory

        self.actions[self.step,:] = action
        self.weights[self.step] = reward
        self.step += 1
        self.locs[self.step , :] = loc
        self.misc.append(misc)

        if CONFIG.MISC_grid_game:
            self.dists[self.step] = ((self.loc_goal[0,:2] - loc[0,:2])/int(self.p_H*CONFIG.RL_softmax_step_size)).abs().int().max()
        else:
            self.dists[self.step] = torch.linalg.norm((loc-self.loc_goal)[:2])
        self.crops[self.step, :,:,:] = crop
        self.loc_current = loc.detach().clone()

        # fill in the grid location of where the agent is currently
        if CONFIG.RL_priv_grid_location:
            grid_loc = int(loc[0,0].item() / self.p_H) , int(loc[0,1].item() / self.p_W)

            # If agent has walked outside, don't fill in
            if (grid_loc[0] < 0 or grid_loc[0] >= self.grid_size[0]) or (grid_loc[1] < 0 or grid_loc[1] >= self.grid_size[1]):
                pass
            else:
                self.grid_loc[(self.step-1): , grid_loc[0] , grid_loc[1] ] = 1
                self.grid_curr_loc[(self.step -1), grid_loc[0] , grid_loc[1] ] = 1

    def finish(self):
        self.time = (time.perf_counter() - self.time)/self.step
        self.dists = self.dists[:self.step]
        self.crops = self.crops[:self.step + 1,:,:,:]
        self.weights  = self.weights[:self.step]
        self.weights = rewards_to_go(self.weights)
        self.actions  = self.actions[:self.step,:]
        self.locs = self.locs[0:self.step +1, :]

        # Compute final iou
        self.iou = compute_iou(self.loc_goal , self.locs[-1,:])
        self.hasConverged = self.iou >= CONFIG.RL_done_iou
