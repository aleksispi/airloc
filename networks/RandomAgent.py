import torch
import importlib.util
import torch.nn as nn
import torch.nn.init as init
import math
import random
import sys
import os
import torch.nn.functional as F
from config import CONFIG
from networks.resnets import ResNet18
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import json
from utils.utils import calculate_cnn_output_size
import inspect

# During dev
#sys.path.append(os.path.abspath('..'))
#execfile(os.path.join(__file__, '../config.py'))


class RandomAgent(nn.Module):
    "Implementation of non-rnn agent"
    def __init__(self, unit_size  = 64):
        super(RandomAgent,self).__init__()

        self.softmax = nn.Softmax(dim=1)
        self.AGENT_TYPE = 'RANDOM'
    def _has_visited(self,loc,locs):

        return 2 in (loc == locs).sum(dim=1)


    def forward(self, episode):
        # Assign Uniform probability to all classes
        x_fc = torch.ones([1,8])

        step_sz = int(CONFIG.RL_softmax_step_size*CONFIG.MISC_patch_size[0])
        visited_locs = (episode.locs[:episode.step,:2]/step_sz).int()
        loc = (episode.locs[episode.step,:2]/step_sz).int()

        # Ensure no stepping outside the image
        if loc[0] <= 0:
            x_fc[0,-1] = 0
            x_fc[0,:2] = 0
        if loc[1] <= 0:
            x_fc[0, 5:] = 0
        if loc[0] >= 4:
            x_fc[0, 3:6] = 0
        if loc[1] >= 4:
            x_fc[0, 1:4] = 0

        # Ensure not stepping in same location if possible
        if  episode.step == 0:
            return x_fc, None

        if  self._has_visited(loc - torch.as_tensor([1,0]),visited_locs):
           x_fc[0,0] = 0
        if  self._has_visited(loc - torch.as_tensor([1,-1]),visited_locs):
           x_fc[0,1] = 0
        if  self._has_visited(loc - torch.as_tensor([0,-1]),visited_locs):
           x_fc[0,2] = 0
        if  self._has_visited(loc - torch.as_tensor([-1,-1]),visited_locs):
           x_fc[0,3] = 0
        if  self._has_visited(loc - torch.as_tensor([-1,0]),visited_locs):
           x_fc[0,4] = 0
        if  self._has_visited(loc - torch.as_tensor([-1,1]),visited_locs):
           x_fc[0,5] = 0
        if  self._has_visited(loc - torch.as_tensor([0,1]),visited_locs):
           x_fc[0,6] = 0
        if  self._has_visited(loc - torch.as_tensor([1,1]),visited_locs):
           x_fc[0,7] = 0

        if x_fc.sum() == 0:

            x_fc = torch.ones([1,8])
            # Ensure no stepping outside the image
            # If the vector has been reset
            if loc[0] <= 0:
                x_fc[0,:2] = 0
                x_fc[0,-1] = 0
            if loc[1] <= 0:
                x_fc[0, 5:] = 0
            if loc[0] >= 4:
                x_fc[0, 3:6] = 0
            if loc[1] >= 4:
                x_fc[0, 1:4] = 0

        return x_fc, None
