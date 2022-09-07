
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

 
class DeterministicAgent(nn.Module): 
    "Implementation of non-rnn agent" 
    def __init__(self, mode='spiral'): 
        super(DeterministicAgent,self).__init__()
        self.mode = mode
        self.softmax = nn.Softmax(dim=1) 
        self.AGENT_TYPE = 'DETERMINISTIC'
        self.dir = -2 
        self.locked_in = False

    def _has_visited(self,loc,locs):
        if (loc<0).any() or (loc>=CONFIG.MISC_im_size[0]).any():
            return True 
        return 2 in (loc == locs).sum(dim=1)
        
    def _get_move(self,action):
        if action < 0: action += 8
        if action > 7: action -= 8
        if action == 0: 
            return torch.as_tensor([-48,0])
        if action == 1: 
            return torch.as_tensor([-48,48])
        if action == 2: 
            return torch.as_tensor([0,48])
        if action == 3: 
            return torch.as_tensor([48,48])
        if action == 4: 
            return torch.as_tensor([48,0])
        if action == 5: 
            return torch.as_tensor([48,-48])
        if action == 6: 
            return torch.as_tensor([0,-48])
        if action == 7: 
            return torch.as_tensor([-48,-48])

    def _forward_spiral(self,episode):
        # Assign zero probability to all classes
        x_fc = torch.zeros([1,8])
       
        # Get the current an all visited locations from the episode storage 
        visited_locs = episode.locs[:episode.step,:2]
        loc = episode.locs[episode.step,:2]
       
        # Ensure not stepping in same location if possible 

        if  episode.step == 0:
            for action in [0,2,4,6]:
                next_loc = loc + self._get_move(action)
                if not self._has_visited(next_loc,torch.as_tensor([[-1,-1]])):
                    if  self._has_visited(next_loc + self._get_move(action+self.dir),torch.as_tensor([[-1,-1]])):
                        if self.dir == -2: self.dir=2
                        else: self.dir=-2
                    x_fc[0,action] = 1
                    return x_fc

        last_action = episode.actions[episode.step - 1,:].argmax()
       
        if self.locked_in and self._has_visited(loc + self._get_move(last_action),visited_locs):
            x_fc[0,last_action] = 1
            return x_fc
        
        self.locked_in = False 
        if not self._has_visited(loc + self._get_move(last_action+self.dir),visited_locs):
            a = last_action + self.dir 
            if a>7: a-=8
            x_fc[0,a] = 1
            return x_fc
        elif not self._has_visited(loc + self._get_move(last_action),visited_locs): 
            x_fc[0,last_action] = 1
            return x_fc
        elif not self._has_visited(loc + self._get_move(last_action-self.dir),visited_locs): 
            a = last_action - self.dir 
            if a>7: a-=8
            x_fc[0,a] = 1
            if self.dir == -2: self.dir=2
            else: self.dir=-2
            return x_fc
        else:
            x_fc[0,last_action-4] = 1
            self.locked_in = True
            if self.dir == -2: self.dir=2
            else: self.dir=-2
            return x_fc



    def _forward_rows(self, episode):
        pass 
    def forward(self, episode):
        if self.mode == 'spiral':
            return self._forward_spiral(episode),None
        elif self.mode == 'rows': 
            return self._forward_rows(episode),None
        else:
            raise(Exception(f"Unknown mode: \t {self.mode} selected for the DeterministicAgent"))
























