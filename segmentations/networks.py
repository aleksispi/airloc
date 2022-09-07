

import random

from config import CONFIG

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from utils.utils import calculate_cnn_output_size


class BakeNet(nn.Module):

    def __init__(self, n_out_chan = 2):
        super(BakeNet ,self).__init__()
        
        # network output is always in One Hot Encoding format

        # Very simple demonstration network
        modules = [] 
        modules.append( nn.Conv2d(3 , 16 , 3 , 1 , padding = 1))
        modules.append( nn.ReLU())
        modules.append( nn.BatchNorm2d( 16 ))
        modules.append( nn.Conv2d( 16 , 32 , 3 , 1 , padding = 1))
        modules.append( nn.ReLU()) 
        modules.append( nn.BatchNorm2d( 32))
        modules.append( nn.Conv2d( 32 , 16 , 3 , 1 , padding = 1))
        modules.append( nn.ReLU())
        modules.append( nn.BatchNorm2d(16))
        modules.append( nn.Conv2d( 16 , n_out_chan , 3, 1 , padding = 1))
       
        modules.append(nn.Softmax(dim  = 1))

        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)

    def get_latent_space_size(self):
        return 0 # Not clear latent space

