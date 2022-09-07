

import torch
import torch.nn as nn
import math
import random
import sys
import os
import torch.nn.functional as F


import torchvision.models as models
from torch.autograd import Variable


# During dev
#sys.path.append(os.path.abspath('..'))
#execfile(os.path.join(__file__, '../config.py'))

from config import CONFIG



class ResNet18(nn.Module):

    def __init__(self, channels_in = 3, output_dim = 512 , use_pretrained_net = False):
        super(ResNet18 , self).__init__()

        # Construct a ResNet18 submodule
        self.resnet = models.resnet18( pretrained = use_pretrained_net)

        # Overwrite initial layer to match our specifications
        self.resnet.conv1 = nn.Conv2d( in_channels = channels_in , out_channels = 64 ,
                            kernel_size = 7 , stride = 2 , padding = 3 , bias = False)


        self.resnet.fc = nn.Linear(512 , output_dim )


    def forward(self, x):

        x_res = self.resnet.forward(x)



        return x_res

