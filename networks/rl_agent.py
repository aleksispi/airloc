import torch
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
import inspect

# During dev
#sys.path.append(os.path.abspath('..'))
#execfile(os.path.join(__file__, '../config.py'))


def _weights_init(m):
    classname = m.__class__.__name__

    if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
        init.kaiming_normal_(m.weight)



class LambdaLayer(nn.Module):
    def __init__(self,lambd):
        super(LambdaLayer,self).__init__()
        self.lambd = lambd

    def forward(self,x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,in_filters,filters,stride=1,option='A'):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_filters,filters,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters,filters,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

        self.shortcut = nn.Sequential()
        if stride !=1 or in_filters!=filters:
            if option=='A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:,:,::2,::2],(0,0,0,0,filters//4,filters//4),"constant",0))

            elif option =='B':
                self.shortcut=nn.Sequential(
                    nn.Conv2d(in_filters,self.expansion*filters,kernel_size=1,stride=stride,bias=False),
                    nn.BatchNorm2d(self.expansion*filters)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=2):
        super(CustomResNet,self).__init__()
        self.in_filters = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1,
                               bias = True)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride = 2)

        # Make pool layer
        noise = torch.randn(1, 3, CONFIG.MISC_patch_size[0] , CONFIG.MISC_patch_size[1])

        # Run noise through layers to find out size for pool
        out = F.relu(self.bn1(self.conv1(noise)))
        out = self.layer1(out)
        out = self.layer2(out)
        shape = self.layer3(out).shape
        self.pool_layer = nn.AvgPool2d(shape[3])

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, filters, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_filters , filters, stride))
            self.in_filters = filters * block.expansion


        return nn.Sequential(*layers)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool_layer(out)
        out  = torch.flatten(out,1)

        # Add linear layer for 10 class prediction
        out = self.linear(out)

        return out
