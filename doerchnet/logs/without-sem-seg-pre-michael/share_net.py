

import torch
import torch.nn as nn

from config import CONFIG


class ShareNet(nn.Module):

    def __init__(self , num_out_classes = 8):
        super(ShareNet, self).__init__()


        # Check wether to include segmentation channel 
        if CONFIG.RL_priv_use_seg or CONFIG.RL_predict_seg_mask:
            self.seg_chan = 1
        else:
            self.seg_chan = 0

        self.n_chan = 3 + self.seg_chan

        self.latent_space_dim = 256

        # Start assembling the network.
       
        # For now non-siamese setup
        
        self.start_double_conv1 = DoubleConvBlock(self.n_chan)
        self.goal_double_conv1 = DoubleConvBlock(self.n_chan)

        intermediate_n_chans = self.start_double_conv1.out_chan 

        self.start_double_conv2 = DoubleConvBlock( intermediate_n_chans * 2)
        self.goal_double_conv2 = DoubleConvBlock( intermediate_n_chans * 2)
      
        self.flat = nn.Flatten(start_dim = 1)
       
        self.common_layer_dim = 2 * self.start_double_conv2.out_chan * ((CONFIG.MISC_patch_size[0] // ( 2**4) )**2)

        # Common linear layer
        self.fc_int = nn.Linear( self.common_layer_dim , self.latent_space_dim)
        self.fc_out = nn.Linear( self.latent_space_dim , num_out_classes)

        self.act = nn.ReLU()

    def forward(self, start , goal):

        start_1 = self.start_double_conv1(start)
        goal_1 = self.goal_double_conv1(goal)

        start_goal = torch.cat((start_1, goal_1 ) , dim = 1)
        goal_start = torch.cat((goal_1, start_1 ) , dim = 1)

        start_2 = self.start_double_conv2(start_goal)
        goal_2 = self.goal_double_conv2(goal_start)

        emb = torch.cat((start_2, goal_2), dim = 1)
        emb = self.flat(emb)

        # First linear layer to latent space
        emb = self.fc_int(emb)
        emb = self.act(emb) 

        out = self.fc_out(emb)

        return emb, out


class DoubleConvBlock(nn.Module):

    def __init__(self, in_channels, kernel_size = 3):
        super(DoubleConvBlock, self).__init__()

        self.block1 = ConvBlock(in_channels, 2 * in_channels , kernel_size)
        self.block2 = ConvBlock(2 * in_channels, 3 * in_channels , kernel_size)
      
        self.out_chan = 3 * in_channels

    def forward(self, x): 
        x = self.block1(x)
        x = self.block2(x)

        return x


class ConvBlock(nn.Module):

    def __init__(self , in_channels, out_channels , kernel_size = 3):
        super(ConvBlock, self).__init__()
        
        
        self.conv = nn.Conv2d(in_channels , out_channels , kernel_size, padding = 'same')
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)


    def forward(self, x):

        x = self.conv(x)
        x = self.act(x)
        x = self.pool(x)

        return x

    

