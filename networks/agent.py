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
from utils.utils import calculate_cnn_output_size, cosine_sim_heatmap
import inspect
import time

# During dev
#sys.path.append(os.path.abspath('..'))
#execfile(os.path.join(__file__, '../config.py'))


class Agent(nn.Module):
    "Implementation of non-rnn agent"
    def __init__(self, unit_size  = 256):
        super(Agent,self).__init__()

        # Define the
        self.n_chan = 3
        if CONFIG.RL_priv_use_seg:
            if CONFIG.MISC_dataset in ['dubai']:
                self.n_chan = 6
            elif CONFIG.MISC_dataset in ['masa_filt']:
                self.n_chan = 4
            else:
                raise(Exception("Define which type of segmentation this dataset has"))

        # Size of the embeddings before concatinating
        self.unit_size = unit_size

        # Input size of the first concatenated layer
        common_fc_inpt = 2 * unit_size

        # Define the embedder
        if 'Doerch' in CONFIG.RL_patch_embedder:
            self.patch_emb = self._make_doerch_patch_embedder()
        elif 'ShareNet' in CONFIG.RL_patch_embedder:  # DEFAULT
            self.patch_emb = self._make_doerch_patch_embedder()
        else:
            raise(Exception("Unknown Embedder:\t%s" % CONFIG.RL_patch_embedder) )

        # Define the final fully connected layer
        self.fc = nn.Linear(256 , 8) # Directions from current
        self.softmax = nn.Softmax( dim = 1 )

        self.AGENT_TYPE = 'REGULAR'

        if CONFIG.RL_LSTM_pos_emb:
            self._preallocate_pos_enc( max_len = 29)

        # If enabled load a U-net and use it to predict the building segmentation mask
        if CONFIG.RL_predict_seg_mask:

            # First check if segmentation info is enabled, it shouldn't
            if CONFIG.RL_priv_use_seg:
                raise(Exception("Prediction of segmentation mask and ground truth segmentation mask cannot be enabled at the same time."))

            self.seg_net = self._make_seg_net()

    def _make_seg_net(self):

        # Check that the specified segmentaion log exists
        if not os.path.exists(CONFIG.RL_pretrained_segmentation_net):
            raise(Exception("Segmentation log does not exist:\t%s" % (CONFIG.RL_pretrained_segmentation_net)))


        networks_file_path = os.path.join(CONFIG.RL_pretrained_segmentation_net,"u_net.py")
        spec = importlib.util.spec_from_file_location("u_net", networks_file_path)
        segmentation_networks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(segmentation_networks)

        with open(os.path.join(CONFIG.RL_pretrained_segmentation_net, "info.json"),'r') as io:
            info = json.load(io)

        # Determine which network should be loaded for the rl agent
        network_type = info['NetType']
        if network_type == 'UNet':
            seg_net = segmentation_networks.UNet(3,2)
        else:
            raise(Exception(f"Uknkown segmentation network {network_type}"))


        # Load the weihgts
        if not os.path.exists(os.path.join(CONFIG.RL_pretrained_segmentation_net,"final_unet")):
            raise(Exception("No U-net found in:\t%s" % CONFIG.RL_pretrained_segmentation_net))

        seg_net.load_state_dict(torch.load(os.path.join(CONFIG.RL_pretrained_segmentation_net, "final_unet"), map_location = torch.device('cpu')))

        if True:
            for param in seg_net.parameters():
                param.requires_grad = False
        else:
            print("WARNING: No freeze selected for seg net. Training might not be possible since argmax is used")


        return seg_net

    def _has_visited(self,loc,locs):
        return 2 in (loc == locs).sum(dim=1)

    def _preallocate_pos_enc(self,  dropout: float = 0.1, max_len: int = 16):
        d_model = self.unit_size

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        div_term = torch.exp(torch.arange(0, d_model, 2)  * ( - 0.01 ))
        self.pe = torch.zeros( 1,max_len, d_model).to(CONFIG.device)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

        # Also construct two separate embedding with half dmodel
        div_term_half = torch.exp(torch.arange(0, d_model // 2, 2)  * ( - 0.01 ))
        self.pe_half = torch.zeros(1 , max_len , d_model // 2).to(CONFIG.device)
        self.pe_half[0, :, 0::2] = torch.sin(position * div_term_half)
        self.pe_half[0, :, 1::2] = torch.cos(position * div_term_half)
        if False:
            grid_size = int( CONFIG.MISC_im_size[0] / CONFIG.MISC_patch_size[0] + 1 )
            cosine_sim_pos = [(0,0) , (1,2) , (4,2),(0,2) , (3,1),(0,1),(1,0),(4,4),(4,3),(5,1),(0,5),(2,5)]
            os.makedirs(os.path.join(CONFIG.STATS_log_dir, "positional_embeddings"),exist_ok=True)

            for pos in cosine_sim_pos:
                cosine_sim_heatmap(self.pe_half , pos = pos , grid_size = grid_size )

    def embedd_position(self, x, locs , goal_emb_included = False):
        """ Embedds position into the sequence. """

        # First get position in image (x,y) normalized to (0,1)
        xy_pos = locs[:,0:2] / (torch.tensor(CONFIG.MISC_im_size))
        # Append x and y position (-1,-1) for goal crop(which is the last)
        if goal_emb_included:
            xy_pos = torch.cat((xy_pos, torch.tensor([[-1,-1]])) , dim = 0)

        # Get position in grid
        xy_grid = (locs[:,0:2] / (torch.tensor(CONFIG.MISC_patch_size))).long()
        # We want the goal crop to get f(0) add that to grid
        if goal_emb_included:
            xy_grid = torch.cat(( xy_grid , torch.tensor([[0,0]])) , dim = 0 )

        # Half the positional embedding is for x other half for y
        pos_embedding = torch.flatten(self.pe_half[0,xy_grid] , start_dim = 1, end_dim = 2)
        x_pos_emb = x + pos_embedding

        return x_pos_emb

    def _construct_patch_embedder(self):
        """ Constructs the embedder network. A series of cnn layers. """

        # [in_chan, out_chan, kernel, stride, padding]
        max_pool = [3,2]
        layers = []
        modules = []


        layers.append([self.n_chan, 16, 3, 1, 0])
        layers.append([16, 32, 3, 1, 0])

        # Construct layers
        for layer in layers:
            modules.append(nn.Conv2d(layer[0],layer[1],layer[2],layer[3],layer[4]))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool2d(max_pool[0],max_pool[1]))

        # Calculate output size from CNN layers
        out_size = calculate_cnn_output_size(layers, CONFIG.MISC_patch_size, max_pool)
        linear_input_size = int(out_size[0] * out_size[1] * out_size[2])

        # Flatten and add final linear layer
        modules.append(nn.Flatten(start_dim = 1))
        modules.append(nn.Linear(linear_input_size , self.unit_size))

        embedder = nn.Sequential(*modules)

        return embedder

    def _make_doerch_patch_embedder(self):

        if not os.path.exists(CONFIG.RL_pretrained_doerch_net):
            print("The pretrained doerch net does not exist check the file path")
            exit(1)

        # Load the json file generated during pretraining
        with open(os.path.join(CONFIG.RL_pretrained_doerch_net, "info.json"),'r') as io:
            info = json.load(io)

        dim = 8

        # Determine which network should be loaded
        network_type = info['NetType']
        if network_type.startswith("Doerch"):
            networks_file_path = os.path.join(CONFIG.RL_pretrained_doerch_net, "networks.py")
            spec = importlib.util.spec_from_file_location("networks",networks_file_path)
            doerch_networks = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(doerch_networks)
            feature_net = doerch_networks.AJNet(network_type, num_classes = dim)

        elif network_type.startswith("ShareNet"):
            networks_file_path = os.path.join(CONFIG.RL_pretrained_doerch_net, "share_net.py")
            spec = importlib.util.spec_from_file_location("share_net",networks_file_path)
            doerch_networks = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(doerch_networks)

            feature_net = doerch_networks.ShareNet(num_out_classes = dim)
        else:
            raise(Exception("Unknown encoder network "))

        latentSpaceSize = info['LatentSpaceSize']

        if CONFIG.RL_priv_pretrained:
            feature_net.load_state_dict(torch.load(os.path.join(CONFIG.RL_pretrained_doerch_net,"doerch_embedder"), map_location=torch.device('cpu')))

        return feature_net

    def forward(self, episode):

        step = episode.step

        # First calculate embeddings of current crop and goal crop
        x_curr = episode.crops[step , : , :,:].unsqueeze(0)
        x_goal = episode.crop_goal

        if CONFIG.RL_predict_seg_mask:
            seg_net_input = torch.cat((x_curr, x_goal) , dim = 0)
            seg_mask = self.seg_net(seg_net_input)

            # Output is one channel per class, i.e, need to do argmax to get it to one channel for two class problems
            # TODO add exception when not using two class seg masks

            seg_mask = torch.argmax( seg_mask , dim = 1, keepdim = True)

            x_curr = torch.cat((x_curr, seg_mask[0,:].unsqueeze(0)) , dim = 1)
            x_goal = torch.cat((x_goal, seg_mask[1,:].unsqueeze(0)) , dim = 1)

        if 'Doerch' in CONFIG.RL_patch_embedder or 'ShareNet' in CONFIG.RL_patch_embedder:
            output, softmax = self.patch_emb(x_curr,x_goal)
        if CONFIG.RL_LSTM_pos_emb:
            x_curr_emb = self.embedd_position(output , episode.locs[step, :][None,:])
            x_fc = self.fc(x_curr_emb)
            return self.softmax(x_fc),softmax

        else:
            if CONFIG.MISC_priv:
                x_o = torch.zeros([1,8]).to(CONFIG.device)

                step_sz = int(CONFIG.RL_softmax_step_size*CONFIG.MISC_patch_size[0])
                visited_locs = (episode.locs[:episode.step,:2]/step_sz).int()
                loc = (episode.locs[episode.step,:2]/step_sz).int()

                # Ensure no stepping outside the image
                if loc[0] <= 0:
                    x_o[0,-1] = -1
                    x_o[0,:2] = -1
                if loc[1] <= 0:
                    x_o[0, 5:] = -1
                if loc[0] >= CONFIG.MISC_game_size - 1:
                    x_o[0, 3:6] = -1
                if loc[1] >= CONFIG.MISC_game_size - 1:
                    x_o[0, 1:4] = -1

                # Ensure not stepping in same location if possible
                if  episode.step == 0:
                    return self.softmax(softmax +1000000* x_o), None

                if  self._has_visited(loc - torch.as_tensor([1,0]),visited_locs):
                   x_o[0,0] = -1
                if  self._has_visited(loc - torch.as_tensor([1,-1]),visited_locs):
                   x_o[0,1] = -1
                if  self._has_visited(loc - torch.as_tensor([0,-1]),visited_locs):
                   x_o[0,2] = -1
                if  self._has_visited(loc - torch.as_tensor([-1,-1]),visited_locs):
                   x_o[0,3] = -1
                if  self._has_visited(loc - torch.as_tensor([-1,0]),visited_locs):
                   x_o[0,4] = -1
                if  self._has_visited(loc - torch.as_tensor([-1,1]),visited_locs):
                   x_o[0,5] = -1
                if  self._has_visited(loc - torch.as_tensor([0,1]),visited_locs):
                   x_o[0,6] = -1
                if  self._has_visited(loc - torch.as_tensor([1,1]),visited_locs):
                   x_o[0,7] = -1
                if x_o.sum() == -8:
                    x_o = torch.zeros([1,8]).to(CONFIG.device)
                    # Ensure no stepping outside the image
                    # If the vector has been reset
                    if loc[0] <= 0:
                        x_o[0,:2] = -1
                        x_o[0,-1] = -1
                    if loc[1] <= 0:
                        x_o[0, 5:] = -1
                    if loc[0] >= CONFIG.MISC_game_size - 1:
                        x_o[0, 3:6] = -1
                    if loc[1] >= CONFIG.MISC_game_size - 1:
                        x_o[0, 1:4] = -1

                return self.softmax(softmax+1000000*x_o), None

            return self.softmax(softmax),output
        x_goal_emb = self.patch_emb(x_goal)
        x_curr_emb = self.patch_emb(x_curr)
        # Concat all results
        x_fc = torch.cat((x_curr_emb ,x_goal_emb) , dim = 1)
        x_fc = self.fc(x_fc)
        x_fc = self.softmax(x_fc)

        return x_fc
