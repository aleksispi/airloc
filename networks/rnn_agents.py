
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import importlib.util
import time
import json
import math
from utils.utils import calculate_cnn_output_size, cosine_sim_heatmap
from utils.agent_utils import visualize_cnn_filter, get_outside
import matplotlib
import matplotlib.pyplot as plt


from config import CONFIG


class LSTMAgent(nn.Module):
    """ AiRLoc agent """
    def __init__(self, unit_size = 256):
        super(LSTMAgent, self).__init__()

        # Output size of individual units
        self.unit_size = unit_size

        # If ResNet8 is selected as embedder this options sets wether to freeze the weights or not
        self.freeze_patch_embedder = CONFIG.RL_freeze_patch_embedder

        # Determine the number of segmentation info channels to use
        if CONFIG.RL_priv_use_seg or CONFIG.RL_predict_seg_mask:
            # The number of segmentation channels available depend on the dataset
            if CONFIG.MISC_dataset == 'dubai':
                self.seg_chan = 3
            elif CONFIG.MISC_dataset == 'masa':
                self.seg_chan = 1
            elif CONFIG.MISC_dataset == 'masa_filt':
                self.seg_chan = 1
            elif CONFIG.MISC_dataset == 'masa_seven':
                self.seg_chan = 1
            elif CONFIG.MISC_dataset == 'dubai_seven':
                self.seg_chan = 1
            else:
                raise(Exception("Use segmentation information was selected but the dataset has no segmentation info."))
        else:
            self.seg_chan = 0

        # Define the embedder
        if CONFIG.RL_patch_embedder == 'Segmenter':
            self.patch_emb = self._make_segmenter_patch_embedder()
        elif 'Doerch' in CONFIG.RL_patch_embedder:
            self.patch_emb = self._make_doerch_patch_embedder()
        elif 'ShareNet' in CONFIG.RL_patch_embedder:  # DEFAULT
            self.patch_emb = self._make_doerch_patch_embedder()
        elif CONFIG.RL_patch_embedder is None:
            self.patch_emb = None
            self.x_curr_emb = torch.zeros(1,1,256).to(CONFIG.device)
        else:
            raise(Exception("Unknown patch embedder selected in LSTMAgent:\t%s" % CONFIG.RL_patch_embedder))

        # If enabled will also send a flattened grid location of the agent to the lstm
        if CONFIG.RL_priv_grid_location:
            im_H, im_W , patch_H , patch_W = *CONFIG.MISC_im_size , *CONFIG.MISC_patch_size

            # Times two because two grids for current and previous positions
            self.unit_size += 2 * int(im_H / patch_H) * int(im_W / patch_W)

        if self.freeze_patch_embedder:
            for param in self.patch_emb.parameters():
                param.requires_grad = False

        # Define the RNN
        self.rnn = nn.LSTM(input_size = self.unit_size + 8 * CONFIG.EE_temporal, hidden_size = self.unit_size, num_layers = 1,
                           bias = True, batch_first = True, dropout = 0, bidirectional = False)

        # If enabled, load a U-net and use it to predict the building segmentation mask
        if CONFIG.RL_predict_seg_mask:

            # First check if segmentation info is enabled, it shouldn't
            if CONFIG.RL_priv_use_seg:
                raise(Exception("Prediction of segmentation mask and ground truth segmentation mask cannot be enabled at the same time."))

            self.seg_net = self._make_seg_net()

        # TODO: Overwrites the line common_fc_input += self.unit_size above
        common_fc_input = self.unit_size

        # Define the final fully connected layer
        self.fc = nn.Linear(common_fc_input , 8) # Directions from current
        self.softmax = nn.Softmax( dim = 1 )

        # Reset the agent to initialize the hidden states
        self.reset()

        # Set agent type to be able to rseset hidden states
        self.AGENT_TYPE = 'RNN'

        # If enabled the patch embeddings will be embedded with their absolute position in the image
        # The below happens with default config (CONFIG.RL_LSTM_pos_emb = 'half')
        if CONFIG.RL_LSTM_pos_emb:
            self._preallocate_pos_enc(max_len = 29)

    def _has_visited(self,loc,locs):
        return 2 in (loc == locs).sum(dim=1)

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

        # Load the weights
        if not os.path.exists(os.path.join(CONFIG.RL_pretrained_segmentation_net,"final_unet")):
            raise(Exception("No U-net found in:\t%s" % CONFIG.RL_pretrained_segmentation_net))
        seg_net.load_state_dict(torch.load(os.path.join(CONFIG.RL_pretrained_segmentation_net, "final_unet"), map_location = torch.device('cpu')))
        for param in seg_net.parameters():
            param.requires_grad = False

        return seg_net

    def _preallocate_pos_enc(self,  dropout: float = 0.1, max_len: int = 16):
        d_model = self.unit_size

        if CONFIG.MISC_game_size == 7:
            position = torch.arange(max_len).unsqueeze(1) * (5 - 2) / (CONFIG.MISC_game_size - 2)
        else:
            position = torch.arange(max_len).unsqueeze(1)

        div_term_factor = -0.01

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        div_term = torch.exp(torch.arange(0, d_model, 2) * div_term_factor)
        self.pe = torch.zeros( 1,max_len, d_model).to(CONFIG.device)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

        # Also construct two separate embedding with half dmodel
        div_term_half = torch.exp(torch.arange(0, d_model // 2, 2)  * div_term_factor)
        self.pe_half = torch.zeros(1 , max_len , d_model // 2).to(CONFIG.device)
        self.pe_half[0, :, 0::2] = torch.sin(position * div_term_half)
        self.pe_half[0, :, 1::2] = torch.cos(position * div_term_half)
        if False:  # Plot of the similarity scores
            grid_size = int( CONFIG.MISC_im_size[0] / CONFIG.MISC_patch_size[0] + 1 )
            # cosine_sim_pos = [(0,0) , (1,2) , (4,2),(0,2) , (3,1),(0,1),(1,0),(4,4),(4,3),(5,1),(0,5),(2,5)]
            cosine_sim_pos = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6)]
            cosine_sim_pos = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4),(4,0),(4,1),(4,2),(4,3),(4,4)]
            os.makedirs(os.path.join(CONFIG.STATS_log_dir, "positional_embeddings"),exist_ok = True)
            for pos in cosine_sim_pos:
                cosine_sim_heatmap(self.pe_half , pos = pos , grid_size = grid_size )
            print("DONE")
            time.sleep(999)

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

    def _make_segmenter_patch_embedder(self):
        """
        Retrieves the encoder part of the segmentation,
        this requires only rgb input and the latent space from a segmentation
        task
        """

        if not os.path.exists(CONFIG.RL_pretrained_segmentation_net):
            print("The segmentation encoder does not exist, check that the path is correct")
            exit(1)
        # Gets the network file from the log file and load it as a module
        networks_file_path = os.path.join(CONFIG.RL_pretrained_segmentation_net,"u_net.py")
        spec = importlib.util.spec_from_file_location("u_net", networks_file_path)
        segmentation_networks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(segmentation_networks)

        # Load the json file generated during the segmentation training
        with open(os.path.join(CONFIG.RL_pretrained_segmentation_net, "info.json"),'r') as io:
            info = json.load(io)

        # Determine which network should be loaded for the rl agent
        network_type = info['NetType']
        if network_type == 'UNet':
            feature_net = segmentation_networks.UNet(3,2)
        else:
            raise(Exception(f"Ukn segmentation network {network_type}"))

        latentSpaceSize = info['LatentSpaceSize']

        if CONFIG.RL_priv_pretrained:
            feature_net.load_state_dict(torch.load(os.path.join(CONFIG.RL_pretrained_segmentation_net, "final_unet")))

        if self.freeze_patch_embedder:
            for param in feature_net.parameters():
                param.requires_grad = False

        # Unroll the network and allow it to get the embedding
        modules = list(feature_net.children())
        middle = int(len(modules)/2)
        modules = modules[:middle]
        # Assumes 64,3,3
        # modules.append(nn.Conv2d(64,64,3))

        modules.append(nn.Flatten(start_dim = 1))
        modules.append(nn.Linear(9*64, 128))
        modules.append(nn.ReLU())
        embedder = nn.Sequential(*modules)

        return embedder

    def _make_doerch_patch_embedder(self):

        if not os.path.exists(CONFIG.RL_pretrained_doerch_net):
            print("The pretrained doerch net does not exist check the file path")
            exit(1)

        # Load the json file generated during pretraining
        with open(os.path.join(CONFIG.RL_pretrained_doerch_net, "info.json"),'r') as io:
            info = json.load(io)

        # Determine which network should be loaded
        network_type = info['NetType']
        if network_type.startswith("Doerch"):
            networks_file_path = os.path.join(CONFIG.RL_pretrained_doerch_net, "networks.py")
            spec = importlib.util.spec_from_file_location("networks",networks_file_path)
            doerch_networks = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(doerch_networks)
            feature_net = doerch_networks.AJNet(network_type)
        elif network_type.startswith("ShareNet"):
            networks_file_path = os.path.join(CONFIG.RL_pretrained_doerch_net, "share_net.py")
            spec = importlib.util.spec_from_file_location("share_net",networks_file_path)
            doerch_networks = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(doerch_networks)
            feature_net = doerch_networks.ShareNet()
        else:
            raise(Exception("Unknown encoder network "))

        latentSpaceSize = info['LatentSpaceSize']

        if CONFIG.RL_priv_pretrained:
            feature_net.load_state_dict(torch.load(os.path.join(CONFIG.RL_pretrained_doerch_net,"doerch_embedder"), map_location=torch.device('cpu')))

        if self.freeze_patch_embedder:
            for param in feature_net.parameters():
                param.requires_grad = False

        return feature_net

    def reset(self):
        """ Resets the hidden states of the RNN network."""

        # The size of the hidden states depend on the LSTM network
        D = 2 if self.rnn.bidirectional else 1

        self._hidden = torch.zeros(D * self.rnn.num_layers, 1, self.unit_size).to(CONFIG.device)
        self._cell = torch.zeros(D * self.rnn.num_layers , 1,self.unit_size).to(CONFIG.device)

    def forward(self, episode):

        # For now only one mode, use latest available crop and step once
        step = episode.step

        x_curr = episode.crops[step , : , :,:].unsqueeze(0)
        x_goal = episode.crop_goal

        # If enabled predict segmentation mask and concat it to the patch input
        if CONFIG.RL_predict_seg_mask:
            seg_net_input = torch.cat((x_curr, x_goal) , dim = 0)
            seg_mask = self.seg_net(seg_net_input)

            # Output is one channel per class, i.e, need to do argmax to get it to one channel for two class problems
            # TODO add exception when not using two class seg masks

            seg_mask = torch.argmax( seg_mask , dim = 1, keepdim = True)

            x_curr = torch.cat((x_curr, seg_mask[0,:].unsqueeze(0)) , dim = 1)
            x_goal = torch.cat((x_goal, seg_mask[1,:].unsqueeze(0)) , dim = 1)

        # First calculate embeddings of current crop and goal crop
        if CONFIG.RL_patch_embedder is None:
            x_curr_emb = self.x_curr_emb
            x_softmax_emb = None
        elif "Doerch" in CONFIG.RL_patch_embedder or "ShareNet" in CONFIG.RL_patch_embedder:
            # DEFAULT BRANCH
            x_curr_emb , x_softmax_emb = self.patch_emb(x_curr, x_goal)
            x_curr_emb = x_curr_emb.unsqueeze(0)
        else:
            x_curr_emb = self.patch_emb(x_curr).unsqueeze(0)
            x_goal_emb = self.patch_emb(x_goal).unsqueeze(0)
            x_curr_emb = torch.cat((x_curr_emb,x_goal_emb), dim=2)
            x_softmax_emb = None

        # embedd position into result from patch_emb
        if CONFIG.RL_LSTM_pos_emb:
            x_curr_emb = self.embedd_position(x_curr_emb , episode.locs[step, :][None,:])

        # If enabled send a flattened version of the grid location of the agent
        if CONFIG.RL_priv_grid_location:
            x_curr_emb = torch.cat((x_curr_emb ,
                torch.flatten(episode.grid_loc[step,:]).unsqueeze(0).unsqueeze(0),
                torch.flatten(episode.grid_curr_loc[step,:]).unsqueeze(0).unsqueeze(0)) , dim = 2)

        # Append the patch embedders softmax embedding for further guidance
        if CONFIG.EE_temporal:
            x_curr_emb = torch.cat((x_curr_emb, x_softmax_emb.unsqueeze(0)), dim=2)

        # Run embedding of current crop through LSTM network
        x_curr_lstm , (self._hidden , self._cell) = self.rnn(x_curr_emb, (self._hidden , self._cell))

        # Squeeze away the sequence dimension since it will always be one
        x_curr_lstm = x_curr_lstm.squeeze(1)

        # If the goal patch is not in the lstm append it after the lstm
        x_fc = x_curr_lstm
        x_fc = self.fc(x_fc)

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
                return self.softmax(x_fc +1000000* x_o), None

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

            return self.softmax(x_fc+1000000*x_o), None

        if not CONFIG.RL_agent_allowed_outside:
            outside = get_outside(episode).to(CONFIG.device)
            x_fc = x_fc -10000* outside

        if CONFIG.EE_residual:
            x_fc += x_softmax_emb

        x_fc = self.softmax(x_fc)

        return x_fc, x_softmax_emb
