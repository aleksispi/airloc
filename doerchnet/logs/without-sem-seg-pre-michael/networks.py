
import torch 
import torch.nn as nn 
import torch.nn.functional as F


from config import CONFIG



class AJNet(nn.Module):
    def __init__(self,net_type,mode = 'train', unit_size = 128,num_classes =8 , both_branches = True): 
        super(AJNet,self).__init__()

        # Allow for pretraining network with ground truth segmentation mask
        if CONFIG.RL_priv_use_seg:

            if CONFIG.MISC_dataset == 'masa_filt':
                self.n_chan = 4 
            else:
                raise(Exception("Unkown dataset with segmentation info"))
        else:
            # Regular RGB inputs
            self.n_chan = 3

        self.mode = mode
        self.both_branches = both_branches

        # Choose the embedder for the Doerch net 
        # We start by assuming that we are using fixed weights 


        if net_type == 'Doerch': 
            self.start_enc = Encoder(unit_size , input_channels =  self.n_chan)
            self.goal_enc = self.start_enc 
        elif net_type == 'Doerch2': 
            self.start_enc = Encoder2(unit_size , input_channels = self.n_chan)
            self.goal_enc = self.start_enc 
        else: 
            print(f"Unknown embedder {net_type} for this task")
            exit(1) 

        if both_branches:
            first_common_size = 2 * unit_size
        else:
            first_common_size = unit_size

        self.relu = nn.ReLU()

        self.common_fc_1 = nn.Linear(first_common_size, 2*unit_size) 
        self.common_fc_2 = nn.Linear(2*unit_size,num_classes) 

    def forward(self,start,goal, only_classify = False): 

        if self.both_branches: 
            start_emb = self.start_enc(start) 

        goal_emb = self.goal_enc(goal) 

        if self.both_branches:
            common_emb = torch.cat([start_emb, goal_emb], dim = 1) 
        else:
            common_emb = goal_emb
        common_emb = self.relu(common_emb)
        common_emb = self.common_fc_1(common_emb)
        
        softmax_emb = self.common_fc_2(self.relu(common_emb))
        softmax_emb = F.softmax(softmax_emb, dim = 1 )

        # Sometimes we only want the classification
        if only_classify:
            return softmax_emb
        else:
            return common_emb , softmax_emb


class Encoder(nn.Module): 
    def __init__(self, unit_size = 64, input_channels = 3): 
            
        super(Encoder,self).__init__()
        

        self.modules = []


        # Conv block 1 
        self.modules += [nn.Conv2d(input_channels,16,3,1,1)]
        self.modules += [nn.ReLU()]
        self.modules += [nn.MaxPool2d(2,2)]
        # Conv block 2 
        self.modules += [nn.Conv2d(16,32,3,1,1)]
        self.modules += [nn.ReLU()]
        self.modules += [nn.MaxPool2d(2,2)]
        # Conv block 3 
        self.modules += [nn.Conv2d(32,64,3,1,1)]
        self.modules += [nn.ReLU()]
        self.modules += [nn.MaxPool2d(2,2)]
        # Conv block 4
        self.modules += [nn.Conv2d(64,64,3,1,1)]
        self.modules += [nn.ReLU()]
        self.modules += [nn.MaxPool2d(2,2)]
        # Conv block 5 
        self.modules += [nn.Conv2d(64,64,3,1,1)]
        self.modules += [nn.ReLU()]
        self.modules += [nn.MaxPool2d(2,2)]
        
        # Fc layer to map to correct unit size 
        self.modules += [nn.Flatten()]
        self.modules += [nn.Linear(64,unit_size)]

        self.net = nn.Sequential(*self.modules)


    def forward(self,patch):
        return self.net(patch) 



class Encoder2(nn.Module): 
    def __init__(self, unit_size = 64, input_channels = 3): 
            
        super(Encoder2,self).__init__()
        

        self.modules = []


        # Conv block 1 
        self.modules += [nn.Conv2d(input_channels,8,3,1,padding = 'same')]
        self.modules += [nn.ReLU()]
        self.modules += [nn.MaxPool2d(2,2)]
        # Conv block 2 
        self.modules += [nn.Conv2d(8,16,5,1,padding = 'same')]
        self.modules += [nn.ReLU()]
        self.modules += [nn.MaxPool2d(2,2)]
        # Conv block 3 
        self.modules += [nn.Conv2d(16,32,5,1,padding = 'same')]
        self.modules += [nn.ReLU()]
        self.modules += [nn.MaxPool2d(2,2)]
        # Conv block 4
        self.modules += [nn.Conv2d(32,64,5,1,padding = 'same')]
        self.modules += [nn.ReLU()]
        self.modules += [nn.MaxPool2d(2,2)]
        # Out comes shape 3x3x64 
        
        # Fc layer to map to correct unit size 
        self.modules += [nn.Flatten()]
        self.modules += [nn.Linear(9*64,unit_size)]

        self.net = nn.Sequential(*self.modules)


    def forward(self,patch):
        return self.net(patch) 
        





