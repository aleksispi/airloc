


from datetime import datetime
import random
import time
import numpy as np
import json
from shutil import copyfile
import os
import sys
# matplotlib is used for debugging image inputs to networks
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
from copy import deepcopy
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import sys
import math
from config import CONFIG
from utils.utils import load_normalize_data, visualize_trajectory, get_random_crops,\
                        get_deterministic_crops , get_crop_distance , compute_iou,\
                        check_outside, _setup_fixed_games, visualize_batch


# Import agent utils
from utils.agent_utils import get_reward , compute_loss , \
                           take_step ,  check_if_done , \
                           rewards_to_go , get_policy ,\
                           get_action , run_eval_trajectory

from doerchnet.utils import sample_doerch_crops

from networks.rl_agent import AJRLAgent, PatchResNetAgent

from networks.agent import Agent

from networks.RandomAgent import RandomAgent

from networks.pretrained_resnet_agent import PretrainedResNetAgent

from networks.rnn_agents import LSTMAgent

from networks.deterministic_agent import DeterministicAgent

from utils.stat_collector import StatCollector

from torch.utils.tensorboard import SummaryWriter

from utils.training_utils import BatchStorage , EpisodeStorage

#Create a directory to save the info
if not os.path.exists(CONFIG.STATS_log_dir_base):
    os.makedirs(CONFIG.STATS_log_dir_base)
os.makedirs(CONFIG.STATS_log_dir, exist_ok = False)

# Save visualizations in separate directory
vis_dir = os.path.join(CONFIG.STATS_log_dir, "visualizations")
os.makedirs(vis_dir)

scripts_dir = os.path.join(CONFIG.STATS_log_dir, "saved_scripts")
os.makedirs(scripts_dir)

# Save this training file
copyfile("training/train_agent.py" , os.path.join(scripts_dir, "train_agent.py"))

# Save config file
copyfile("config.py" , os.path.join(scripts_dir , "config.py"))

# Save network
copyfile("networks/early_rl_agents.py" , os.path.join(scripts_dir , "early_rl_agents.py"))
copyfile("networks/resnets.py" , os.path.join(scripts_dir, "resnets.py"))
copyfile("networks/rnn_agents.py", os.path.join(scripts_dir, "rnn_agents.py"))

# Save Utils files
copyfile("utils/utils.py", os.path.join(scripts_dir, "utils.py"))
copyfile("utils/training_utils.py" , os.path.join(scripts_dir , "training_utils.py"))
copyfile("utils/agent_utils.py" , os.path.join(scripts_dir, "agent_utils.py"))

# TODO - Add pretrained log dir


# Create folder for saving intermediate models
os.makedirs(os.path.join(CONFIG.STATS_log_dir, "models"))

metrics_dir = os.path.join(CONFIG.STATS_log_dir, "metrics")
os.makedirs(metrics_dir)


info = dict([
    ("AgentType" ,CONFIG.RL_agent_network),
    ("PatchEmbedder" , CONFIG.RL_patch_embedder),
    ("Completed" , False),
    ('Metrics' , [
        'Steps',
        'FinalDistanceOnlyFailure',
        'IoU',
        'ValSteps',
        'ValFinalDistanceOnlyFailure',
        'ValIoU',
#        'GradientSize',
        #'GradientSize',
        'CumulativeRewardToGo',
        'HasConverged',
        'StepRatioOnlySuccess',
        #'StepTime',
        #'PropTime',
        #'CorrectActions',
        'ActionsTaken',
        'ValHasConverged',
        'ValStepRatioOnlySuccess',
        'ValCumulativeRewardToGo',
        #'ValStepTime',
        #'ValPropTime',
        'FullValSteps',
        'FullValIoU',
        'FullValDistance',
        'FullValHasConverged',
        'ValActionsTaken',
        #'ValCorrectActions',
        'SeparatedHasConverged',
        'SeparatedCumulativeRewardToGo',
        'SeparatedSteps',
        'SeparatedIoU',
        'ValSeparatedHasConverged',
        'ValSeparatedCumulativeRewardToGo',
        'ValSeparatedSteps',
        'ValSeparatedIoU',
    ]),
    ("StartedTraining" , str(datetime.now())),
    ("FinishedTraining" , 0),
    ("Dataset",CONFIG.MISC_dataset),
    ("MultiplyImages" , CONFIG.RL_multiply_images),
    ("NbrOfTrainableParameters" , 0),
    ("AgentClass" , "RL"),
    ("FullValIters", []) # At which iterations is the model evaluated on full validation
])


# Set random seed
random.seed(CONFIG.MISC_random_seed)
np.random.seed(CONFIG.MISC_random_seed)
torch.manual_seed(CONFIG.MISC_random_seed)

# load the dataset
trainloader,valloader = load_normalize_data(download = False, batch_size = CONFIG.RL_batch_size , multiply_images = CONFIG.RL_multiply_images)

valloader_iterator = iter(valloader)

im_H, im_W = CONFIG.MISC_im_size
p_H,p_W  = CONFIG.MISC_patch_size
max_len_batch = CONFIG.RL_max_episode_length*CONFIG.RL_batch_size * CONFIG.RL_multiply_images
ep_len = CONFIG.RL_max_episode_length


device = torch.device("cuda:0" if CONFIG.MISC_use_gpu and torch.cuda.is_available() else "cpu")
# Make device globaly available
CONFIG.device = device


# Setup Agent
if CONFIG.RL_agent_network == 'RandomAgent':
    agent_net = RandomAgent()
elif CONFIG.RL_agent_network == 'SpiralAgent':
    agent_net = DeterministicAgent(mode = 'spiral')
else:
    raise "Unknown RL agent selected."


agent_net = agent_net.to(device)

agent_parameters = filter(lambda p: p.requires_grad, agent_net.parameters())
params = sum([np.prod(p.size()) for p in agent_parameters])


# Add number of parameters to the info json
info['NbrOfTrainableParameters'] = int(params)

# Write info dictionary to log directory
with open(os.path.join(CONFIG.STATS_log_dir , "info.json") , 'w') as info_file:
    json.dump(info , info_file ,indent = 4)

# Setup StatCollector
metrics = info['Metrics']
exclude_prints = ['ActionsTaken' , 'ValActionsTaken', 'ValPropTime', 'ValStepTime', 'StepTime',
                    'CumulativeRewardToGo', 'ValCumulativeRewardToGo','StepRatio','ValStepRatio',
                    'HasConverged' , 'ValHasConverged','ValCorrectActions','CorrectActions'] # Does not print these statistics

tot_nbr_iter = CONFIG.RL_nbr_epochs * len(trainloader)
tot_itr = 0
sc = StatCollector(metrics_dir, tot_nbr_iter , print_iter = CONFIG.MISC_print_iter, exclude_prints = exclude_prints)

# Add all metrics to StatCollector
for metric in metrics:
    sc.register(metric , {'type':'avg' ,'freq':'step'})

# Open statistics for dataset to find unnormalizing transform
if CONFIG.MISC_dataset.startswith('custom_'):
    stat_path = os.path.join(CONFIG.MISC_dataset_path,"Custom", CONFIG.MISC_dataset[7:],'stats.json')
else:
    stat_path = os.path.join(CONFIG.MISC_dataset_path,CONFIG.MISC_dataset,'stats.json')
with open(stat_path) as json_file:
    stats = json.load(json_file)

dataset_means = torch.tensor(stats['means'][:3])
dataset_stds = torch.tensor(stats['stds'][:3])
unNormImage = transforms.Normalize( ( - dataset_means / dataset_stds).tolist() , (1.0 / dataset_stds).tolist() )

# Storage objects for all actions, weights, rewards etcetera
batch = BatchStorage(sc)
episode = EpisodeStorage()
val_batch = BatchStorage(sc)
val_episode = EpisodeStorage()

# If enabled there will be an entropy bonus and a linear annealing of this bonus
entropy_bonus = CONFIG.RL_entropy
if CONFIG.RL_entropy_lower is not None and CONFIG.RL_entropy is not None:
    # Linear annealing
    #entropy_anneal_k = (entropy_bonus - CONFIG.RL_entropy_lower ) / tot_nbr_iter
    entropy_anneal_k_exp = math.exp( math.log( CONFIG.RL_entropy_lower / entropy_bonus) / tot_nbr_iter)

# Increase recursion limit for debugging
sys.setrecursionlimit(2000)

# Initialize all clocations to zero to avoid crashing later
loc_goal = None
loc_start = None
start_coord = None
goal_coord = None
rep_counter = 91

# Print out info regarding this training run
print("Starting training at:\t%s" % info['StartedTraining'])
print("Agent Network:\t%s" % CONFIG.RL_agent_network)
print("Patch Embedder:\t%s" % CONFIG.RL_patch_embedder)
print("Dataset:\t%s" % CONFIG.MISC_dataset)

try:
    for epoch in range(CONFIG.RL_nbr_epochs):

        # Passes through the dataset in batches
        for batch_counter,batch_data in enumerate(trainloader):

            batch_images , (start_crops_ , goal_crops_)  = batch_data

            batch.initialize(batch_size = len(batch_images))


            # Each image/episode is handled seperatly
            for (episode_counter, episode_data) in enumerate(batch_images):

                full_image = episode_data[None,:].to(device)
                loc_start , loc_goal = None, None

                # Initializes all training tensors and sets start and goal patch
                episode.initialize(image = full_image, loc_goal = loc_goal, loc_start =  loc_start)
                done = False

                # Run one episode of training
                while not done:
                    # Get an action from the agent, given current trajectory
                    action, softmax_embedding = get_action(agent_net , episode)

                    # Update environment according to action
                    loc_next, reward, done = take_step(action , episode, softmax_embedding)

                    # Get the visible crop at current position
                    crop_current,loc_current = get_deterministic_crops( full_image, coords = loc_next[0])

                    # Update the episode
                    episode.update(action, reward, loc_current , crop_current)

                # Finish the episode, count rewards to go, iou etcetera
                episode.finish()

                # RNN networks need to reset their hidden states
                if agent_net.AGENT_TYPE == 'RNN':
                    agent_net.reset()

                # Add result from episode to batch
                batch.append_episode(episode)

            batch.prepare_for_loss()


            batch.store()

            # Run the current model on one batch from the valloader
            with torch.no_grad():

                try:
                    images , (start_coords_, goal_coords_) = next(valloader_iterator)
                except:
                    valloader_iterator = iter(valloader)
                    images , (start_coords_, goal_coords_) = next(valloader_iterator)

                val_batch.initialize(batch_size=len(images))

                for (counter_val , episode_data) in enumerate(images):
                    episode_image = episode_data[None , :].to(device)
                    start_coord , goal_coord = None, None

                    val_episode = run_eval_trajectory( episode_image, val_episode, agent_net, loc_start = start_coord , loc_goal= goal_coord)
                    val_batch.append_episode(val_episode)

                if tot_itr % CONFIG.MISC_save_vis_iter == 0:
                    visualize_batch(val_batch, PATH = vis_dir, transform = unNormImage, save_name = 'val_batch_' + str(tot_itr))

                # Save result
                val_batch.store(mode = 'Val')

            if CONFIG.RL_entropy_lower is not None and CONFIG.RL_entropy is not None:
                entropy_bonus *=  entropy_anneal_k_exp

            if tot_itr % CONFIG.MISC_print_iter == 0 or tot_itr == tot_nbr_iter:
                print("Iter: %d / %d" % (tot_itr , tot_nbr_iter))
                batch.sc.print()
                batch.sc.save()

            # Increment total iteration counter by one
            tot_itr += 1

            # BATCH COMPLETE

except KeyboardInterrupt:
    print("\nInterrupted")
    while True:
        i = input("\n Save model? (y/n)")
        if i == "y":
            print("Saving Model")
            torch.save(agent_net.state_dict() , os.path.join(CONFIG.STATS_log_dir , "final_model"))
            sys.exit(1)
        elif i == "n":
            print("Not Saving Model")
            sys.exit(1)


        print("No valid input")



print("Training finished!")

info['Completed'] = True

info["FinishedTraining"] = str(datetime.now())

# Write completed status to info.json
with open(os.path.join(CONFIG.STATS_log_dir , "info.json") , 'w') as info_file:
    json.dump(info , info_file ,indent = 4) # Save final model

torch.save(agent_net.state_dict() , os.path.join(CONFIG.STATS_log_dir , "final_model"))
