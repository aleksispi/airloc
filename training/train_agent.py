import traceback
import pdb
from datetime import datetime
import random
import signal
import time
import numpy as np
import json
import os
import sys
# matplotlib is used for debugging image inputs to networks
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import math
from config import CONFIG
from utils.utils import load_normalize_data, visualize_trajectory,\
                        get_deterministic_crops, _setup_fixed_games,\
                        visualize_batch, setupLogDir
# Import agent utils
from utils.agent_utils import take_step, get_action, run_eval_trajectory, update_net

from networks.agent import Agent
from networks.RandomAgent import RandomAgent
from networks.rnn_agents import LSTMAgent

from utils.stat_collector import StatCollector
from torch.utils.tensorboard import SummaryWriter
from utils.training_utils import BatchStorage , EpisodeStorage

# Creates logging directory for this run
setupLogDir(CONFIG.STATS_log_dir)

def signalInterruptHandler(*args):
    """
        Signal handler for Interrup Signal. Saves current network weights and exits.
    """
    response = ''
    if CONFIG.MISC_main_pid != os.getpid():
        return
    while (response != "y" and response != "n"):
        response = input("Program interrupted, do you want to exit? (y/n)\t")
    if response == "n":
        print("Continuing training!")
        return
    elif response == "y":

        # Aborting training save information and exit
        info['FinishedOnIter'] = tot_itr
        with open(os.path.join(CONFIG.STATS_log_dir , "info.json") , 'w') as info_file:
            json.dump(info , info_file ,indent = 4) # Save final model

        response = ''
        while (response != "y" and response != "n"):
            response = input("Do you want to save network weights? (y/n)\t")
        if response == "y":
            print("Saving network weights!")
            torch.save(agent_net.state_dict() , os.path.join(CONFIG.STATS_log_dir , "final_model"))
        elif response == "n":
            print("Not saving network weights!")

        # Needs to terminate from main thread. There is a check for this variable in intitalize episode
        CONFIG.TERMINATE = True

def signalSIGUSR1Handler(*args):
    """
        When program receives SIGUSR1 print the log directory.
    """
    if CONFIG.MISC_main_pid == os.getpid():
        print("\nLog directory:\t%s\n" % CONFIG.STATS_log_dir)


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
        'StepMSE',
        'StepMax',
        'CumulativeRewardToGo',
        'HasConverged',
        'StepRatioOnlySuccess',
        'Entropy',
        'Loss',
        'Difficulty',
        'ActionsTaken',
        'ValDifficulty',
        'ValHasConverged',
        'ValStepRatioOnlySuccess',
        'ValCumulativeRewardToGo',
        'FullValSteps',
        'FullValIoU',
        'FullValDistance',
        'FullValHasConverged',
        'ValActionsTaken',
        'SeparatedSteps',
        'SeparatedIoU',
        'SeparatedCumulativeRewardToGo',
        'SeparatedHasConverged',
        'ValSeparatedSteps',
        'ValSeparatedIoU',
        'ValSeparatedCumulativeRewardToGo',
        'ValSeparatedHasConverged',
    ]),
    ("StartedTraining" , str(datetime.now())),
    ("FinishedTraining" , 0),
    ("Dataset",CONFIG.MISC_dataset),
    ("MultiplyImages" , CONFIG.RL_multiply_images),
    ("NbrOfTrainableParameters" , 0),
    ("AgentClass" , "RL"),
    ("FinishedOnIter" , 0),
    ("FullValIters", []), # At which iterations is the model evaluated on full validation
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
if CONFIG.RL_agent_network == 'LSTMAgent':
    agent_net = LSTMAgent()
elif CONFIG.RL_agent_network == 'Agent':
    agent_net = Agent()
elif CONFIG.RL_agent_network == 'RandomAgent':
    agent_net = RandomAgent()
else:
    raise "Unknown RL agent selected."

agent_net = agent_net.to(device)

agent_parameters = filter(lambda p: p.requires_grad, agent_net.parameters())
params = sum([np.prod(p.size()) for p in agent_parameters])


# Add number of parameters to the info json
info['NbrOfTrainableParameters'] = int(params)
# Setup loss
criterion = nn.MSELoss()
if CONFIG.RL_optimizer == 'sgd':
    optimizer = optim.SGD(agent_net.parameters() , lr = CONFIG.RL_learning_rate ,
                        weight_decay = CONFIG.RL_weight_decay ,
                        momentum = CONFIG.RL_momentum)
elif CONFIG.RL_optimizer == 'adam':
    optimizer = optim.Adam(agent_net.parameters() , lr = CONFIG.RL_learning_rate ,
                        weight_decay = CONFIG.RL_weight_decay ,
                        betas = (CONFIG.RL_beta1 , CONFIG.RL_beta2) )

# Write info dictionary to log directory
with open(os.path.join(CONFIG.STATS_log_dir , "info.json") , 'w') as info_file:
    json.dump(info , info_file ,indent = 4)

# Setup StatCollector
metrics = info['Metrics']
exclude_prints = [
                'ActionsTaken' , 'ValActionsTaken', 'ValPropTime', 'ValStepTime', 'StepTime', 'SeparatedSteps', 'SeparatedIoU',
                  'SeparatedCumulativeRewardToGo', 'SeparatedHasConverged', 'ValSeparatedSteps', 'ValSeparatedIoU',
                  'ValSeparatedCumulativeRewardToGo', 'ValSeparatedHasConverged','CumulativeRewardToGo', 'ValCumulativeRewardToGo',
                  'StepRatio','ValStepRatio', 'HasConverged' , 'ValHasConverged','ValCorrectActions','CorrectActions'
                 ] # Does not print these statistics


tot_nbr_iter = CONFIG.RL_nbr_epochs * len(trainloader)
tot_itr = 0
sc = StatCollector(CONFIG.STATS_metrics_dir, tot_nbr_iter , print_iter = CONFIG.MISC_print_iter, exclude_prints = exclude_prints)

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

if CONFIG.MISC_include_baseline:
    base_batch = BatchStorage(sc)
    base_episode = EpisodeStorage()
    base_agent = RandomAgent()

# If enabled there will be an entropy bonus and a linear annealing of this bonus
entropy_bonus = CONFIG.RL_entropy
if CONFIG.RL_entropy_lower is not None and CONFIG.RL_entropy is not None:
    # Linear annealing
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
print("Trainloader length:\t%d" % len(trainloader) )
print("Valloader length:\t%d" % len(valloader) )

# Attach signal handler for Interrupt signal
signal.signal(signal.SIGINT , signalInterruptHandler)
signal.signal(signal.SIGUSR1 , signalSIGUSR1Handler)

CONFIG.TERMINATE = False

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

                # Debugging: Visualize training trajectory
                #visualize_trajectory(episode, save_name= 'train_vis_%d_%d' %(tot_itr,episode_counter),transform = unNormImage)

                # Add result from episode to batch
                batch.append_episode(episode)

            batch.prepare_for_loss()

            if any(batch.weights[:,0]==torch.nan): print("Weights contains Nan")


            #batch.sc.s('GradientSize').collect(agent_net.patch_emb.common_fc_2.weight.grad.abs().mean().item())

            prev_net = deepcopy(agent_net)

            t = time.perf_counter()
            update_net(batch, agent_net, optimizer, entropy_bonus)
            batch.proptime =  time.perf_counter() - t

            batch.store()

            se = 0
            max = 0.
            tot_params = 0
            for params1,params2 in zip(agent_net.parameters(),prev_net.parameters()):
                temp_se = (params1-params2).square().sum().item()
                se += temp_se
                max = np.maximum(max ,(params1-params2).abs().max().item())
                if temp_se > 0:
                    tot_params += torch.numel(params1)

            se = np.sqrt(se/tot_params)
            # Old loss update method now we send in the optimizer to the
            # network
            batch.sc.s('StepMSE').collect(se)
            batch.sc.s('StepMax').collect(max)

            # Run the current model on one batch from the valloader
            with torch.no_grad():

                try:
                    images , (start_coords_, goal_coords_) = next(valloader_iterator)
                except:
                    valloader_iterator = iter(valloader)
                    images , (start_coords_, goal_coords_) = next(valloader_iterator)

                val_batch.initialize(batch_size=len(images))
                if tot_itr % CONFIG.MISC_save_vis_iter == 0 and CONFIG.MISC_include_baseline:
                    base_batch.initialize(batch_size=len(images))

                for (counter_val , episode_data) in enumerate(images):
                    episode_image = episode_data[None , :].to(device)
                    start_coord , goal_coord = None, None

                    # RNN networks need to reset their hidden states
                    if agent_net.AGENT_TYPE == 'RNN':
                        agent_net.reset()
                    val_episode = run_eval_trajectory( episode_image, val_episode, agent_net, loc_start = start_coord , loc_goal= goal_coord)
                    # RNN networks need to reset their hidden states
                    if agent_net.AGENT_TYPE == 'RNN':
                        agent_net.reset()

                    if tot_itr % CONFIG.MISC_save_vis_iter == 0 and CONFIG.MISC_include_baseline:
                        loc_start = val_episode.loc_start
                        loc_goal = val_episode.loc_goal
                        base_episode = run_eval_trajectory( episode_image,
                                                           base_episode,
                                                           agent_net, loc_start = loc_start , loc_goal= loc_goal, deterministic=False)
                        # RNN networks need to reset their hidden states
                        if agent_net.AGENT_TYPE == 'RNN':
                            agent_net.reset()
                        base_batch.append_episode(base_episode)

                        # RNN networks need to reset their hidden states
                        if agent_net.AGENT_TYPE == 'RNN':
                            agent_net.reset()

                    val_batch.append_episode(val_episode)


                if tot_itr % CONFIG.MISC_save_vis_iter == 0:
                    visualize_batch(val_batch, PATH = CONFIG.STATS_vis_dir, transform = unNormImage, save_name = 'val_batch_' + str(tot_itr))
                    if  CONFIG.MISC_include_baseline:
                        visualize_batch(base_batch, PATH = CONFIG.STATS_vis_dir, transform = unNormImage, save_name = 'val_batch_' + str(tot_itr), prefix='random')

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
            if tot_itr % CONFIG.MISC_save_model_iter == 0:
                torch.save(agent_net.state_dict(), os.path.join(CONFIG.STATS_log_dir, "model_%d" % tot_itr))

            # BATCH COMPLETE

except Exception as e:
    info['Exception'] = str(e)
    info['BackTrace'] = traceback.format_exc()
    info['FinishedTraining'] = str(datetime.now())
    info['FinishedOnIter'] = tot_itr

    with open(os.path.join(CONFIG.STATS_log_dir , "info.json") , 'w') as info_file:
        json.dump(info , info_file ,indent = 4)

    print("\nAn exception occurred in the main loop!\n" + "\n"+ "#"*60 + "\n")
    print(info['BackTrace'])
    print("#"*60)

    # Enter pdb to investigate
    pdb.set_trace()

print("Training finished!")

info['Completed'] = True

info["FinishedTraining"] = str(datetime.now())
info['FinishedOnIter'] = tot_itr

# Write completed status to info.json
with open(os.path.join(CONFIG.STATS_log_dir , "info.json") , 'w') as info_file:
    json.dump(info , info_file ,indent = 4) # Save final model

torch.save(agent_net.state_dict() , os.path.join(CONFIG.STATS_log_dir , "final_model"))
