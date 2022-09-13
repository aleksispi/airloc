from datetime import datetime
import random
import time
import numpy as np
import json
import os
import sys
import importlib
# matplotlib is used for debugging image inputs to networks
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import torch
import torchvision.transforms as transforms
import argparse
from config import CONFIG
from utils.utils import load_normalize_data, visualize_batch, find_latest_log
# Import agent utils
from utils.agent_utils import run_eval_trajectory
from networks.agent import Agent
from networks.RandomAgent import RandomAgent
from networks.rnn_agents import LSTMAgent
from utils.stat_collector import StatCollector
from torch.utils.tensorboard import SummaryWriter
from utils.training_utils import BatchStorage, EpisodeStorage


from config import CONFIG

def replace_config(loaded_config):
    for key in loaded_config.keys():
        if 'EVAL' not in key and 'device' not in key and 'MISC_main_pid' not in key and not 'dataset' in key and not 'allowed_outside' in key:
            CONFIG[key] = loaded_config[key]


def main(args ):
    if args.seed:
        seed = args.seed
    else:
        seed = 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Get the log in the correct position in the log dir
    log_base = "logs"
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = find_latest_log(log_base , args.n)

    eval_metrics_path = os.path.join(log_base,log_dir,'metrics_eval')
    os.makedirs(eval_metrics_path,exist_ok=True)

    # Import the CONFIG file from the log
    scripts_dir = os.path.join(log_base,log_dir)
    networks_file_path = os.path.join(scripts_dir, "config.py")
    spec = importlib.util.spec_from_file_location("config",networks_file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    loaded_config = config_module.CONFIG

    replace_config(loaded_config)
    CONFIG.STATS_dir_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..')
    CONFIG.STATS_log_dir_base = os.path.join(CONFIG.STATS_dir_base, log_base)
    CONFIG.STATS_log_dir = os.path.join(CONFIG.STATS_log_dir_base,log_dir)
    CONFIG.MISC_vis_per_batch = 1
    CONFIG.MISC_vis_iter = 1
    CONFIG.TERMINATE = False

    vis_dir = os.path.join(log_base,log_dir, "visualizations")
    # The info dictionary for what to store
    info = dict([
        ("AgentType" ,CONFIG.RL_agent_network),
        ("PatchEmbedder" , CONFIG.RL_patch_embedder),
        ("Completed" , False),
        ('Metrics' , [
            'EpisodeTime',
            'StocSteps',
            'DetSteps',
            'StocIoU',
            'DetIoU',
            'StocStepRatioOnlySuccess',
            'DetStepRatioOnlySuccess',
            'StocFinalDistanceOnlyFailure',
            'DetFinalDistanceOnlyFailure',
            'StocCumulativeRewardToGo',
            'DetCumulativeRewardToGo',
            'StocHasConverged',
            'DetHasConverged',
            'StocDifficulty',
            'DetDifficulty',
            'StocActionsTaken',
            'DetActionsTaken',
            'StocSeparatedSteps',
            'DetSeparatedSteps',
            'StocSeparatedIoU',
            'DetSeparatedIoU',
            'StocSeparatedCumulativeRewardToGo',
            'DetSeparatedCumulativeRewardToGo',
            'StocSeparatedHasConverged',
            'DetSeparatedHasConverged',
            'StocGoalLoc',
            'DetGoalLoc',
            'StocActionProbs',
            'DetActionProbs',
        ]),
        ("StartedEvalAt" , str(datetime.now())),
        ("FinishedTraining" , 0),
        ("Dataset",CONFIG.MISC_dataset),
        ("MultiplyImages" , CONFIG.RL_multiply_images),
        ("NbrOfTrainableParameters" , 0),
        ("AgentClass" , "RL"),
        ("FullValIters", []) # At which iterations is the model evaluated on full validation
    ])
    _,valloader = load_normalize_data(download = False, batch_size = 1 ,
                                      multiply_images = 1,split=args.split,
                                      use_eval_split = args.eval_split
                                      )
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
    agent_net.load_state_dict(torch.load(os.path.join(log_base,log_dir,'final_model')), strict=False)
    agent_net.eval()
    agent_net = agent_net.to(device)

    metrics = info['Metrics']
    exclude_prints = [
            'StocFinalDistanceOnlyFailure',
            'StocCumulativeRewardToGo',
            'StocStepRatioOnlySuccess',
            'StocActionsTaken',
            'StocSeparatedSteps',
            'StocSeparatedIoU',
            'StocSeparatedCumulativeRewardToGo',
            'StocSeparatedHasConverged',
            'DetFinalDistanceOnlyFailure',
            'DetStepRatioOnlySuccess',
            'DetCumulativeRewardToGo',
            'DetActionsTaken',
            'DetSeparatedSteps',
            'DetSeparatedIoU',
            'DetSeparatedCumulativeRewardToGo',
            'DetSeparatedHasConverged',
            'StocGoalLoc',
            'DetGoalLoc',
            'StocActionProbs',
            'DetActionProbs',
                     ] # Does not print these statistics
    num_passes = 1
    tot_nbr_iter = num_passes* len(valloader)
    tot_itr = 0
    sc = StatCollector(eval_metrics_path, tot_nbr_iter , print_iter = CONFIG.MISC_print_iter, exclude_prints = exclude_prints)

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
    stoc_batch = BatchStorage(sc)
    stoc_episode = EpisodeStorage()

    det_batch = BatchStorage(sc)
    det_episode = EpisodeStorage()

    # Print out info regarding this training run
    print("Starting eval at:\t%s" % info['StartedEvalAt'])
    print("Agent Network:\t%s" % CONFIG.RL_agent_network)
    print("Patch Embedder:\t%s" % CONFIG.RL_patch_embedder)
    print("Dataset:\t%s" % CONFIG.MISC_dataset)

    misc_info = []
    with torch.no_grad():
        for epoch in range(num_passes):

            # Passes through the dataset in batches
            for batch_counter,batch_data in enumerate(valloader):

                # Get the images from the batch
                batch_images , (start_crops_ , goal_crops_)  = batch_data

                # Initialize the utils
                stoc_batch.initialize(batch_size = len(batch_images))
                det_batch.initialize(batch_size = len(batch_images))

                # Loop over the batch of images
                for (episode_counter, episode_data) in enumerate(batch_images):
                    episode_image = episode_data[None, :].to(device)

                    start_coord, goal_coord = start_crops_[0, :], goal_crops_[0, :]

                    # TODO: ENABLE NOT RUNNING STOC TO SAVE TIME !!

                    #stoc_episode = run_eval_trajectory(episode_image, stoc_episode, agent_net, loc_start=start_coord, loc_goal=goal_coord, deterministic=False)
                    if agent_net.AGENT_TYPE == 'RNN':
                        agent_net.reset()
                    #t1 = time.process_time()
                    det_episode = run_eval_trajectory(episode_image, det_episode, agent_net, loc_start=start_coord, loc_goal=goal_coord, deterministic=True)
                    if True:
                        misc_info.append(np.concatenate([det_episode.actions.cpu().detach().numpy(),
                                                         np.squeeze(det_episode.misc, axis=1),
                                                         det_episode.weights.cpu().detach().numpy()[:, np.newaxis],
                                                         det_episode.dists.cpu().detach().numpy()[:, np.newaxis]], axis=1))

                    #t2 = t1 - time.process_time()
                    if agent_net.AGENT_TYPE == 'RNN':
                        agent_net.reset()

                #stoc_batch.append_episode(stoc_episode)
                det_batch.append_episode(det_episode)
                #stoc_batch.store(mode = 'Stoc',eval=True)
                det_batch.store(mode = 'Det',eval=True)
                #det_batch.sc.s('EpisodeTime').collect(t2)
                if tot_itr % CONFIG.EVAL_save_vis_iter == 0 :
                    visualize_batch(det_batch, PATH = vis_dir, transform = unNormImage, save_name = 'eval_' + str(tot_itr), prefix='Det')
                    #visualize_batch(stoc_batch, PATH = vis_dir, transform = unNormImage, save_name = 'eval_' + str(tot_itr), prefix='Stoc')

                if tot_itr % CONFIG.MISC_print_iter == 0 or tot_itr == tot_nbr_iter:
                    print("Iter: %d / %d" % (tot_itr , tot_nbr_iter))
                    det_batch.sc.print()
                    det_batch.sc.save()

                # Increment total iteration counter
                tot_itr += 1

        print("Iter: %d / %d" % (tot_itr , tot_nbr_iter))
        det_batch.sc.save()
        stat_path = os.path.join(CONFIG.STATS_log_dir,'final_stats.txt')
        with open(stat_path, 'a') as file:
            print(f"Restults from {CONFIG.MISC_dataset} using {args.split}-set", file=file)
        print(f"Restults from {CONFIG.MISC_dataset} using {args.split}-set")
        det_batch.sc.print()
        det_batch.sc.exclude_prints = None
        det_batch.sc.print(path=stat_path)
        np.save('misc_info', misc_info)
        print("Evaluation completed!")

if __name__ == '__main__':
    # This part is used to be able to simply generate split files for datasets

    parser = argparse.ArgumentParser()
    log_args = parser.add_mutually_exclusive_group()
    log_args.add_argument( "-n", type = int, help = "Select log number",default = 0)
    log_args.add_argument("--log_dir", type = str, help = "Select log name", default=None)
    parser.add_argument("--no_cuda", action='store_true', help = "Disable cuda", default=False)
    parser.add_argument("--seed", type = int, help = "Set seed", default=None)
    parser.add_argument("--split", type = str, help = "Set split", default='val')
    parser.add_argument("--eval_split", type = str, help = "Set split", default='basic')
    args = parser.parse_args()
    main(args)
