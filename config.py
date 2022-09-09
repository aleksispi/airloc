"""
Central configration file for the project. Acts as a storage
of global variables and various configuration settings.

TODO TODO TODO: Remove unnecessary fields below; this is copied from an earlier
project
"""

import os
import pprint
from easydict import EasyDict as edict
from datetime import datetime
import socket # To Resolve host name and set path appropriatly


# TODO: If we're to use rllib, then this may need to be uncommented.
#from rllib.utils import rllib_get_config

CONFIG = edict()

"""
Evaluation of RL-agent
"""
# These settings control the evaluation runs of the saved agents.
# EVAL_RL_log is which saved agent should be used. If a number n, it picks the n:th,
# latest log availabled. Note. n=1 picks the penultimate available log
# If set to a specific log it tries to load that log
CONFIG.EVAL_RL_log = None
CONFIG.EVAL_RL_saved_logs = False # If enabled picks the model from those in saved_logs
CONFIG.EVAL_RL_multiply_images = 1
CONFIG.EVAL_save_vis_iter = 10
CONFIG.EVAL_RL_use_val_set = True

"""
RL-agent
"""
######################### This is where the important settings start #########################
# Batch n Stuff
CONFIG.RL_nbr_epochs = 10000
CONFIG.RL_batch_size = 32
CONFIG.RL_multiply_images = 2
CONFIG.RL_max_episode_length = 10
CONFIG.MISC_priv = False
# Architecture
CONFIG.RL_agent_network = 'LSTMAgent'  # AiRLoc agent
CONFIG.RL_patch_embedder = 'ShareNet'
CONFIG.RL_freeze_patch_embedder = True
CONFIG.RL_priv_pretrained = True
CONFIG.EE_temporal = True
CONFIG.EE_residual = True

# Optimizer
CONFIG.RL_learning_rate = 1e-4
CONFIG.RL_nbr_eps_update = (CONFIG.RL_batch_size * CONFIG.RL_multiply_images)//1
CONFIG.RL_weight_decay = 0
CONFIG.RL_momentum = 0.90
CONFIG.RL_optimizer = 'adam'
CONFIG.RL_beta1 = 0.9
CONFIG.RL_beta2 = 0.999
#Env setup
CONFIG.RL_agent_allowed_outside = True
CONFIG.RL_normalize_weights = True
CONFIG.RL_eval_deterministic = True
CONFIG.RL_priv_grid_location = False
CONFIG.RL_priv_use_seg = True  # Set to True when training sem seg-based RL-agent (but False during inference -- should not use ground truth then!)

"""
RL Rewards
"""
CONFIG.RL_reward_goal = 3
CONFIG.RL_reward_failed = 0
CONFIG.RL_reward_closer = 0
CONFIG.RL_reward_iou_scale = 0
CONFIG.RL_reward_step_outside = 0
CONFIG.RL_reward_distance = False
CONFIG.RL_reward_step = -1

# LSTM Agent settings
CONFIG.RL_LSTM_pos_emb = True

# Pretrained doerch
#CONFIG.RL_pretrained_doerch_net = 'doerchnet/logs/without-sem-seg'  # without sem-seg
CONFIG.RL_pretrained_doerch_net = 'doerchnet/logs/with-sem-seg'  # with sem-seg

######################### This is where they end #########################

CONFIG.RL_max_start_goal_dist = 8
CONFIG.RL_min_start_goal_iou = None
CONFIG.RL_done_iou = 0.40
CONFIG.RL_discount_factor = 0.9
CONFIG.RL_softmax_step_size = 1.1 # When 1 step equal non-overlapping patches

CONFIG.RL_entropy = None
CONFIG.RL_entropy_lower = None

# Pretrained segmenter
CONFIG.RL_pretrained_segmentation_net = 'segmentations/logs/sem-seg-model'
CONFIG.RL_predict_seg_mask = False  # Set to True during inference if using a sem-seg based RL-agent

"""
Random Search baseline agent
"""
CONFIG.RANDOM_batch_size = 1
CONFIG.RANDOM_using_memory = True  # If true, the agent cannot visit the same patch twice
CONFIG.RANDOM_stop_iou = 0.2  # Not used in grid game setup
CONFIG.RANDOM_min_iou_visited = 0.3  # At what IoU should a location be considered already visited (not used in grid game setup)
CONFIG.RANDOM_WARNING_steps = 500  # Warn user if agent takes this many step without funding goal

"""
Statistics / Logging / Plotting
"""
CONFIG.STATS_dir_base = os.path.dirname(os.path.abspath(__file__))
CONFIG.STATS_log_dir_base = os.path.join(CONFIG.STATS_dir_base, 'logs')
CONFIG.STATS_log_dir = os.path.join(CONFIG.STATS_log_dir_base,
                                    str(datetime.now()).replace(' ', '_')
                                    .replace(':', '-').replace('.', '-'))

"""
Plotting
"""
# The option below lets the user choose which LOG directory to plot information from
# An integer signifies the n:th most recent log. A specific log name tries to find that directory
CONFIG.PLOT_log_dir = 1
# The option below lets the user choose which EVAL directory to plot information from.
# I.e, choose which eval session to plot from given a specific training session
CONFIG.PLOT_eval_dir = None

"""
Miscellaneous
"""
CONFIG.MISC_include_baseline = True
CONFIG.MISC_use_gpu = True
CONFIG.MISC_dataset = 'masa_filt'
CONFIG.MISC_dataset_split_file = None
CONFIG.MISC_grid_game = True
CONFIG.MISC_random_seed = 0
#CONFIG.MISC_rnd_crop = True
CONFIG.MISC_rgb_max = 255
#CONFIG.MISC_im_size = (256, 256)
CONFIG.MISC_step_sz = int(48*CONFIG.RL_softmax_step_size)
CONFIG.MISC_game_size = 5
CONFIG.MISC_im_size = (int(CONFIG.MISC_step_sz*(CONFIG.MISC_game_size-1)+48),
                       int(CONFIG.MISC_step_sz*(CONFIG.MISC_game_size-1)+48))
CONFIG.MISC_patch_size = (48, 48)
CONFIG.MISC_print_iter = 50
CONFIG.MISC_save_vis_iter = 400  # How often we save a visualiation
CONFIG.MISC_vis_per_batch = 12
CONFIG.MISC_save_model_iter = 5000  # How often should we save the model weights
CONFIG.MISC_project_root_path = os.path.dirname(__file__)
CONFIG.MISC_main_pid = os.getpid()

CONFIG.MISC_dataset_path = "data"  # Set accordingly
