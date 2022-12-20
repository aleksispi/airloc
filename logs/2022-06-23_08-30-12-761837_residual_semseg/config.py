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
Evaluation RL agent
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
CONFIG.RL_repeat_game = False
CONFIG.MISC_priv = False
# Architecture
CONFIG.RL_agent_network = 'LSTMAgent'
CONFIG.RL_patch_embedder = 'ShareNet'
CONFIG.RL_mask_embedder = 'Regular' # Not active
CONFIG.RL_freeze_patch_embedder = True
CONFIG.RL_priv_pretrained = True

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
CONFIG.RL_normalize_method = 'grid' # When grid game enabled, dist becomes grid_distance
CONFIG.RL_eval_deterministic = True
CONFIG.RL_priv_grid_location = False
CONFIG.RL_priv_use_seg = False#True

# Continuous
CONFIG.RL_dist_var = 2162.25
CONFIG.RL_anneal_lwr = 132.56 # The lower limit for the annealing of the variace, if none no annealing
"""
RL Rewards
"""
CONFIG.RL_reward_goal = 3
CONFIG.RL_reward_difficulty = 0
CONFIG.RL_reward_failed = 0
CONFIG.RL_reward_closer = 0
CONFIG.RL_reward_iou_scale = 0
CONFIG.RL_reward_step_outside = 0
CONFIG.RL_reward_distance = False
CONFIG.RL_reward_step = -1
CONFIG.RL_reward_exploit_not_adjacent = 0#-0.4
CONFIG.RL_reward_exploit_adjacent2goal = 0#1

# LSTM Agent settings
CONFIG.RL_LSTM_pos_emb = True
CONFIG.LSTM_global_pos = True


# Pretrained doerch
#CONFIG.RL_pretrained_doerch_net = 'doerchnet/logs/without-sem-seg'  # without sem-seg
CONFIG.RL_pretrained_doerch_net = 'doerchnet/logs/with-sem-seg'  # with sem-seg

#########################This is where they end #########################

CONFIG.RL_max_start_goal_dist = 8
CONFIG.RL_min_start_goal_iou = None
CONFIG.RL_done_iou = 0.40
CONFIG.RL_discount_factor = 0.9
CONFIG.RL_softmax_agent = True
CONFIG.RL_softmax_step_size = 1.1 # When 1 step equal non-overlapping patches

CONFIG.RL_entropy = None
CONFIG.RL_entropy_lower = None
CONFIG.RL_froozen_embedder_iter = None


# Agent MISCs
CONFIG.RL_freeze_mask_embedder = False

# Control the difficulty of the sampled games
CONFIG.RL_init_diff = None  # [0.8, 0.15, 0.05 , 0.0]
CONFIG.RL_final_diff = None  # [0.25 ,0.25, 0.25 , 0.25]
CONFIG.RL_anneal_diff_iter = 10000

# Priv info to agent
CONFIG.RL_agent_mask = False # Agent centric mask
CONFIG.RL_priv_full_image = False
CONFIG.RL_priv_visit_mask = False  # Masksize equal to image size
CONFIG.RL_small_mask = 48*2
CONFIG.extended_mask = False

# Transformer Agent Settings
CONFIG.RL_TRANSFORMER_pos_emb = 'half'

# Pretrained segmenter
CONFIG.RL_pretrained_segmentation_net = 'segmentations/logs/sem-seg-model'
CONFIG.RL_predict_seg_mask = True#False

# Doerch 80 dim
# CONFIG.RL_pretrained_doerch_net = 'doerchnet/logs/2022-04-25_13-54-35-226221'

# Used to load the pretrained resnet
CONFIG.RL_pretrained_cifar_net = 'cifar/logs/2022-02-23_16-27-52-943928'
CONFIG.RL_freeze_pretrained_cifar = False

"""
EE-Agent settings
"""
CONFIG.EE_residual = True
CONFIG.EE_hierarch = False
CONFIG.EE_exploit_priv = True
#CONFIG.EE_start_hierarch_it = 10000

"""
One-step supervised agent
"""
CONFIG.ONESTEP_optimizer = 'adam'  # 'sgd' or 'adam'
CONFIG.ONESTEP_nbr_epochs = 1200
CONFIG.ONESTEP_learning_rate =  1e-4
CONFIG.ONESTEP_batch_size = 16

CONFIG.ONESTEP_momentum = 0.9
CONFIG.ONESTEP_beta1 = 0.5
CONFIG.ONESTEP_beta2 = 0.999
CONFIG.ONESTEP_weight_decay = 0.0
CONFIG.ONESTEP_use_pen_fun = False

# Networks: SimplestNet,SimplestNet_with_targets,DoerschNet,DoerschNetWithPriv
CONFIG.ONESTEP_network = 'SimplestBranchedNet'  # 'DoerschNet' or 'SimplestNet'
CONFIG.ONESTEP_max_start_goal_dist = 100
CONFIG.ONESTEP_min_start_goal_iou = 0.0
# CONFIG.ONESTEP_augment_training_data = True
CONFIG.ONESTEP_pretrain_weights = ""

# Enable privlidged information
CONFIG.ONESTEP_priv_full_img = True
CONFIG.ONESTEP_priv_target_distance_fc = False # Adds target distance to final fc layer
CONFIG.ONESTEP_priv_target_distance_ch = False # Adds target distance as a channel in input
CONFIG.ONESTEP_priv_target_direction_fc = True



"""
Exaustive Search Agent
"""
CONFIG.EXAUST_batch_size=1
CONFIG.EXAUST_stop_iou=1
CONFIG.EXAUST_max_start_goal_dist = None
CONFIG.EXAUST_min_start_goal_iou = None


"""
Random Search baseline agent.
"""
CONFIG.RANDOM_batch_size = 1
CONFIG.RANDOM_using_memory = True #If true the agent cannot visit the same patch twice
CONFIG.RANDOM_max_start_goal_dist = None
CONFIG.RANDOM_min_start_goal_iou = None
CONFIG.RANDOM_stop_iou = 0.2
CONFIG.RANDOM_min_iou_visited = 0.3 # At what IoU should a location be considered already visited
CONFIG.RANDOM_WARNING_steps = 500 # Warn user if agent takes this many step without funding goal

"""
Statistics / Logging / Plotting
"""
CONFIG.STATS_dir_base = os.path.dirname(os.path.abspath(__file__))
CONFIG.STATS_log_dir_base = os.path.join(CONFIG.STATS_dir_base, 'logs')
CONFIG.STATS_log_dir = os.path.join(CONFIG.STATS_log_dir_base,
                                    str(datetime.now()).replace(' ', '_')
                                    .replace(':', '-').replace('.', '-'))
CONFIG.STATS_tensorboard_dir = os.path.join(CONFIG.STATS_log_dir, 'tb')


# CONFIG.STATS_save_model_batch = 250

"""
Plotting
"""
CONFIG.PLOT_plot_vis = True # TODO - not yet implemented
CONFIG.PLOT_use_saved_logs = True # This option selects wether to use "saved_logs" or "logs"
CONFIG.PLOT_training = True # plot training data (or eval data)

# The option below lets the user choose which LOG directory to plot information from
# As before, an integer signifies the n:th most recent log
# A specific log name tries to find that directory
CONFIG.PLOT_log_dir = 1
# The option below lets the user choose which EVAL directory to plot information from.
# I.e, choose which eval session to plot from given a specific training session
CONFIG.PLOT_eval_dir = None


"""
Miscellaneous
"""
CONFIG.MISC_include_baseline = True
CONFIG.MISC_use_gpu = False#True
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
CONFIG.MISC_save_vis_iter = 400 # How often we save a visualiation
CONFIG.MISC_vis_per_batch = 12
CONFIG.MISC_send_to_board = False # Should we send the logging to tensorboard
CONFIG.MISC_use_subset_of_data = None  # If set to None uses entire dataset. Otherwise only uses a subset of the images
CONFIG.MISC_use_fixed_patch = None # Use fixed image patches and not change them
CONFIG.MISC_data_aug = True
CONFIG.MISC_save_model_iter = 5000 # How often should we save the model weights
CONFIG.MISC_same_train_eval_set = False
CONFIG.MISC_project_root_path = os.path.dirname(__file__)
CONFIG.MISC_nbr_training_models = None

CONFIG.MISC_main_pid = os.getpid()


CONFIG.MISC_dataset_path = ""
hostname = socket.gethostname()
if hostname in ["john-UX430UA", "anton-Aspire-R5-471T", "dgxrise"]:
	CONFIG.MISC_dataset_path = "../../datasets/"
elif hostname == "rise-gpu0":
	CONFIG.MISC_dataset_path = "/home/datasets_thesis_aj/"
elif len(CONFIG.MISC_dataset_path) == 0:
	print("Unknown computer, set dataset path manually in config.py")
