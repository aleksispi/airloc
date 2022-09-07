#!/bin/env python3


import numpy as np
import random
import pandas as pd
from shutil import copyfile

import argparse
import os, sys

from config import CONFIG

argparse = argparse.ArgumentParser()

argparse.add_argument("--seed" , type = int , default = 0 , help = "Set seed for random number generators")
argparse.add_argument("dataset" , type = str , help = "Select which dataset to use")
argparse.add_argument("--number-games-diff" , "-n" , type = int , default = 4, help = "Select the number of games per difficulty")

args = argparse.parse_args()


# Set seeds
np.random.seed(args.seed)
random.seed(args.seed)


# Check that dataset is available
dataset_path = os.path.join(CONFIG.MISC_dataset_path, args.dataset)
if not os.path.exists(dataset_path):
    print("Dataset does not exist:\t%s" % args.dataset)
    sys.exit(0)


# Check that it has a split file with fixed distance
if not os.path.exists(os.path.join(dataset_path, "val_eval_partition_grid_fixed_distance.csv")):
    print("Dataset does not have 'val_eval_partition_grid_fixed_distance.csv'.")
    sys.exit(0)

data_path = os.path.join(dataset_path , "image")


# Read split file
split = pd.read_csv(os.path.join(dataset_path, "val_eval_partition_grid_fixed_distance.csv"))


# Create map structure to save game splits
os.makedirs("game_files", exist_ok = True)

i = 1
game_split = "game_split_%d" % i
while (os.path.exists(os.path.join("game_files" , game_split))):
    i += 1
    game_split = "game_split_%d" % i

game_split_dir = os.path.join("game_files" , game_split)
os.makedirs(game_split_dir)

n_rows = split.shape[0]
n_images = n_rows // 4

# Divide the dataset into n number of divisions
divs = n_images // ( args.number_games_diff * 4) 

for div in range(divs):
    for i in range(4):
        # select rows in split file
        offset = [x % 4 for x in range(i, i + 4)]

        base = div * args.number_games_diff 
        indx = []
        for p in range(4):
            indx += list(range(base + offset[p] * args.number_games_diff + p * n_images, base + offset[p] *  args.number_games_diff + p * n_images + args.number_games_diff))

        try:
            local_split = split.iloc[indx]
        except:
            print("Unable to split with iloc:\t%s"%div)
            pass

        # Also create warmup folder with some randomly sampled images
        warmup_indx = []
        for loop in range(10): # Use ten warmup images
            warm_ind = random.randint(0, split.shape[0])
            while warm_ind in indx:
                warm_ind = random.randint(0, split.shape[0])
            warmup_indx.append(warm_ind)

        warm_split = split.iloc[warmup_indx]

        local_dir = os.path.join(game_split_dir, "game_split_%d_%d" % (div, i))

        local_data_dir = os.path.join(local_dir,  "images")
        local_warmup_data_dir = os.path.join(local_dir, "warmup_images")

        # Create folder for split
        os.makedirs(local_data_dir)
        os.makedirs(local_warmup_data_dir)
       
        # Write split file 
        local_split.to_csv(os.path.join(local_dir, "split_file.csv"))
        warm_split.to_csv(os.path.join(local_dir, "warmup_split_file.csv"))

        # Move images to local data dir
        for image_id in local_split.image_id:
            copyfile(os.path.join(data_path, image_id) , os.path.join(local_data_dir, image_id))            
        
        for image_id in warm_split.image_id:
            copyfile(os.path.join(data_path, image_id) , os.path.join(local_warmup_data_dir, image_id))            








