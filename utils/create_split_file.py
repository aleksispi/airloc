
import pandas as pd
import os
import matplotlib
import argparse
import torch
from utils.dataset_utils import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import numpy as np

from doerchnet.utils import visualize_doerch

from utils.utils import sample_grid_games_fixed_distance

from config import CONFIG

def create_fixed_game_split_file(args ):

    if args.dataset is None:
        print("\nERROR:\tPlease select a dataset\n")
        exit(1)

    # Check if dataset exists
    dataset_path = os.path.join(CONFIG.MISC_dataset_path , args.dataset)

    if not os.path.exists(dataset_path):
        print("\nERROR:\tDataset not found:\t%s\n" % args.dataset)
        exit(1)

    # Locate current split file
    if args.grid_game:
        split_file_path = os.path.join(dataset_path , "list_eval_partition_grid.csv")
    else:
        split_file_path = os.path.join(dataset_path, "list_eval_partition.csv")

    # Check its existence
    if not os.path.exists(split_file_path):
        print("Current %s split file not found." % str(args.grid_game))
        exit(2)

    # Read split file and filter out for this split
    orig_split = pd.read_csv(split_file_path)

    if args.split == 'train':
        orig_split = orig_split[orig_split.partition == 0.0]
    elif args.split == 'val':
        orig_split = orig_split[orig_split.partition == 1.0]
    elif args.split == 'test':
        orig_split = orig_split[orig_split.partition == 2.0]
    else:
        raise(Exception("Unexpected split:\t%s"  % args.split))

    # For each file in the dataframe create args.number_games games and append to the new split

    # First two are start coordinates and last two are goal coordinates
    n_imgs = orig_split.shape[0]
    games = np.zeros((orig_split.shape[0] * args.number_games , 4))

    locs_start = np.array([[0,4]]).repeat(orig_split.shape[0],axis=0)
    for i in range(1,args.number_games + 1):
        if args.number_games == 24:
            idx = i
            idx-=1
            if idx >=20: idx+=1
            y,x = divmod(idx,5)
            print(x,y)
            locs_goal = np.array([[x,y]]).repeat(orig_split.shape[0],axis=0)
            temp = np.hstack((locs_start,locs_goal))
        else:
            if i > CONFIG.MISC_game_size - 1:
                difficulty = i % (CONFIG.MISC_game_size - 1)
            else:
                difficulty = i
            print(difficulty)
            temp = sample_grid_games_fixed_distance(difficulty , n_imgs)
        games[(i-1)*n_imgs:i*n_imgs,:] = temp


    # Now games have been sampled. Multiply by patch size to get coordinates and write to file
    games *= CONFIG.MISC_step_sz

    file_names = np.expand_dims(np.array(list(orig_split['image_id']) * args.number_games) , axis = 1)

    data = np.hstack( (file_names, games, np.ones((orig_split.shape[0] * args.number_games, 2)) * CONFIG.MISC_patch_size))
    cols = ["image_id" , "start_y", "start_x" , "goal_y", "goal_x" , "patch_size_x" , "patch_size_y"]

    new_split = pd.DataFrame(data , columns = cols)

    if args.number_games == 24:
        # Write to file
        new_split_file_path = os.path.join(dataset_path , "%s_eval_partition_grid_fixed_start.csv" % args.split)
        new_split.to_csv(new_split_file_path)
    else:

        # Write to file
        new_split_file_path = os.path.join(dataset_path , "%s_eval_partition_grid_fixed_distance.csv" % args.split)
        new_split.to_csv(new_split_file_path)

def verify_split_file(args):
    matplotlib.use('TkAgg')
    transform = transforms.Compose([
        transforms.Resize(CONFIG.MISC_im_size),
        transforms.ToTensor()
        ])
    # Load dataset

    if args.dataset == 'masa_filt':
        dataset = MasaFilt(CONFIG.MISC_dataset_path,split = 'val', use_eval_split = True)
    else:
        raise(Exception("Unkown dataset"))

    image , locs = dataset.getitem__by_image_id(args.image_id)

    # Convert image to tensor
    image = transform(image)
    locs = np.concatenate((locs, 48 * np.ones((4,2,2))) , axis = 2)

    locs = torch.from_numpy(locs)

    for i in range(locs.shape[0]):
        visualize_doerch(image, locs[i,0,:] , locs[i,1,:] ,torch.tensor([1,0]), save_name = 'vis_%d'%i)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset" , "-d" , type=str , default = None, help = "Select which dataset to create split file for.")
    parser.add_argument("--split" ,  type = str , default = "val" , help = "Which split should be created.")
    parser.add_argument("--grid-game" , type = bool, default = True)
    parser.add_argument("--number-games", type = int, default = CONFIG.MISC_game_size - 1 , help = "Number of games to create for each image.")

    parser.add_argument("--verify", type = bool, default = False, help = "Verify new split file visually.")
    parser.add_argument("--image-id" , type = str, help = "Which image to visualize")


    args = parser.parse_args()

    if not args.verify:

        print("Creating eval split file for %s" % args.dataset)
        print("Partition selected:\t%s" % args.split)
        print("Number of games per image:\t%d"  % args.number_games)

        create_fixed_game_split_file(args)
    else:
        print("Verifying current eval split file.")
        verify_split_file(args)
