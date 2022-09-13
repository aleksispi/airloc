import os
import random
import torch
from PIL import Image
import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import glob
from config import CONFIG
import argparse


class DubaiSeven(Dataset):

    def __init__(self, dataset_root_path = CONFIG.MISC_dataset_path , split = 'train' , transform = None , generate_new_split_file = False, use_eval_split = False):
        dataset_root_path = os.path.join(dataset_root_path , "masa_seven")

        # Check if we are to use the special eval_split_file

        if use_eval_split == 'basic':
            partition_file_path = os.path.join(dataset_root_path , "%s_eval_partition_grid_fixed_distance.csv" % split)
            split_frame = pd.read_csv(partition_file_path)
        elif use_eval_split == 'exhaustive':
            partition_file_path = os.path.join(dataset_root_path , "%s_eval_partition_grid_fixed_start.csv" % split)
            split_frame = pd.read_csv(partition_file_path)
            # Check so that saved patches matches with current settings
            #assert split_frame['patch_x'][0] == CONFIG.MISC_patch_size[0] , "

        else:
            # If grid game enabled load that file
            if CONFIG.MISC_grid_game:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition_grid.csv")
            else:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition.csv")

            # Check if it exists
            if not os.path.exists(partition_file_path):
                raise(FileNotFoundError("Split file not found:\t%s" % partition_file_path))
                exit(1)

            split_frame = pd.read_csv(partition_file_path)

            # Make correct split
            if split == 'train':
                split_frame = split_frame.loc[split_frame['partition'] == 0]
            elif split == 'val':
                split_frame = split_frame.loc[split_frame['partition'] == 1]
            elif split == 'test':
                split_frame = split_frame.loc[split_frame['partition'] == 2]
            else:
                print("Unknown split selected for Massachusetts:\t%s" % split)
                exit(1)

        self.image_list = split_frame['image_id'].tolist()
        self.start_crop = split_frame[['start_x', 'start_y']].to_numpy()
        self.goal_crop = split_frame[['goal_x' , 'goal_y']].to_numpy()

        # Allow for image preprocessing by transformer
        self.transform = transform
        if transform is None:
            self.tt = transforms.ToTensor()

        self.base_dir = os.path.join(dataset_root_path , "image")
        self.base_seg_dir = os.path.join(dataset_root_path , "label")

        # If we are to laod image segmentations aswell prepare a transform
        if CONFIG.RL_priv_use_seg:
            self.seg_transform = transforms.Compose([
                transforms.ToTensor(),
                ])

    def getitem__by_image_id(self, image_id):
        """
            Only used for debbugging.
        """
        ids , start = [] , 0
        try:
            while (True):
                id = self.image_list.index(image_id, start)
                ids.append(id)
                start = id + 1
        except:
            # Throws an error when no more availble
            pass
        if len(ids) == 0:
            raise(Exception("No such image"))

        image_path = os.path.join(self.base_dir, self.image_list[ids[0]])
        image = Image.open(image_path)

        locs = [self.__getitem__(id)[1] for id in ids]

        return image , np.stack(locs)

    def __getitem__(self, idx):
        image_path =  os.path.join(self.base_dir, self.image_list[idx])

        image = Image.open(image_path)


        if CONFIG.RL_priv_use_seg:
            # Load image segmentation
            seg_path = os.path.join(self.base_seg_dir , self.image_list[idx])
            seg = Image.open(seg_path)

            # Transform from tif image to a pytorch tensor
            seg = self.seg_transform(seg)[0,:,:][None,:]
            image = self.seg_transform(image)

            # Concatenate segmentation mask to end of crops
            image = torch.cat((image , seg ) , dim = 0)

        # If transformer available use it
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.tt(image)
            pass
        # Get start and goal crop
        start_crop = self.start_crop[idx, :]
        goal_crop = self.goal_crop[idx ,:]

        return (image , ( start_crop , goal_crop))

    def __len__(self):
        return len(self.image_list)

    def _get_image_and_file(self, idx):
        filepath = self.image_list[idx]
        img = Image.open(filepath)
        if self.transform is not None:
            img = self.transform(img)

        return (img , filepath )

class MasaFilt(Dataset):

    def __init__(self, dataset_root_path = CONFIG.MISC_dataset_path , split = 'train' , transform = None , generate_new_split_file = False, use_eval_split = False):
        dataset_root_path = os.path.join(dataset_root_path , "masa_filt")

        # Check if we are to use the special eval_split_file
        if use_eval_split == 'basic':
            partition_file_path = os.path.join(dataset_root_path , "%s_eval_partition_grid_fixed_distance.csv" % split)
            split_frame = pd.read_csv(partition_file_path)
        elif use_eval_split == 'exhaustive':
            partition_file_path = os.path.join(dataset_root_path , "%s_eval_partition_grid_fixed_start.csv" % split)
            split_frame = pd.read_csv(partition_file_path)

            # Check so that saved patches matches with current settings
            #assert split_frame['patch_x'][0] == CONFIG.MISC_patch_size[0] , "

        else:
            # If grid game enabled load that file
            if CONFIG.MISC_grid_game:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition_grid.csv")
            else:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition.csv")

            # Check if it exists
            if not os.path.exists(partition_file_path):
                raise(FileNotFoundError("Split file not found:\t%s" % partition_file_path))
                exit(1)

            split_frame = pd.read_csv(partition_file_path)

            # Make correct split
            if split == 'train':
                split_frame = split_frame.loc[split_frame['partition'] == 0]
            elif split == 'val':
                split_frame = split_frame.loc[split_frame['partition'] == 1]
            elif split == 'test':
                split_frame = split_frame.loc[split_frame['partition'] == 2]
            else:
                print("Unknown split selected for Massachusetts:\t%s" % split)
                exit(1)

        self.image_list = split_frame['image_id'].tolist()
        self.start_crop = split_frame[['start_x', 'start_y']].to_numpy()
        self.goal_crop = split_frame[['goal_x' , 'goal_y']].to_numpy()

        # Allow for image preprocessing by transformer
        self.transform = transform

        self.base_dir = os.path.join(dataset_root_path , "image")
        self.base_seg_dir = os.path.join(dataset_root_path , "label")

        # If we are to laod image segmentations aswell prepare a transform
        if CONFIG.RL_priv_use_seg:
            self.seg_transform = transforms.Compose([
                transforms.ToTensor(),
                ])

    def getitem__by_image_id(self, image_id):
        """
            Only used for debbugging.
        """
        ids , start = [] , 0
        try:
            while (True):
                id = self.image_list.index(image_id, start)
                ids.append(id)
                start = id + 1
        except:
            # Throws an error when no more availble
            pass
        if len(ids) == 0:
            raise(Exception("No such image"))

        image_path = os.path.join(self.base_dir, self.image_list[ids[0]])
        image = Image.open(image_path)

        locs = [self.__getitem__(id)[1] for id in ids]

        return image , np.stack(locs)

    def __getitem__(self, idx):
        image_path =  os.path.join(self.base_dir, self.image_list[idx])

        image = Image.open(image_path)


        if CONFIG.RL_priv_use_seg:
            # Load image segmentation
            seg_path = os.path.join(self.base_seg_dir , self.image_list[idx])
            seg = Image.open(seg_path)

            # Transform from tif image to a pytorch tensor
            seg = self.seg_transform(seg)[0,:,:][None,:]
            image = self.seg_transform(image)

            # TODO Fix split_images.py
            # Due to the labels containing some smoothed pixels we do a round to nereast integer here
            seg = torch.round(seg).float()

            # Concatenate segmentation mask to end of crops
            image = torch.cat((image , seg ) , dim = 0)

        # If transformer available use it
        if self.transform is not None:
            image = self.transform(image)

        # Round labels in the image mask. They are smoothed by the interpolation
        if CONFIG.RL_priv_use_seg:
            seg = image[-1,:,:]
            seg = torch.round(seg).float()
            image[-1,:,:] = seg

        # Get start and goal crop
        start_crop = self.start_crop[idx, :]
        goal_crop = self.goal_crop[idx ,:]

        return (image , ( start_crop , goal_crop))

    def __len__(self):
        return len(self.image_list)


    def _get_image_and_file(self, idx):
        filepath = self.image_list[idx]
        img = Image.open(filepath)
        if self.transform is not None:
            img = self.transform(img)

        return (img , filepath )

class MasaSeven(Dataset):

    def __init__(self, dataset_root_path = CONFIG.MISC_dataset_path , split = 'train' , transform = None , generate_new_split_file = False, use_eval_split = False):
        dataset_root_path = os.path.join(dataset_root_path , "masa_seven")

        # Check if we are to use the special eval_split_file

        if use_eval_split == 'basic':
            partition_file_path = os.path.join(dataset_root_path , "%s_eval_partition_grid_fixed_distance.csv" % split)
            split_frame = pd.read_csv(partition_file_path)
        elif use_eval_split == 'exhaustive':
            partition_file_path = os.path.join(dataset_root_path , "%s_eval_partition_grid_fixed_start.csv" % split)
            split_frame = pd.read_csv(partition_file_path)
            # Check so that saved patches matches with current settings
            #assert split_frame['patch_x'][0] == CONFIG.MISC_patch_size[0] , "

        else:
            # If grid game enabled load that file
            if CONFIG.MISC_grid_game:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition_grid.csv")
            else:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition.csv")

            # Check if it exists
            if not os.path.exists(partition_file_path):
                raise(FileNotFoundError("Split file not found:\t%s" % partition_file_path))
                exit(1)

            split_frame = pd.read_csv(partition_file_path)

            # Make correct split
            if split == 'train':
                split_frame = split_frame.loc[split_frame['partition'] == 0]
            elif split == 'val':
                split_frame = split_frame.loc[split_frame['partition'] == 1]
            elif split == 'test':
                split_frame = split_frame.loc[split_frame['partition'] == 2]
            else:
                print("Unknown split selected for Massachusetts:\t%s" % split)
                exit(1)

        self.image_list = split_frame['image_id'].tolist()
        self.start_crop = split_frame[['start_x', 'start_y']].to_numpy()
        self.goal_crop = split_frame[['goal_x' , 'goal_y']].to_numpy()

        # Allow for image preprocessing by transformer
        self.transform = transform
        if transform is None:
            self.tt = transforms.ToTensor()

        self.base_dir = os.path.join(dataset_root_path , "image")
        self.base_seg_dir = os.path.join(dataset_root_path , "label")

        # If we are to laod image segmentations aswell prepare a transform
        if CONFIG.RL_priv_use_seg:
            self.seg_transform = transforms.Compose([
                transforms.ToTensor(),
                ])

    def getitem__by_image_id(self, image_id):
        """
            Only used for debbugging.
        """
        ids , start = [] , 0
        try:
            while (True):
                id = self.image_list.index(image_id, start)
                ids.append(id)
                start = id + 1
        except:
            # Throws an error when no more availble
            pass
        if len(ids) == 0:
            raise(Exception("No such image"))

        image_path = os.path.join(self.base_dir, self.image_list[ids[0]])
        image = Image.open(image_path)

        locs = [self.__getitem__(id)[1] for id in ids]

        return image , np.stack(locs)

    def __getitem__(self, idx):
        image_path =  os.path.join(self.base_dir, self.image_list[idx])

        image = Image.open(image_path)


        if CONFIG.RL_priv_use_seg:
            # Load image segmentation
            seg_path = os.path.join(self.base_seg_dir , self.image_list[idx])
            seg = Image.open(seg_path)

            # Transform from tif image to a pytorch tensor
            seg = self.seg_transform(seg)[0,:,:][None,:]
            image = self.seg_transform(image)

            # Concatenate segmentation mask to end of crops
            image = torch.cat((image , seg ) , dim = 0)

        # If transformer available use it
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.tt(image)
            pass
        # Get start and goal crop
        start_crop = self.start_crop[idx, :]
        goal_crop = self.goal_crop[idx ,:]

        return (image , ( start_crop , goal_crop))

    def __len__(self):
        return len(self.image_list)

    def _get_image_and_file(self, idx):
        filepath = self.image_list[idx]
        img = Image.open(filepath)
        if self.transform is not None:
            img = self.transform(img)

        return (img , filepath )

class Masa(Dataset):

    def __init__(self, dataset_root_path = CONFIG.MISC_dataset_path , split = 'train' , transform = None , generate_new_split_file = False):
        dataset_root_path = os.path.join(dataset_root_path , "masa")

        # If grid game enabled laod that file
        if CONFIG.MISC_grid_game:
            partition_file_path = os.path.join(dataset_root_path , "list_eval_partition_grid.csv")
        else:
            partition_file_path = os.path.join(dataset_root_path , "list_eval_partition.csv")

        # Check if it exists
        if not os.path.exists(partition_file_path):
            raise(FileNotFoundError("Split file not found:\t%s" % partition_file_path))
            exit(1)

        split_frame = pd.read_csv(partition_file_path)

        # Make correct split
        if split == 'train':
            split_frame = split_frame.loc[split_frame['partition'] == 0]
        elif split == 'val':
            split_frame = split_frame.loc[split_frame['partition'] == 1]
        elif split == 'test':
            split_frame = split_frame.loc[split_frame['partition'] == 2]
        else:
            print("Unknown split selected for Massachusetts:\t%s" % split)
            exit(1)

        self.image_list = split_frame['image_id'].tolist()
        self.start_crop = split_frame[['start_x', 'start_y']].to_numpy()
        self.goal_crop = split_frame[['goal_x' , 'goal_y']].to_numpy()

        # Allow for image preprocessing by transformer
        self.transform = transform

        self.base_dir = os.path.join(dataset_root_path , "image")
        self.base_seg_dir = os.path.join(dataset_root_path , "label")

        # If we are to laod image segmentations aswell prepare a transform
        if CONFIG.RL_priv_use_seg:
            self.seg_transform = transforms.Compose([
                transforms.ToTensor(),
                ])

    def __getitem__(self, idx):
        image_path =  os.path.join(self.base_dir, self.image_list[idx])

        image = Image.open(image_path)


        if CONFIG.RL_priv_use_seg:
            # Load image segmentation
            seg_path = os.path.join(self.base_seg_dir , self.image_list[idx])
            seg = Image.open(seg_path)

            # Transform from tif image to a pytorch tensor
            seg = self.seg_transform(seg)[0,:,:][None,:]
            image = self.seg_transform(image)

            # Concatenate segmentation mask to end of crops
            image = torch.cat((image , seg ) , dim = 0)

        # If transformer available use it
        if self.transform is not None:
            image = self.transform(image)


        # Get start and goal crop
        start_crop = self.start_crop[idx, :]
        goal_crop = self.goal_crop[idx ,:]

        return (image , ( start_crop , goal_crop))

    def __len__(self):
        return len(self.image_list)


    def _get_image_and_file(self, idx):
        filepath = self.image_list[idx]
        img = Image.open(filepath)
        if self.transform is not None:
            img = self.transform(img)

        return (img , filepath )


class Dubai(Dataset):

    def __init__(self, dataset_root_path = CONFIG.MISC_dataset_path , split =
                 'train' , transform = None , generate_new_split_file = False,
                use_eval_split = False):
        dataset_root_path = os.path.join(dataset_root_path , "dubai")

        if use_eval_split == 'basic':
            partition_file_path = os.path.join(dataset_root_path , "%s_eval_partition_grid_fixed_distance.csv" % split)
            split_frame = pd.read_csv(partition_file_path)
        elif use_eval_split == 'exhaustive':
            partition_file_path = os.path.join(dataset_root_path , "%s_eval_partition_grid_fixed_start.csv" % split)
            split_frame = pd.read_csv(partition_file_path)

            # Check so that saved patches matches with current settings
            #assert split_frame['patch_x'][0] == CONFIG.MISC_patch_size[0] , "

        else:
            # If grid game enabled load that file
            if CONFIG.MISC_grid_game:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition_grid.csv")
            else:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition.csv")

            # Check if it exists
            if not os.path.exists(partition_file_path):
                raise(FileNotFoundError("Split file not found:\t%s" % partition_file_path))
                exit(1)

            split_frame = pd.read_csv(partition_file_path)

            # Make correct split
            if split == 'train':
                split_frame = split_frame.loc[split_frame['partition'] == 0]
            elif split == 'val':
                split_frame = split_frame.loc[split_frame['partition'] == 1]
            elif split == 'test':
                split_frame = split_frame.loc[split_frame['partition'] == 2]
            else:
                print("Unknown split selected for Massachusetts:\t%s" % split)
                exit(1)

        self.image_list = split_frame['image_id'].tolist()
        self.start_crop = split_frame[['start_x', 'start_y']].to_numpy()
        self.goal_crop = split_frame[['goal_x' , 'goal_y']].to_numpy()

        # Allow for image preprocessing by transformer
        self.transform = transform

        self.base_dir = os.path.join(dataset_root_path , "image")
        self.base_seg_dir = os.path.join(dataset_root_path , "label")

        # If we are to laod image segmentations aswell prepare a transform
        if CONFIG.RL_priv_use_seg:
            self.seg_transform = transforms.Compose([
                transforms.ToTensor(),
                ])

    def __getitem__(self, idx):
        image_path =  os.path.join(self.base_dir, self.image_list[idx])

        image = Image.open(image_path)


        if CONFIG.RL_priv_use_seg:
            # Load image segmentation
            seg_path = os.path.join(self.base_seg_dir , self.image_list[idx])
            seg = Image.open(seg_path)

            # Transform from tif image to a pytorch tensor
            seg = self.seg_transform(seg)
            image = self.seg_transform(image)

            # Concatenate segmentation mask to end of crops
            image = torch.cat((image , seg ) , dim = 0)

        # If transformer available use it
        if self.transform is not None:
            image = self.transform(image)


        # Get start and goal crop
        start_crop = self.start_crop[idx, :]
        goal_crop = self.goal_crop[idx ,:]

        return (image , ( start_crop , goal_crop))


    def __len__(self):
        return len(self.image_list)


    def _get_image_and_file(self, idx):
        filepath = self.image_list[idx]
        img = Image.open(filepath)
        if self.transform is not None:
            img = self.transform(img)

        return (img , filepath )



# Hmm maybe should inherit ImageFolder... but decided not to
class CustomDataset():

    def __init__(self, datasets_root_path , custom_dataset_name,split = 'train' , transform = None, custom_split_file = None):
        dataset_root_path = os.path.join(datasets_root_path , "Custom", custom_dataset_name)

        if not os.path.exists(datasets_root_path):
            os.makedirs(datasets_root_path)

        # Try to locate this specific custom dataset
        if not os.path.exists(dataset_root_path):
            print("Unable to find this dataset:\t%s" % custom_dataset_name)
            exit(1)

        if custom_split_file is None:
            # No custom split file selected use standard split file
            # If grid game enabled laod that file
            if CONFIG.MISC_grid_game:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition_grid.csv")
            else:
                partition_file_path = os.path.join(dataset_root_path , "list_eval_partition.csv")
        else:
            # Use custom split file
            partition_file_path = os.path.join(dataset_root_path , custom_split_file)

        # Check if it exists
        if not os.path.exists(partition_file_path):
            raise(FileNotFoundError("Split file not found:\t%s" % partition_file_path))
            exit(1)

        split_frame = pd.read_csv(partition_file_path)

        # Make correct split
        if split == 'train':
            split_frame = split_frame.loc[split_frame['partition'] == 0]
        elif split == 'val':
            split_frame = split_frame.loc[split_frame['partition'] == 1]
        elif split == 'test':
            split_frame = split_frame.loc[split_frame['partition'] == 2]
        else:
            print("Unknown split selected:\t%s" % split)
            exit(1)

        self.image_list = split_frame['image_id'].tolist()
        self.start_crop = split_frame[['start_x', 'start_y']].to_numpy()
        self.goal_crop = split_frame[['goal_x' , 'goal_y']].to_numpy()

        # Allow for image preprocessing by transformer
        self.transform = transform

        self.base_dir = os.path.join(dataset_root_path , "image" )

    def __getitem__(self, idx):
        image_path = os.path.join(self.base_dir , self.image_list[idx] )
        image = Image.open(image_path)

        # If transformer available use it
        if self.transform is not None:
            image = self.transform(image)

        # Get start and goal crop
        start_crop = self.start_crop[idx, :]
        goal_crop = self.goal_crop[idx ,:]

        return (image , (start_crop , goal_crop))

    def __len__(self):
        return len(self.image_list)


    def _get_image_and_file(self, idx):
        filepath = os.path.join(self.base_dir ,self.image_list[idx])
        img = Image.open(filepath)

        if self.transform is not None:
            img = self.transform(img)
        return (img ,filepath )

class MasaFull(Dataset):

    def __init__(self, datasets_root_path , split = 'train' , transform = None , randomRotation = False ):
        dataset_root_path = os.path.join(datasets_root_path , "masa_full")

        self.randomRotation = randomRotation

        if not os.path.exists(datasets_root_path):
            os.makedirs(datasets_root_path)

        # Also pngs are available
        image_folder_path = os.path.join(dataset_root_path, "tiff")

        # Dataset is already split no need for split file
        if split == 'train':
            image_folder_path = os.path.join(image_folder_path , "train")
        elif split == 'val':
            image_folder_path = os.path.join(image_folder_path , "val")
        elif split == 'test':
            image_folder_path = os.path.join(image_folder_path , "test")
        else:
            raise(Exception("Unknown split:\t%s" % split))

        # File names
        self.image_list = [os.path.join(image_folder_path , x) for x in os.listdir(image_folder_path)]

        # Random Crop
        self.rc = transforms.RandomCrop(size = [500 , 500])

        # from PIL to tensor transform
        self.tt = transforms.ToTensor()

        self.transform = transform

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(self.image_list[idx])

        # Make it tensor
        #image = self.tt(image)

        # If random rotation is enabled. do it
        if self.randomRotation:
            #angle = torch.rand(size = (1,)).item() * 90.0
            angle = torch.randint( low = -180 , high = 180 , size = (1,)).float().item()
            image = F.rotate(image , angle , fill = None)

            # Get center crop dimensions
            while ( angle < 0 or angle > 90):
                angle = angle + 90 if angle < 0 else angle
                angle = angle - 90 if angle > 90 else angle


            size = int( image.size[-1] / (np.cos(np.radians(angle)) + np.sin(np.radians(angle))) )
            image = F.center_crop(image , [size , size])

        # Now do random 500x500 crop
        image = self.rc(image)

        # Do regular data augmentation if available
        if self.transform:
            image = self.transform(image)
        else:
            image = self.tt(image)

        # No start and goal crops. Only for show
        return (image, ([],[]))

    def __len__(self):
        return len(self.image_list)

def _generate_split_file( split_file_path , grid_game = False):
    # Generates a separate csv file for the split with crop locations
    # Useful for creating a static validation set (or training if you want)

    # Read partition file
    # list_eval_partition: 0: training, 1: validation, 2:test
    try:
        dataframe = pd.read_csv(split_file_path)
    except FileNotFoundError:
        raise(FileNotFoundError("Eval partion file not found:\t%s" % split_file_path ))

    # Generate start and goal crops, insert into dataframe and write back to file
    N = dataframe.shape[0]

    H_img , W_img = CONFIG.MISC_im_size
    H_patch , W_patch = CONFIG.MISC_patch_size

    # To make sure crops always are in same location use separate random number generator with hardcoded seed
    num_gen = np.random.default_rng(0)

    # If the grid_game flag is enabled divide image into uniform grid and sample patch location from that gird
    if grid_game:
        grid_H , grid_W = int(H_img / H_patch)  , int(W_img / W_patch)
        grid_locs = np.floor( np.random.uniform( low = [0,0, 0,0] , high = [grid_H , grid_W, grid_H ,grid_W] , size = (N, 4)))

        # TODO - Add so that start and goal does not get on same patch
        #number_same_patch = ((grid_locs[:,0] == grid_locs[:,2]) & (grid_locs[:,1] == grid_locs[:,3])).sum()

        # Convert from grid location to pixel location
        crop_locs = (grid_locs * np.array(2*CONFIG.MISC_patch_size)).astype('int64')
        #crop_locs = np.concatenate( (start_crop_loc , top_left_loc + np.array(CONFIG.MISC_patch_size)) ,axis = 1).astype('int64')

        # Also alter split file name and append 'grid'
        dirname = os.path.dirname(split_file_path)
        filename = os.path.basename(split_file_path)
        temp = filename.split('.') # Assumes only single dot in filename
        new_filename = temp[0] + '_grid.' + temp[1]
        split_file_path = os.path.join(dirname ,new_filename)


    else:

        # Else just sample any location in the image
        # ( start_x , start_y , goal_x , goal_y)
        crop_locs = num_gen.integers((0,0,0,0) , (H_img-H_patch , W_img-W_patch , H_img-H_patch , W_img - W_patch) , size = (N , 4))

    # Insert into dataframe
    dataframe['start_x'] = crop_locs[:,0]
    dataframe['start_y'] = crop_locs[:,1]
    dataframe['goal_x'] = crop_locs[:,2]
    dataframe['goal_y'] = crop_locs[:,3]


    # Write back!
    dataframe.to_csv(split_file_path, index = False)

def _create_base_split_file( datasets_root_path = None, dataset = None, relative_image_path = None):

    datasets_root_path = datasets_root_path if datasets_root_path is not None else CONFIG.MISC_dataset_path
    dataset = dataset if dataset is not None else CONFIG.MISC_dataset
    relative_image_path = relative_image_path if relative_image_path is not None else "image/*"

    dataset_root_path = os.path.join(datasets_root_path , dataset)
    image_list = glob.glob(os.path.join(dataset_root_path,relative_image_path))

    # Make sure only to keep relative name
    image_list = list(map(os.path.basename , image_list))

    nbr_img = len(image_list)
    partition  = np.zeros(nbr_img)
    partition[int(nbr_img*0.7):int(nbr_img*0.85)] += 1
    partition[int(nbr_img*0.85):] +=2
    random.shuffle(partition)
    data = {"image_id":image_list,
            "partition":partition}
    dataframe = pd.DataFrame(data)
    split_file_path = os.path.join(dataset_root_path,'list_eval_partition.csv')
    dataframe.to_csv(split_file_path,index = False)

if __name__ == '__main__':
    # This part is used to be able to simply generate split files for datasets

    # Set random seed
    random.seed(CONFIG.MISC_random_seed)
    np.random.seed(CONFIG.MISC_random_seed)
    torch.manual_seed(CONFIG.MISC_random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset" , "-d", type = str, help = "Select dataset")
    parser.add_argument("--grid-game" ,"-g" , action = 'store_true', default = False)


    # DEBUGGING
    parser.add_argument("--debug-masa-full" , action = 'store_true' , default = False)
    parser.add_argument("--random-rotation" , action = 'store_true' , default = False)

    args = parser.parse_args()


    if args.debug_masa_full:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('TkAgg')

        torch.manual_seed(0)
        d_rot = MasaFull(".." , randomRotation = True)
        d_no_rot = MasaFull(".." , randomRotation = False)

        for i in range(10):
            image_rot = d_rot.__getitem__(i)
            image_no_rot = d_no_rot.__getitem__(i)

            fig , axes = plt.subplots(1,2)

            axes[0].imshow(image_rot.permute(1,2,0))
            axes[1].imshow(image_no_rot.permute(1,2,0))

            plt.show()
            plt.close('all')


    known_datasets = ['dubai','masa','masa_filt','masa_seven','dubai_seven']

    if not args.dataset is None:
        # Check if custom or regular dataset
        if (args.dataset.startswith('custom_')):
            # Check that custom dataset actually existss
            if not os.path.exists(os.path.join(CONFIG.MISC_dataset_path , "Custom", args.dataset[7:])):
                print("Dataset does not exists:\t%s" % os.path.join("Custom", args.dataset[7:]))
                exit(1)
            _create_base_split_file(dataset = os.path.join("Custom", args.dataset[7:])  )
            split_file_path = os.path.join(CONFIG.MISC_dataset_path, "Custom", args.dataset[7:], "list_eval_partition.csv")
        elif args.dataset in known_datasets:
            # Regular dataset
            _create_base_split_file(dataset = args.dataset)
            split_file_path = os.path.join(CONFIG.MISC_dataset_path , args.dataset , "list_eval_partition.csv")
        else:
            print("No dataset is found")
            exit(1)
    else:
        print("Not a valid dataset!")
        exit(1)
    # then generate split file with random start and goal crops
    _generate_split_file(split_file_path , args.grid_game)
