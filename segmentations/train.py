
import os
import torch
import random 
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import json
import numpy as np

from torchinfo import summary

from shutil import copyfile
from utils.utils import get_random_crops, get_deterministic_crops, load_normalize_data

from segmentations.utils import compute_loss, format_labels_for_loss, visualize_segmentation_est

from segmentations.utils import DiceLoss

from segmentations.networks import BakeNet
from segmentations.u_net import UNet

import torch.optim as optim

import numpy as np

import matplotlib 
import matplotlib.pyplot as plt

from utils.stat_collector import StatCollector

from config import CONFIG

# Set all hyperparameters here to not clutter the config file 
try:
    netType = 'UNet'

    seed = 0
    batch_size = 64
    epochs = 10000
    multiply_images = 2

    optimizer_type = 'adam'
    learning_rate = 1e-4
    lower_learning_rate_factor = 0.33
    momentum = 0.95
    beta1 = 0.9
    beta2 = 0.999

    print_iter = 100
    vis_iter = 125

    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not CONFIG.RL_priv_use_seg:
        raise(Exception("ERROR: segmentation information not enabled!"))

    info = dict([
        ("NetType" , netType),
        ("Completed" , False),
        ("Metrics", [
            "Loss",
            "ValLoss",
            "Accuracy",
            "ValAccuracy"
            ]),
        ("LogDir" , None),
        ("Blocks" , [1 ,1 , 1]),
        ("NbrParameters" , 0),
        ("LatentSpaceSize" , 0),
        ("NbrImagesInTrain", 0),
        ("NbrImagesInVal" , 0)
        ])

    metrics = info['Metrics']

    # Function used to update leanring rate
    def update_learning_rate(optimizer , learning_rate):
        for params in opimizier.param_groups:
            params['lr'] = learning_rate


    # Transform all input images
    loadTransform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(CONFIG.MISC_patch_size),
            transforms.Normalize((0.5,0.5,0.5) , (0.5,0.5,0.5))
        ])


    # Find device
    device = torch.device("cuda:0" if CONFIG.MISC_use_gpu and torch.cuda.is_available() else "cpu")
    CONFIG.device = device

    # Load data
    trainloader,valloader = load_normalize_data(download = False, batch_size = batch_size , multiply_images = multiply_images)
    valiter = iter(valloader)

    info['NbrImagesInTrain'] = len(trainloader) * batch_size
    info['NbrImagesInVal'] = len(valloader) * batch_size

    tot_itr = 0
    tot_nbr_itr = epochs * len(trainloader)

    if netType == 'BakeNet':
        net = BakeNet()
    elif netType == 'UNet':
        net = UNet( n_channels = 3, n_classes = 2)
    else:
        raise(Exception("Unkown network type"))

    # Count number of paramters
    net_parameters = filter(lambda p: p.requires_grad, net.parameters())
    info['NbrParameters'] = int(sum([np.prod(p.size()) for p in net_parameters]))

    info['LatentSpaceSize'] = net.get_latent_space_size()

    # Set optimizer
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(net.parameters() , lr = learning_rate, momentum = momentum)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(net.parameters() , lr = learning_rate, betas =(beta1 , beta2))
    else:
        raise(Exception("Unknown optimizer type:\t%s" % optimizer_type))


    # Select loss function
    class_weights = [1,3]
    #criterion = nn.MSELoss()
    criterion = DiceLoss(class_weights)

    # Setup stat collector
    log_dir_name = os.path.basename(CONFIG.STATS_log_dir) 
    CONFIG.STATS_SEG_log_dir = os.path.join(CONFIG.MISC_project_root_path, "segmentations", "logs", log_dir_name)
    os.makedirs(CONFIG.STATS_SEG_log_dir)

    sc = StatCollector(CONFIG.STATS_SEG_log_dir, tot_nbr_itr, print_iter = print_iter) 
    for metric in metrics:
        sc.register(metric , {'type':'avg','freq':'step'})

    # Enter log dir and write to file
    info['LogDir'] = CONFIG.STATS_SEG_log_dir
    with open(os.path.join(CONFIG.STATS_SEG_log_dir, "info.json") , 'w') as io:
        json.dump(info, io , indent = 4)

    # Construct noise for network summary
    noise = torch.randn(1,3, CONFIG.MISC_patch_size[0] , CONFIG.MISC_patch_size[1])

    # Print model summary to separate file
    with open(os.path.join(CONFIG.STATS_SEG_log_dir , "network_summary") , 'w') as io:
        print(summary(net, input_data = noise, verbose=0), file = io)

    # Save all fileus
    copyfile("segmentations/train.py" , os.path.join(CONFIG.STATS_SEG_log_dir, "train.py"))
    copyfile("config.py" , os.path.join(CONFIG.STATS_SEG_log_dir, "config.py"))
    copyfile("segmentations/networks.py" , os.path.join(CONFIG.STATS_SEG_log_dir, "networks.py"))
    copyfile("segmentations/u_net.py" , os.path.join(CONFIG.STATS_SEG_log_dir, "u_net.py"))
    copyfile("segmentations/utils.py" , os.path.join(CONFIG.STATS_SEG_log_dir, "utils.py"))

    if CONFIG.MISC_dataset.startswith('custom_'):
        stat_path = os.path.join(CONFIG.MISC_dataset_path,"Custom", CONFIG.MISC_dataset[7:],'stats.json')
    else:
        stat_path = os.path.join(CONFIG.MISC_dataset_path,CONFIG.MISC_dataset,'stats.json')

    with open(stat_path) as json_file:
        stats = json.load(json_file)
        
    dataset_means = torch.tensor(stats['means'][:3])
    dataset_stds = torch.tensor(stats['stds'][:3])
    unNormImage = transforms.Normalize( ( - dataset_means / dataset_stds).tolist() , (1.0 / dataset_stds).tolist() )

    # Decrease learning rate every 20 epoch
    current_learning_rate = learning_rate

    def update_learning_rate(optimizer , learning_rate):
        for params in optimizer.param_groups:
            params['lr'] = learning_rate

    net = net.to(device)

    for epoch in range(epochs):

        for (i,data) in enumerate(trainloader):

            batch_images , (start_crops_ , goal_crops_) = data
            batch_images = batch_images

            # Get random crops
            crops , _ = get_random_crops(batch_images)
           
            # The input is the RGB image
            inpt = crops[:,0:3,:,:].to(device)

            # The ground truth are the segmentation mask
            labels = crops[:,3:,:,:].to(device)

            # Labels need to be transformed to correct format 
            labels = format_labels_for_loss(labels)

            outputs = net(inpt)

            loss = criterion(outputs, labels)
            #loss = compute_loss(outputs , labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            sc.s('Loss').collect(loss.item()) 

            # Calculate and logg pixelwise accuracy
            acc = (torch.argmax(labels, dim = 1) == torch.argmax(outputs, dim = 1)).float().mean()
            sc.s('Accuracy').collect(acc.item())

            with torch.no_grad():
                # Run one batch on the valloader aswell
                try:
                    val_images, (start_crops_ , goal_crops_) = next(valiter)
                except:
                    valiter = iter(valloader)
                    val_images, ( start_crops_ , goal_crops_) = next(valiter)
                val_images = val_images.to(device)

                # Due to how the dataloading is devised the valloader has the same multiply images
                # as the trainloader. During validation fixed crops are used therefore 
                # there is no need to run same image with same patch multiple times
                val_images = val_images[0:-1:multiply_images ,:]

                val_crops , _ =  get_deterministic_crops(val_images, coords = start_crops_)
                
                val_inpt = val_crops[:,0:3,:,:].to(device)
                val_labels = val_crops[:,3:,:,:].to(device) # new added extra channel for image boundaries

                val_labels = format_labels_for_loss(val_labels)

                val_outs = net(val_inpt).to(device)

                val_loss = criterion(val_outs, val_labels)

                # Logging
                sc.s('ValLoss').collect(val_loss.item())
                val_acc = (torch.argmax(val_labels, dim = 1) == torch.argmax(val_outs, dim = 1)).float().mean()
                sc.s('ValAccuracy').collect(val_acc.item())

            if tot_itr % print_iter == 0 or tot_itr == tot_nbr_itr - 1:
                print("Iteration:\t%d / %d" % ( tot_itr, tot_nbr_itr))
                sc.print()
                sc.save()

            # Do visualize
            if tot_itr % vis_iter == 0 or tot_itr == tot_nbr_itr - 1:
                visualize_segmentation_est(val_labels , val_outs , vis_name = "vis_%d" % tot_itr)
                
            tot_itr += 1

        # Lower the learning rate
        if (epoch + 1) % 10 == 0 and False:
            current_learning_rate *= lower_learning_rate_factor
            update_learning_rate(optimizer, current_learning_rate)

    info['Completed'] = True

    with open(os.path.join(CONFIG.STATS_SEG_log_dir, "info.json") , 'w') as io:
        json.dump(info , io , indent = 4)

finally: 
    while(True): 
        user = input("\nSave model [yes/no]\n") 
        if user == 'y':
            # Save the encoder and the decoder
            torch.save(net.state_dict() , os.path.join(CONFIG.STATS_SEG_log_dir, "final_unet"))
            print("Model Saved")
            break
        elif user == 'n':
            print("Model Not Saved")
            break

print("\nTraining Completed!\n")


