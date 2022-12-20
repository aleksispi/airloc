import os
import torch
import random 
import torch.nn.functional as F
import torchvision.transforms as transforms 
import torch.nn as nn
import json
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

from shutil import copyfile
from utils.utils import get_random_crops, get_deterministic_crops, load_normalize_data 
                        

from doerchnet.utils import sample_doerch_crops , visualize_doerch, get_label, visualize_batch_doerch, calculate_precision
from doerchnet.networks import AJNet
from doerchnet.share_net import ShareNet

import torch.optim as optim

from utils.stat_collector import StatCollector

from config import CONFIG




try:
    # Set all hyperparameters here to not clutter the config file 
    net_type = 'ShareNet'

    seed = 0
    batch_size = 64
    epochs = 100000
    multiply_images = 4
    # 8 dim -> local estimation of neibouring patches
    # 25 dim -> global guess on where target is
    # 80 dim -> Global relative to start 
    dim = 8 
    max_dist = 1

    optimizer_type = 'adam'
    learning_rate = 1e-4
    lower_learning_rate_factor = 0.33
    momentum = 0.95
    beta1 = 0.9
    beta2 = 0.999

    print_iter = 100
    vis_iter = 1000

    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    info = dict([
        ("NetType" , net_type),
        ("Completed" , False),
        ("Metrics", [
            "Loss",
            "ValLoss",
            "Accuracy",
            "ValAccuracy",
            "ActionsTaken",
            "ValActionsTaken",
            "CorrectActions",
            "ValCorrectActions",
            "AccuracyCorners",
            "AccuracyBoundaries",
            "AccuracyMiddle",
            "ValAccuracyCorners",
            "ValAccuracyBoundaries",
            "ValAccuracyMiddle",
            ]),
        ("LogDir" , None),
        ("Blocks" , [2 ,2 , 2]),
        ("NbrParameters" , 0),
        ("LatentSpaceSize" , 0)
        ])


    if dim not in [25 , 80]:
        # Remove accuracy region metrics
        for m in ["AccuracyCorners" , "AccuracyBoundaries" , "AccuracyMiddle"]:
            info['Metrics'].remove(m)
            info['Metrics'].remove("Val" + m)

    metrics = info['Metrics']

    # Function used to update leanring rate
    def update_learning_rate(optimizer , learning_rate):
        for params in opimizier.param_groups:
            params['lr'] = learning_rate

    # Find device
    device = torch.device("cuda:0" if CONFIG.MISC_use_gpu and torch.cuda.is_available() else "cpu")
    CONFIG.device = device

    # Load data
    trainloader,valloader = load_normalize_data(download = False, batch_size = batch_size , multiply_images = multiply_images)
    print(f"Trainloader: {len(trainloader)}")
    print(f"Valloader: {len(valloader)}")
    
    # Save information about dataloaders
    info['Dataset'] = CONFIG.MISC_dataset
    info['ValLoaderLength'] = len(valloader)
    info['TrainLoaderLength'] = len(trainloader)

    valiter = iter(valloader)

    tot_itr = 0
    tot_nbr_itr = epochs * len(trainloader)

    if net_type == 'ShareNet':
        net = ShareNet(num_out_classes = dim)
    else:
        net = AJNet(net_type, num_classes=dim, both_branches = True)

    # Count number of paramters
    net_parameters = filter(lambda p: p.requires_grad, net.parameters())
    info['NbrParameters'] = int(sum([np.prod(p.size()) for p in net_parameters]))

    # Record latentspace size in info file
    noise = torch.randn(1, net.n_chan , CONFIG.MISC_patch_size[0] , CONFIG.MISC_patch_size[1]).cpu()
    latent , _ = net(noise,noise)
    latentSpaceSize = int(np.prod(list(latent.shape)))
    info['LatentSpaceSize'] = latentSpaceSize
    info['InputSpaceSize'] = int(np.prod(list(noise.shape)))

    net = net.to(device)

    # Set optimizer
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(net.parameters() , lr = learning_rate, momentum = momentum)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(net.parameters() , lr = learning_rate, betas =(beta1 , beta2))
    else:
        raise(Exception("Unknown optimizer type:\t%s" % optimizer_type))


    # Select loss function
    criterion = nn.CrossEntropyLoss()

    # Setup stat collector
    log_dir_name = os.path.basename(CONFIG.STATS_log_dir) 
    CONFIG.STATS_DORCH_log_dir = os.path.join(CONFIG.MISC_project_root_path, "doerchnet", "logs", log_dir_name)
    os.makedirs(CONFIG.STATS_DORCH_log_dir)
    exclude_prints = [
                    "ActionsTaken",
                    "ValActionsTaken",
                    "CorrectActions",
                    "ValCorrectActions",
                     ] # Does not print these statistics


    CONFIG.STATS_DOERCH_vis_dir = os.path.join(CONFIG.STATS_DORCH_log_dir, "visualizations")
    os.makedirs(CONFIG.STATS_DOERCH_vis_dir)

    sc = StatCollector(CONFIG.STATS_DORCH_log_dir, tot_nbr_itr, print_iter = print_iter, exclude_prints = exclude_prints) 
    for metric in metrics:
        sc.register(metric , {'type':'avg','freq':'step'})

    # Enter log dir and write to file
    info['LogDir'] = CONFIG.STATS_DORCH_log_dir
    with open(os.path.join(CONFIG.STATS_DORCH_log_dir, "info.json") , 'w') as io:
        json.dump(info, io , indent = 4)

    # Save all files
    copyfile("doerchnet/train.py" , os.path.join(CONFIG.STATS_DORCH_log_dir, "train.py"))
    copyfile("config.py" , os.path.join(CONFIG.STATS_DORCH_log_dir, "config.py"))
    copyfile("doerchnet/networks.py" , os.path.join(CONFIG.STATS_DORCH_log_dir, "networks.py"))
    copyfile("doerchnet/share_net.py" , os.path.join(CONFIG.STATS_DORCH_log_dir, "share_net.py"))

    if CONFIG.MISC_dataset.startswith('custom_'):
        stat_path = os.path.join(CONFIG.MISC_dataset_path,"Custom", CONFIG.MISC_dataset[7:],'stats.json')
    else:
        stat_path = os.path.join(CONFIG.MISC_dataset_path,CONFIG.MISC_dataset,'stats.json')
    with open(stat_path) as json_file:
        stats = json.load(json_file)

    patch_dims = (1,3,CONFIG.MISC_patch_size[0],CONFIG.MISC_patch_size[1])

    dataset_means = torch.tensor(stats['means'][:3])
    dataset_stds = torch.tensor(stats['stds'][:3])
    unNormImage = transforms.Normalize( ( - dataset_means / dataset_stds).tolist() , (1.0 / dataset_stds).tolist() )

    # Decrease learning rate every 20 epoch
    current_learning_rate = learning_rate
    action_dist = torch.zeros(dim)
    action_taken = torch.zeros(dim)
   
    def update_learning_rate(optimizer , learning_rate):
        for params in optimizer.param_groups:
            params['lr'] = learning_rate
    
    print("NetType:\t%s" % net_type)
    print("Dataset:\t%s" % CONFIG.MISC_dataset)

    for epoch in range(epochs):

        for (i,data) in enumerate(trainloader):

            batch_images , (start_crops_ , goal_crops_) = data
            batch_images = batch_images

            # Get random crops 
            crops_start, locs_start = get_random_crops(batch_images)
            crops_goal, locs_goal = get_random_crops(batch_images, locs_start, max_dist =  max_dist)
            actions = get_label(locs_start, locs_goal, dim = dim)

            crops_start = crops_start.to(device)
            crops_goal = crops_goal.to(device)
            action_dist += actions.sum(dim=0)
            temp_action_dist = action_dist / action_dist.sum()
            actions = actions.to(device)
            _ , outputs  = net(crops_start,crops_goal)

            loss = criterion(outputs, actions ).to(device)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # here do visualization
            if tot_itr % vis_iter == 0:
                pass
                #visualize_doerch(batch_images, locs_start, locs_goal ,outputs, unNormImage, save_name = 'train_vis_%d' % tot_itr, PATH = CONFIG.STATS_DORCH_log_dir)

            sc.s('Loss').collect(loss.item()) 

            # Calculate precision of model at corners, boundaries and middle part
            if dim in [25 , 80]:
                prec_corner, prec_boundary , prec_middle = calculate_precision(outputs , actions)

                sc.s('AccuracyCorners').collect(prec_corner)
                sc.s('AccuracyBoundaries').collect(prec_boundary)
                sc.s('AccuracyMiddle').collect(prec_middle)

            # Calculate the accuracy 
            acc = (torch.argmax(actions , dim = 1, keepdim = True) == torch.argmax(outputs , dim = 1, keepdim = True)).float().mean()        
            # print(actions.argmax(dim=1,keepdim=True))
            sc.s('Accuracy').collect(acc.item())
            sc.s('ActionsTaken').collect(F.one_hot(outputs.argmax(dim = 1, keepdim = False ), num_classes=dim).float().mean(dim = 0).cpu().numpy())
            sc.s('CorrectActions').collect(actions.mean(dim = 0).cpu().numpy())

            with torch.no_grad():
                # Run one batch on the valloader aswell
                try:
                    val_images, (start_crops_ , goal_crops_) = next(valiter)
                except:
                    valiter = iter(valloader)
                    val_images, ( start_crops_ , goal_crops_) = next(valiter)
                val_images = val_images.to(device)

                # Get random crops
                val_crops_start, val_locs_start = get_random_crops(val_images)
                val_crops_goal, val_locs_goal = get_random_crops(val_images, val_locs_start, max_dist = max_dist)
                
                val_actions = get_label(val_locs_start, val_locs_goal, dim = dim)
                val_crops_start = val_crops_start.to(device)
                val_crops_goal = val_crops_goal.to(device)
                val_actions = val_actions.to(device)
                
                _ , val_outputs  = net(val_crops_start,val_crops_goal)

                val_loss = criterion(val_outputs, val_actions ).to(device)

                # Logging
                sc.s('ValLoss').collect(val_loss.item())
                
                # Calculate precision of model at corners, boundaries and middle part
                if dim in [25 , 80]:
                    prec_corner, prec_boundary , prec_middle = calculate_precision(val_outputs , val_actions)

                    sc.s('ValAccuracyCorners').collect(prec_corner)
                    sc.s('ValAccuracyBoundaries').collect(prec_boundary)
                    sc.s('ValAccuracyMiddle').collect(prec_middle)

                # Calculate the accuracy 
                val_acc = (torch.argmax(val_actions , dim = 1, keepdim = True) == torch.argmax(val_outputs , dim = 1, keepdim = True)).float().mean()        
                sc.s('ValAccuracy').collect(val_acc.item())

                sc.s('ValActionsTaken').collect(F.one_hot(val_outputs.argmax(dim = 1, keepdim = False), num_classes=dim).float().mean(dim = 0).cpu().numpy())
                sc.s('ValCorrectActions').collect(val_actions.mean(dim = 0).cpu().numpy())

            # here do visualization
            if tot_itr % vis_iter == 0:

                # If segmentation information is enabled remove it and only visualize the RGB imaegs
                if CONFIG.RL_priv_use_seg:
                    val_images = val_images[:,0:3,:]

                visualize_batch_doerch(val_images, val_locs_start , val_locs_goal , val_outputs, unNormImage,PATH = CONFIG.STATS_DOERCH_vis_dir, save_name = "val_%d" % tot_itr)

            if tot_itr % print_iter == 0 or tot_itr == tot_nbr_itr - 1:
                print("Iteration:\t%d / %d" % ( tot_itr, tot_nbr_itr))
                sc.print()
                #print(action_dist)
                sc.save()

            tot_itr += 1

            if tot_itr % 5000 == 0:
                torch.save(net.state_dict() , os.path.join(CONFIG.STATS_DORCH_log_dir, "doerch_embedder"))
        
        # Lower the learning rate NOTE not active
        if (epoch + 1) % 10 == 0 and False:
            current_learning_rate *= lower_learning_rate_factor
            update_learning_rate(optimizer, current_learning_rate)
    info['Completed'] = True

    with open(os.path.join(CONFIG.STATS_DORCH_log_dir, "info.json") , 'w') as io:
        json.dump(info , io , indent = 4)
except: 
    # TODO - Use signal handlers instead so that we can propagate the exceptions
    #raise
    
    while True:

        i = input("save the model")
        if i=='y':
            # Save the encoder and the decoder
            torch.save(net.state_dict() , os.path.join(CONFIG.STATS_DORCH_log_dir, "doerch_embedder"))
            print("model saved")
            exit(1) 
        elif i == 'n':
            print("Not saving")
            exit(1)


torch.save(net.state_dict() , os.path.join(CONFIG.STATS_DORCH_log_dir, "doerch_embedder"))


