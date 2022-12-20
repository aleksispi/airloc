# AiRLoc: Aerial View Goal Localization with Reinforcement Learning

![airloc-img](https://user-images.githubusercontent.com/32370520/188417494-6e1ee3c3-e221-4a4a-b067-f5de1c178e0c.png)

Official PyTorch implementation of [AiRLoc](https://arxiv.org/abs/2209.03694) by [Aleksis Pirinen](https://www.ri.se/en/person/aleksis-pirinen), Anton Samuelsson, John Backsund and [Kalle Åström](https://www.maths.lu.se/staff/kalleastrom/).

### Installation
The code is based on Python 3.

Setup Conda environment:
```
conda create -n airloc
conda activate airloc
pip install -r requirements.txt
```

### News: Models trained on images of pre-disaster scenarios and which generalizes to post-disaster scenarios have been uploaded! These have been trained and evaluated on the [xBD dataset](https://arxiv.org/pdf/1911.09296.pdf). See `logs/`, where a pretrained AiRLoc model has been added for xBD, together with AiRLoc models trained on [*Massachusetts Buildings*](https://central.bac-lac.gc.ca/.item?id=TC-OTU-35911&op=pdf&app=Library&oclc_number=1032987428).

### Code structure overview
All configurations of various models etcetera are set in `config.py`. Files related to the patch embedder is found in the folder `doerchnet`. Various logging (training statistics, final model weights, and so on) is sent to the folder `logs`. The folder `data` contains the data (including splits) used in the paper. When running the code the first time, first unzip all data folders.

### Training
To pretrain the patch embedder in a self-supervised fashion a la [Doersch et al. (2015)](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf), run from the top level folder:
```
python -m doerchnet.train
```
Various statistics (including model weights) are sent to a specific subfolder in the folder `doerchnet/logs`. Within that subfolder, the weights will be called `doersch_embedder`. **Note:** This repo already includes *Massachusetts Buildings*-pretrained patch embedder weights (both with and without segmentation as input), and the weights are already pointed to in `config.py` (see the flag `CONFIG.RL_pretrained_doerch_net`). This repo also includes xBD-pretrained embedder weights (`without-sem-seg-pre-michael`). Hence you may choose to skip the patch embedder pretraining step and immediately proceed to training AiRLoc (see below).

To train AiRLoc, run from the top level folder:
```
python -m training.train_agent
```
Do not forget to point `CONFIG.RL_pretrained_doerch_net` to the pretrained patch embedder (otherwise AiRLoc will be trained with RL from scratch, which significantly reduces the success rate of the final agent). Various statistics, including model weights, are sent to the folder `logs` (if this folder does not exist it will be automatically created).

To plot various statistics during training, run from the top level folder:
```
python -m plot_results --log-dir <folder log path> 
```
These statistics are stored in the folder `logs`.

**Note:** This repo already includes *Massachusetts Buildings*-trained AiRLoc models (both with and without segmentation as input). This repo also includes an xBD-trained AiRLoc model (`2022-09-13_14-07-42-505667_pre_disaster`). Hence you may choose to skip the RL-training step and immediately proceed to evaluating AiRLoc (see below).

### Evaluation
To evaluate an agent on the validation set of _Masschusetts Buildings_, first ensure that `CONFIG.MISC_dataset` points to 'masa_filt'. Then run from the top level folder:
```
python -m eval.eval_agent --log_dir <folder log path> --seed 0 --split val 
```
To evaluate on the test set, change to `--split test` above. To evaluate on the _Dubai_ dataset, first change `CONFIG.MISC_dataset` so that it points to 'dubai'. Then run from the top level folder:
```
python -m eval.eval_agent --log_dir <folder log path> --seed 0
```
Note that `--log_dir` refers to a folder with a trained model within the `logs` folder (every training run yields a subfolder in `logs`). It assumes model weights to be in such a subfolder, called 'final_model'.

To evaluate _Priv local_, do as follows:
1. Create a folder in `logs`, called for example `PrivLocal` (let's call it like that in the following).
2. Copy the `doerch_embedder` weights you wish to use from the desired subfolder in `doerchnet/logs` to the created folder (`PrivLocal`). Rename `doerch_embedder` as `final_model`.
3. Copy a `config.py` file into the `PrivLocal` folder, and set `CONFIG.RL_agent_network = 'Agent'`, `CONFIG.MISC_priv = True`, and `CONFIG.RL_LSTM_pos_emb = False`.
4. Repeat the steps in the same way as described under _Evaluation_ above. Note that `--log_dir` should point to `PrivLocal`.

To evaluate _Priv local (sem seg)_, make sure to copy `doersch_embedder` weights that correspond to using a semantic segmentation channel in the input. Also set `CONFIG.RL_priv_use_seg = True`.

To evaluate _Priv random_, do as follows:
To evaluate _Priv local_, do as follows:
1. Create a folder in `logs`, called for example `PrivRandom` (let's call it like that in the following).
2. Copy a `config.py` file into the `PrivRandom` folder, and set `CONFIG.RL_agent_network = 'RandomAgent'` and `CONFIG.MISC_priv = True`.
3. Repeat the steps in the same way as described under _Evaluation_ above. Note that `--log_dir` should point to `PrivRandom`.

### Create dataset

To create dataset usable by this training and evaluation setup one needs a set of satellite or drone images. Put these images in a folder called 'image' and put this folder in a surrounding folder with the intended dataset name. This folder is to be placed in the location indicated by 'CONFIG.MISC\_dataset\_path' in the config.py file. Then to create a split file with train/val/test split run:

```
python3 -m utils.dataset_utils --dataset <dataset_name> --grid-game
```

where `<dataset_name>` is the name of the folder in which the dataset is placed. Then to calculate mean and std for normalization run:

```
python3 -m utils.normalize_dataset --dataset <dataset_name>
```

where again, `<dataset_name>` is the name of the dataset folder. To create fixed eval splits for this dataset run:

```
python3 -m utils.create_split_file --dataset <dataset_name> --split <split>
```

where `<split>` is the intenteded partition of the dataset, validation is the default.

### Grid sizes other than 5x5

Note that if the dataset is intended to be used with grid sizes other than the default 5x5 the config file has to be adjusted. Change the `CONFIG.MISC_im_size` according to the desired grid size.
