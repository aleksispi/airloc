#!/bin/env python3

import os
import time
import sys
import math
from dateutil.parser import parse
import numpy as np
from config import CONFIG
import json

import argparse

from utils.utils import find_latest_log

MA_SMOOTH = 0.01
START_ITER_PLOT = 30

# For some, inxplicable reason, need to run this here??


# What are the customs_mas ?
def _custom_ma(data, ma_smooth=MA_SMOOTH):
    for idx, val in enumerate(data['values']):
        if idx < 30:
            data['mas_custom'][idx] = data['means'][idx]
        else:

            data['mas_custom'][idx] = (1 - ma_smooth) * data['mas_custom'][idx - 1] + ma_smooth * data['values'][idx]


# Function for plotting each subplot
def _plot(datas, ax, title='plot', xlabel='x', ylabel='y', start_it=0, max_x=None, max_y=None, min_y = None,
         show_draw='show' , legends = []):
    legend_entries = []
    for (i, data) in enumerate(datas):

        # If the current data is full val print all values
        if legends[i] == 'FullVal':
            # Full val values are very sparse, no mean stuff and no filtering by start
            x = data['times']
            y = data['values']
            format = 'x'
        else:
            start_it = START_ITER_PLOT
            x = data['times'][start_it:] #Just iterations
            y = data['mas_custom'][start_it:] #Some special mean value
            format = '-'
        p  = ax.plot(x, y, format)

        if isinstance( p , list):
            for (i, pi) in enumerate(p):
                pi.set_label("%d" % i)
        else:
            if len(legends) > i:
                p.set_label(legends[i])

    if len(legends) > 0:
        ax.legend()
    ax.grid(False)

    # Calculate the axis in plot
    if min_y is None:
        min_y = np.min(y[start_it:])
    if max_x is None:
        max_x = x[-1]
    if max_y is None:
        max_y = np.max(y[start_it:])


# Setup argparse

parser = argparse.ArgumentParser()

# Choose log dir either based on name or on number n
log_selection = parser.add_mutually_exclusive_group()
log_selection.add_argument("--log-dir" , "-l" , type = str , help = "Select log dir based on name")
log_selection.add_argument("-n", type = int , help = 'Select the n:th latest log dir. 0 -> latest',default = 0)

parser.add_argument("--saved-logs", "-s", action="store_true", default = False, help = "Select log dir from the 'saved_logs' folder.")
parser.add_argument("--show" , action="store_true", default = False, help = "Show the plot on the screen instead of saving it.")

args = parser.parse_args()

## Script part

# Load and set correct settings for matplotlib based on wether to show the plot or just save it
if args.show:
    import tkinter
    import matplotlib
    from matplotlib import pyplot as plt
    matplotlib.use('TkAgg')
else:
    import matplotlib
    from matplotlib import pyplot as plt



# First determine wheter to use "saved_logs" or "logs"
if args.saved_logs:
    log_base = "saved_logs"
else:
    log_base = "doerchnet/logs"

log_base = os.path.join(CONFIG.MISC_project_root_path , log_base)


# Next determine which log dir should be used
if args.log_dir is not None:
    # Select dir named PLOT_log_dir
    log_dir = args.log_dir
else:
    # select the n:th latest log dir
    log_dir = find_latest_log(log_base , args.n)

# If log_dir is None, there were not that many logs
if log_dir is None:
    print("There are not that many training results in %s" % log_base)
    exit(1)

path_log_dir = os.path.join(log_base , log_dir)


# We have now identified a log dir from a training session
# We make sure that the directory actually exists before proceeding
if not os.path.exists(path_log_dir):
    print("Error, the selected log dir does not exist:\t%s" % path_log_dir)
    print("Check arguments and/or plot settings in config.py")
    exit(1)

# Open json containing information about the training session
try:
    with open(os.path.join(path_log_dir , "info.json") , 'r')  as json_file:
        training_info = json.load(json_file)
except:
    print("\nERROR: Unable to open info json.\n")
    exit(1)

# Now check wheter to print training data or evaluation data
if True:
    # Plot training data
    # TODO - Put training data in sub folder. like "training_stats"
    data_type = 'Training'
    path_log_data_dir = path_log_dir
    # Since we will be plotting training info. The info json will be the same as the training_info json
    info = training_info
else:
    # Plot evaluation data
    # First need to determine which eval run to plot from
    # This works the same way as when we choose which log dir to use
    path_eval_dir_base = os.path.join(path_log_dir , "eval_logs")
    if type(CONFIG.PLOT_eval_dir) == str:
        eval_dir = CONFIG.PLOT_eval_dir
    elif type(CONFIG.PLOT_eval_dir) == int:
        # select the n:th latest log dir
        eval_dir = find_latest_log(path_eval_dir_base , CONFIG.PLOT_eval_dir)
    else:
        # Select the latest available log dir
        eval_dir = find_latest_log(path_eval_dir_base , 0)
    if eval_dir is None:
        print("There are not that many eval results in %s" % path_log_dir)
        exit(1)

    # We now have path_log_data_dir which contains all metrics
    path_log_data_dir = os.path.join(path_eval_dir_base, eval_dir)
    data_type = 'Eval'
    # Load information about this eval run from the info file TODO change back to info.json
    with open(os.path.join(path_log_data_dir,"info.json"), 'r') as json_file:
        info = json.load(json_file)

# The correct directory containing the data we want to plot is now in 'path_log_data_dir'

metrics = info['Metrics']

# Before plotting, print information about the retrived data
print('')
print("Training session:\t%s" % log_dir)
print("Log directory:\t%s" % log_base)


# Filterd
filterdMetrics = list(filter(lambda s: not s.startswith('Val') and not s.startswith('FullVal'),metrics ))

# Calculate dimensions of subplots
n_cols = math.ceil(math.sqrt(len(filterdMetrics)))
n_rows = math.ceil(len(filterdMetrics) / n_cols)
#n_rows = 2

# Plot all metrics for the selected run in same figure.
fig , axes = plt.subplots(n_rows, n_cols, sharex = False, figsize = (12,8))

for (i, axis_inds) in enumerate(np.ndindex(axes.shape)):
    ix , iy = axis_inds

    if i >= len(filterdMetrics):
        axes[ix,iy].axis('off')
        continue

    metric = filterdMetrics[i]

    # Read data from log path
    log_path = os.path.join(path_log_data_dir, metric + '.npz')
    try:
        data = np.load(log_path)
    except:
        print("\nERROR: Unable to load data for metric:\t%s\n" % metric)
        exit(1)

    data = {'means': data['means'], 'mas': data['mas'],
            'values': data['values'], 'times': data['times'],
            'mas_custom': np.zeros_like(data['mas'])}
    _custom_ma(data)
    legends = ['Train']
    plotData = [data]

    # Check if there is val data availble
    if 'Val' + metric in metrics:
        valData = np.load(os.path.join(path_log_data_dir , 'Val' + metric + '.npz'))
        valData = {'means': valData['means'], 'mas': valData['mas'],
                'values': valData['values'], 'times': valData['times'],
                'mas_custom': np.zeros_like(valData['mas'])}
        _custom_ma(valData)
        legends.append('Val')
        plotData.append(valData)

    # Now check loaded data to make sure there are enough data points
    if data['mas_custom'].shape[0] <= START_ITER_PLOT:
        print("\nERROR: Too few data points saved for plotting.\n")
        exit(1)

    _plot(plotData,axes[ix,iy], show_draw='show' , legends =legends)

    # Set title according to the json data file
    axes[ix ,iy].set_title(metric)

# Set title of entire window
fig.canvas.manager.set_window_title("%s data from :\t%s" %( data_type,   log_dir))

# set padding between plots
fig.tight_layout(pad = 3.0)


if args.show:
    plt.show()
else:
    # Find filepath
    filename = os.path.join(path_log_dir, "Training_Statistics_%s_.png" % ( log_dir))
    plt.savefig(filename)

    print("\nPlot saved as:\t%s\n" % os.path.basename(filename))
