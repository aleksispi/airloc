#!/bin/env python3

import os
import time
import sys
import math
import glob
from dateutil.parser import parse
import numpy as np
from config import CONFIG
import json

import argparse

from utils.utils import find_latest_log

MA_SMOOTH = 0.02
START_ITER_PLOT = 50


# Metrics with shared y_axis
metricsWithSharedY = dict([
    ("ActionsTaken" ,["ValActionsTaken",]),
    #    ("IoU" , ["HasConverged","SeparatedHasConverged", "SeparatedIoU"])
    ])


# What are the customs_mas ?
def _custom_ma(data, ma_smooth=MA_SMOOTH):
    for idx, val in enumerate(data['values']):
        if idx < 30:
            data['mas_custom'][idx] = data['means'][idx]
        else:

            data['mas_custom'][idx] = (1 - ma_smooth) * data['mas_custom'][idx - 1] + ma_smooth * data['values'][idx]


# Function for plotting each subplot
def _plot(datas, ax, title='plot', xlabel='x', ylabel='y', start_it=0, max_x=None, max_y=None, min_y = None, show_draw='show' , legends = [], metric = None):
    legend_entries = []
    for (i, data) in enumerate(datas):

        # If the current data is full val print all values
        if legends[i] == 'FullVal':
            # Full val values are very sparse, no mean stuff and no filtering by start
            x = data['times']
            y = data['values']
            format = 'x'
        elif 'Separated' in metric:
            # Get number of datapoints
            num_last_points = min(1000 , len(data['values']))
            x = range(data['values'].shape[1])
            y = np.nanmean(data['values'][-num_last_points:], axis=0)
            format = 'o'
        else:
            start_it = START_ITER_PLOT
            x = data['times'][start_it:] #Just iterations
            y = data['mas_custom'][start_it:] #Some special mean value
            format = '-'
        p = ax.plot(x, y, format)
        if len(legends) > i:
            if ("Actions" in metric ):# or "Separated" in metric):
                for i in range(len(p)):
                    p[i].set_label('%i' % i)
            else:
                p[0].set_label(legends[i])


    if len(legends) > 0:
        ax.legend()
    ax.grid(False)

    # Calculate the axis in plot
    if min_y is None:
        min_y = np.min(y)
    if max_x is None:
        max_x = x[-1]
    if max_y is None:
        max_y = np.max(y)



def main(args):
    # Open json containing information about the training session
    try:
        with open(os.path.join(path_log_dir , "info.json") , 'r')  as json_file:
            training_info = json.load(json_file)
    except:
        print("\nERROR: Unable to open info json.\n")
        exit(1)

    # Plot training data
    # TODO - Put training data in sub folder. like "training_stats"
    data_type = 'Training'
    if args.legacy:
        path_log_data_dir = path_log_dir
    elif args.eval:
        path_log_data_dir = os.path.join(path_log_dir, "metrics_eval")
        prefix = 'Det'
    else:
        path_log_data_dir = os.path.join(path_log_dir, "metrics")
        prefix = 'Val'

    # Since we will be plotting training info. The info json will be the same as the training_info json
    info = training_info

    # The correct directory containing the data we want to plot is now in 'path_log_data_dir'
    metrics = [os.path.basename(metric)[:-4] for metric in glob.glob(path_log_data_dir+'/*')]

    AgentType = training_info['AgentType']
    startedTrainingAt = training_info['StartedTraining']
    nbrOfTrainableParameters = training_info['NbrOfTrainableParameters']
    dataset = training_info['Dataset']

    # Before plotting, print information about the retrived data
    print('')
    print("Training session:\t%s" % log_dir)
    print("Log directory:\t%s" % log_base)
    print("AgentType:\t%s" % AgentType)
    print("Number of trainable parameters:\t%d" % nbrOfTrainableParameters )
    print("Dataset:\t%s" % dataset)





    # Filterd
    filterdMetrics = list(filter(lambda s: not s.startswith(prefix) and not s.startswith('FullVal') ,metrics ))

    # Calculate dimensions of subplots
    n = len(filterdMetrics)

    # Make exception for Actions taken since otherwise plot would be unreadable
    if prefix + "ActionsTaken" in metrics:
        filterdMetrics.append(prefix + "ActionsTaken")
        n += 1
    if prefix + "CorrectActions" in metrics:
        filterdMetrics.append(prefix + "CorrectActions")
        n += 1

    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)


    # Plot all metrics for the selected run in same figure.
    fig , axes = plt.subplots(n_rows, n_cols, sharex = False, figsize = (25,14))

    axes_ndindicies = list(np.ndindex(axes.shape))

    for (i, axis_inds) in enumerate((axes_ndindicies)):
        ix , iy = axis_inds
        if len(filterdMetrics) <= i:
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

        if args.eval:
            legends = ['Stoc']
            if metric in [prefix + 'ActionsTaken',prefix + 'CorrectActions']:
                legends = [prefix]
        else:
            legends = ['Train']
            if metric in [prefix + 'ActionsTaken',prefix + 'CorrectActions']:
                legends = [prefix]
        plotData = [data]

        # Check if there is val data availble
        if args.eval:
            aux_metric = prefix + metric[4:]
        else:
            aux_metric = prefix + metric

        if aux_metric in metrics and 'CorrectActions' not in metric and 'ActionsTaken' not in metric:
            valData = np.load(os.path.join(path_log_data_dir , aux_metric + '.npz'))
            valData = {'means': valData['means'], 'mas': valData['mas'],
                    'values': valData['values'], 'times': valData['times'],
                    'mas_custom': np.zeros_like(valData['mas'])}
            _custom_ma(valData)
            legends.append(prefix)
            plotData.append(valData)

        try:
            # Check if there is full val available in the data
            if 'FullVal' + metric in metrics:
                fullValData = np.load(os.path.join(path_log_data_dir , 'FullVal' + metric + '.npz'))
                fullValData = {'means': fullValData['means'], 'mas': fullValData['mas'],
                        'values': fullValData['values'], 'times': fullValData['times'],
                        'mas_custom': np.zeros_like(fullValData['mas'])}
                _custom_ma(fullValData)
                fullValData['times'] = np.array(info['FullValIters'])
                if len(fullValData['times']) > 0:
                    legends.append('FullVal')
                    plotData.append(fullValData)
        except:
            pass

        # Now check loaded data to make sure there are enough data points
        if data['mas_custom'].shape[0] <= START_ITER_PLOT:
            print("\nERROR: Too few data points saved for plotting for metric \%s.\n" % metric)
            exit(1)

        # Check if axes should share y_axis with any other plot
        if metric in metricsWithSharedY:
            # Find which axes to share with
            for other_metric in metricsWithSharedY[metric]:
                indx = filterdMetrics.index(other_metric)
                other_ax_ind = axes_ndindicies[indx]

                axes[ix,iy].get_shared_y_axes().join(axes[ix,iy] , axes[other_ax_ind])

        _plot(plotData,axes[ix,iy], show_draw='show' , legends =legends, metric = metric)

        # Set title according to the json data file
        if args.eval:
            metric = metric[4:]

        axes[ix ,iy].set_title(metric)

    # Set title of entire window
    fig.canvas.manager.set_window_title("%s data from %s:\t%s" %( data_type, AgentType ,  log_dir))

    # set padding between plots
    fig.tight_layout(pad = 2.0)


    if args.show:
        plt.show()
    elif args.eval:
        # Find filepath
        filename = os.path.join(path_log_dir, "Eval_Statistics_%s_%s.png" % (AgentType , log_dir))
        plt.savefig(filename)

        print("\nPlot saved as:\t%s\n" % os.path.basename(filename))
    else:
        # Find filepath
        filename = os.path.join(path_log_dir, "Training_Statistics_%s_%s.png" % (AgentType , log_dir))
        plt.savefig(filename)

        print("\nPlot saved as:\t%s\n" % os.path.basename(filename))


if __name__ == '__main__':
    # Setup argparse

    parser = argparse.ArgumentParser()

    # Choose log dir either based on name or on number n
    log_selection = parser.add_mutually_exclusive_group()
    log_selection.add_argument("--log-dir" , "-l" , type = str , help = "Select log dir based on name")
    log_selection.add_argument("-n", type = int , help = 'Select the n:th latest log dir. 0 -> latest',default = 0)
    parser.add_argument("--eval", "-e", action="store_true", default = False, help = "Sets the program in eval mode")

    parser.add_argument("--saved-logs", "-s", action="store_true", default = False, help = "Select log dir from the 'saved_logs' folder.")
    parser.add_argument("--show" , action="store_true", default = False, help = "Show the plot on the screen instead of saving it.")

    parser.add_argument("--legacy" , action="store_true", default = False , help = "Legacy option for when metrics is stored right in log folder.")

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
        log_base = "logs"

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

    main(args)
