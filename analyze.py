import os
import sys
import numpy as np
import pprint

def read_experiment_log(logfile):
    """
        in log file: 
            'log MSE loss (train set)', 'log MSE loss (val set)', 'log MSE loss (test set)'

        {'train':[train_losses], 'val': [val_losses], 'test': [test_losses]}
    """
    log = open(logfile).readlines()
    log = [row.strip().split("\t") for row in log]
    logdata = np.array([map(float, row) for row in log[1:]]) # 0: train, 1: val, 2: test
    logdata = np.exp(logdata)  # convert to normal space
    return {'train': logdata[:, 0], 'val': logdata[:, 1], 'test': logdata[:, 2]}

def gather_val_losses(experiments, epochs):
    # specify paths
    out_root = 'opmjlogs'

    val_losses = {}  # keyed by iter, then by experiment name
    for epoch in epochs:
        val_losses[epoch] = {}

    # plot
    for experiment_folder in experiments:
        experiment_folder = os.path.join(out_root, experiment_folder)

        # read in the experiment log
        exp_val_losses = read_experiment_log(os.path.join(experiment_folder,'experiment.log'))['val']
        for epoch in epochs:
            if len(exp_val_losses) >= epoch:
                val_losses[epoch][os.path.basename(experiment_folder)] = exp_val_losses[epoch]
            else:
                val_losses[epoch][os.path.basename(experiment_folder)] = np.NaN
    return val_losses

def sort_best(val_losses, epochs):
    for epoch in epochs:
        experiments = val_losses[epoch]
        # sort experiments by loss
        nans = {e:v for e,v in experiments.items() if experiments[e] is np.NaN}
        non_nans = {e:v for e,v in experiments.items() if experiments[e] is not np.NaN}
        sorted_experiments = sorted([x for x in non_nans.items()], key=lambda x: x[1])
        val_losses[epoch] = sorted_experiments
    return val_losses

experiments = [
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_lrdecay_every10000',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_lrdecay_every5000',
                # 'balls_n10_t60_ex50000__balls_n10_t60_ex50000_lrdecay_every2500',
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_lrdecay_every2500',
                # 'balls_n2_t60_ex50000__balls_n2_t60_ex50000_batchnorm',
                # 'balls_n5_t60_ex50000__balls_n5_t60_ex50000_lrdecay_every5000',

                # 'balls_n8_t60_ex50000__balls_n8_t60_ex50000_lrdecay_every2500',
                # 'balls_n5_t60_ex50000__balls_n5_t60_ex50000_lrdecay_every2500',
                # 'balls_n7_t60_ex50000__balls_n7_t60_ex50000_lrdecay_every2500',
                # 'balls_n9_t60_ex50000__balls_n9_t60_ex50000_lrdecay_every2500',
                # 'balls_n6_t60_ex50000__balls_n6_t60_ex50000_lrdecay_every2500',

                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_modelind',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_modelcat_lr3e-5',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_modelcat_lr3-e5_lineardecoder',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_modelind_lineardecoder',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000__balls_n5_t60_ex50000',
                # # 'balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n3_t60_ex50000',
                # 'balls_n5_t60_ex50000,balls_n3_t60_ex50000__balls_n4_t60_ex50000',
                # # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m',
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lr3e-3',
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lr7e-4',
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lr5e-4',

                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000__balls_n5_t60_ex50000_modelcat_lr3e-05',
                # 'balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n3_t60_ex50000_modelcat_lr3e-05',
                # 'balls_n5_t60_ex50000,balls_n3_t60_ex50000__balls_n4_t60_ex50000_modelcat_lr3e-05',

                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecayevery5000',


                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers3_lr0.0001_lrdecay_every2500', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers4_lr0.0001_lrdecay_every2500', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers3_lr0.0001_lrdecay_every5000', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers4_lr0.0001_lrdecay_every5000', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers3_lr0.001_lrdecay_every2500', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers4_lr0.001_lrdecay_every2500', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers3_lr0.001_lrdecay_every5000', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers4_lr0.001_lrdecay_every5000', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers3_lr0.005_lrdecay_every2500', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers4_lr0.005_lrdecay_every2500', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers3_lr0.005_lrdecay_every5000', 
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m_lrdecay_every2500_layers4_lr0.005_lrdecay_every5000',

                'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past10',
                'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past9',
                'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past8',
                'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past7',
                'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past6',
                'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past5',
                'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past4',
                'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past3',

                ]

epochs = [20, 40, 60, 80, 100]
val_losses = gather_val_losses(experiments, epochs)
val_losses = sort_best(val_losses, epochs)
pprint.pprint(val_losses)

