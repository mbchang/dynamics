import os
import sys
import plot_results
import errno   

import cv2
import numpy as np
from images2gif import writeGif
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pprint 
import subprocess
import itertools

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import FormatStrFormatter
import itertools
matplotlib.rcParams['axes.linewidth'] = 0.1
import re


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

experiments_dict = {
    'Mixed Prediction': [
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),  # accidentally killed
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_lr0.0003_modelbffobj_seed0', 'NPE No Neighborhood'),
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),  # accidentally killed
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_lr0.0003_modelbffobj_seed1', 'NPE No Neighborhood'),
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),  # accidentally killed
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_lr0.0003_modelbffobj_seed2', 'NPE No Neighborhood'),
        # ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE'),
    ],

    'Mixed Prediction Mass': [
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_rs_fast_lr0.0003_modelind_seed0', 'Independent'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_rs_fast_lr0.0003_modelind_seed1', 'Independent'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_rs_fast_lr0.0003_modelind_seed2', 'Independent'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE'),


        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
    ],


    'Mixed Generalization': [
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE'),
    ], 

    'Mixed Generalization Mass': [
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_rs_fast_lr0.0003_modelind_seed0', 'Independent'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_rs_fast_lr0.0003_modelind_seed1', 'Independent'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_rs_fast_lr0.0003_modelind_seed2', 'Independent'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rda,mixed_n4_t60_ex50000_m_z_o_dras3_rda__mixed_n5_t60_ex50000_m_z_o_dras3_rda,mixed_n6_t60_ex50000_m_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE'),

    ],

    'Balls Prediction': [
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_lr0.0003_modelbffobj_seed0', 'NPE No Neighborhood'),  # accidentally killed
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE: 4 - '),

        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0003_cuda_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0003_cuda_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr3e-05_cuda_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr3e-05_cuda_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0003_cuda_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0003_cuda_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr3e-05_cuda_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr3e-05_cuda_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0003_cuda_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0003_cuda_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr3e-05_cuda_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr3e-05_cuda_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
    ],

    'Balls Prediction Mass': [
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE'),

        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
    ],

    'Balls Generalization': [
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE: 3,4,5 - '),

        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),



    ],

    'Balls Generalization Mass': [
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),  # killed
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE'),

        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim64_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim128_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim256_fast_lr3e-05_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim64_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim128_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_rnn_dim256_fast_lr0.0003_modelbl_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),

    ],

    # 'BLSTM Search': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr3e-05_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr3e-05_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr3e-05_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0001_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0001_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0001_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0003_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0003_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0003_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.001_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.001_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.001_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.0001'),
    # ],

    # 'BLSTM Search Layers 2 Dim 256': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 3e-05'),
    # ],


    # 'BLSTM Search Layers 2 Dim 128': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 3e-05'),
    # ],
    # 'BLSTM Search Layers 2 Dim 64': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.0001'),
    # ],

    # 'BLSTM Search Layers 3 Dim 256': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 3e-05'),
    # ],

    # 'BLSTM Search Layers 3 Dim 128': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 3e-05'),
    # ],
    # 'BLSTM Search Layers 3 Dim 64': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.0001'),
    # ],

    # 'BLSTM Search Layers 3': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr3e-05_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr3e-05_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr3e-05_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0001_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0001_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0001_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0003_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0003_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0003_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.001_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.001_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.001_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 0.001'),
    # ],

    # 'BLSTM Search lr 0.0001': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0001_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0001_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0001_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 0.0001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.0001_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.0001'),
    # ],

    # 'BLSTM Search lr 0.0003': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr0.0003_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.0003_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.0003_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 0.0003'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.0003_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 0.0003'),
    # ],

    # 'BLSTM Search lr 0.001': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr0.001_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr0.001_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr0.001_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 0.001'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr0.001_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 0.001'),
    # ],

    # 'BLSTM Search 3e-05': [  # you want to decide which learning rate to kick out and which layers to kick out
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim64_fast_lr3e-05_cuda_modelbl_seed0', '3 Layers, Dim 64 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim128_fast_lr3e-05_cuda_modelbl_seed0', '3 Layers, Dim 128 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_rnn_dim256_fast_lr3e-05_cuda_modelbl_seed0', '3 Layers, Dim 256 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim64_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 64 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim128_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 128 Learning Rate 3e-05'),
    #     ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers2_rs_rnn_dim256_fast_lr3e-05_cuda_modelbl_seed0', '2 Layers, Dim 256 Learning Rate 3e-05'),
    # ],


    'Tower': [
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_vlambda100_rs_fast_lr0.0003_modelind_lambda100_seed0', 'Independent'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_vlambda100_rs_fast_lr0.0003_modelind_lambda100_seed1', 'Independent'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_vlambda100_rs_fast_lr0.0003_modelind_lambda100_seed2', 'Independent'),
        ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_nbrhd_vlambda100_rs_fast_nlan_lr0.0003_modelnp_lambda100_seed0', 'NP'),
        ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_nbrhd_vlambda100_rs_fast_nlan_lr0.0003_modelnp_lambda100_seed1', 'NP'),
        ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_nbrhd_vlambda100_rs_fast_nlan_lr0.0003_modelnp_lambda100_seed2', 'NP'),
        ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelbffobj_lambda100_seed0', 'NPE'),
        ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelbffobj_lambda100_seed1', 'NPE'),
        ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelbffobj_lambda100_seed2', 'NPE'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim64_fast_nlan_lr3e-05_modelbl_lambda100_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim64_fast_nlan_lr3e-05_modelbl_lambda100_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim128_fast_nlan_lr3e-05_modelbl_lambda100_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim128_fast_nlan_lr3e-05_modelbl_lambda100_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim256_fast_nlan_lr3e-05_modelbl_lambda100_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim256_fast_nlan_lr3e-05_modelbl_lambda100_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim64_fast_nlan_lr0.0003_modelbl_lambda100_seed0', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim64_fast_nlan_lr0.0003_modelbl_lambda100_seed1', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim128_fast_nlan_lr0.0003_modelbl_lambda100_seed0', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim128_fast_nlan_lr0.0003_modelbl_lambda100_seed1', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim256_fast_nlan_lr0.0003_modelbl_lambda100_seed0', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim256_fast_nlan_lr0.0003_modelbl_lambda100_seed1', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
    ],

    'Tower Generalization': [
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_vlambda100_rs_fast_lr0.0003_modelind_seed0_lambda100', 'Independent'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_vlambda100_rs_fast_lr0.0003_modelind_seed1_lambda100', 'Independent'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_vlambda100_rs_fast_lr0.0003_modelind_seed2_lambda100', 'Independent'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim64_fast_nlan_lr3e-05_cuda_modelbl_seed0_lambda100', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim64_fast_nlan_lr3e-05_cuda_modelbl_seed1_lambda100', 'Bidirectional LSTM Layers 3 Dim 64 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim128_fast_nlan_lr3e-05_cuda_modelbl_seed0_lambda100', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim128_fast_nlan_lr3e-05_cuda_modelbl_seed1_lambda100', 'Bidirectional LSTM Layers 3 Dim 128 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim256_fast_nlan_lr3e-05_cuda_modelbl_seed0_lambda100', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim256_fast_nlan_lr3e-05_cuda_modelbl_seed1_lambda100', 'Bidirectional LSTM Layers 3 Dim 256 LR 3e-05'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim64_fast_nlan_lr0.0003_cuda_modelbl_seed0_lambda100', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim64_fast_nlan_lr0.0003_cuda_modelbl_seed1_lambda100', 'Bidirectional LSTM Layers 3 Dim 64 LR 0.0003'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim128_fast_nlan_lr0.0003_cuda_modelbl_seed0_lambda100', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim128_fast_nlan_lr0.0003_cuda_modelbl_seed1_lambda100', 'Bidirectional LSTM Layers 3 Dim 128 LR 0.0003'),  # re run
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim256_fast_nlan_lr0.0003_cuda_modelbl_seed1_lambda100', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        # ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers3_vlambda100_rs_rnn_dim256_fast_nlan_lr0.0003_cuda_modelbl_seed0_lambda100', 'Bidirectional LSTM Layers 3 Dim 256 LR 0.0003'),
        ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelnp_seed0_lambda100', 'NP'),
        ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelnp_seed1_lambda100', 'NP'),
        ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelnp_seed2_lambda100', 'NP'),
        ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelbffobj_seed0_lambda100', 'NPE'),
        ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelbffobj_seed1_lambda100', 'NPE'),
        ('tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelbffobj_seed2_lambda100', 'NPE'),

    ],

    'Balls': [
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE: 4 - '),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE: 4 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE: 3,4,5 - '),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE: 3,4,5 - '),
    ],

    'Balls Mass': [

        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP: 4 - 4'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP: 4 - 4'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP: 4 - 4'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE: 4 - 4'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE: 4 - 4'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE: 4 - 4'),
    
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'NP: 3,4,5 - 6,7,8'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'NP: 3,4,5 - 6,7,8'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'NP: 3,4,5 - 6,7,8'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE: 3,4,5 - 6,7,8'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE: 3,4,5 - 6,7,8'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE: 3,4,5 - 6,7,8'),

    ]

    
}

experiments = list(set(itertools.chain.from_iterable([[x[0] for x in y] for y in experiments_dict.values()])))


experiments_to_visualize = [
#     # 'tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda100_modelbffobj_lambda100_batch_norm',
#     # 'tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda100_modelbffobj_lambda100_batch_norm',
#     'tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda10_modelbffobj_lambda10_batch_norm',
#     # 'tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda10_modelbffobj_lambda10_batch_norm',
    # 'balls_n6_t60_ex50000_rd__balls_n6_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',


    # Balls Prediction
    # 'balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelind',
    # 'balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0',
    # 'balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0',

    # # Balls Prediction Mass
    # 'balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelind',
    # 'balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0',
    # 'balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0',

    # # Balls Generalization 
    # 'balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelind',
    # 'balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0',
    # 'balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0',

    # # Balls Generalization Mass
    # 'balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelind',
    # 'balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0',
    # 'balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0',

    # Towers Prediction
    # 'tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_nbrhd_vlambda100_rs_fast_nlan_lr0.0003_modelnp_lambda100_seed0',  # nothing here
    'tower_n5_t120_ex25000_rda__tower_n5_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelbffobj_lambda100_seed0',

    'tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelnp_seed0_lambda100',
    'tower_n5_t120_ex25000_rda,tower_n6_t120_ex25000_rda__tower_n7_t120_ex25000_rda,tower_n8_t120_ex25000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_nlan_lr0.0003_vlambda100_modelbffobj_seed0_lambda100'

]


# specify paths
out_root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/opmjlogs'
in_root = '/om/user/mbchang/physics/lua/logs'
copy_prefix = 'rsync -avz --exclude \'*.t7\' mbchang@openmind7.mit.edu:'
remote_prefix = '/om/user/mbchang/physics/lua/logs/'
js_root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/demo/js'
    
# pprint.pprint([[x[0] for x in y] for y in experiments_dict.values()])
# pprint.pprint(experiments)
# print(experiments)
# assert False

def wccount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
    return int(out.strip().partition(b' ')[0])

def parse_exp_log(experiments):
    """
    This returns a dictionary of the number of lines in the experiment.log for each experiment
    """
    exp_log_lengths = {}
    for experiment in experiments:
        experiment_folder = os.path.join(out_root, experiment)
        exp_log = os.path.join(experiment_folder, 'experiment.log')
        if os.path.exists(exp_log):
            exp_log_lengths[experiment] = wccount(exp_log)
    return exp_log_lengths

def copy(experiments):
    exp_log_lengths = parse_exp_log(experiments)
    # pprint.pprint(exp_log_lengths)
    # copy
    if len(experiments) > 1:
        remote_paths = remote_prefix + '\{' + ','.join(['\\"' + e + '\\"' for e in experiments]) + '\} '
        command = copy_prefix + remote_paths + out_root
    else:
        remote_paths = remote_prefix + experiments[0] + ' '
        command = copy_prefix + remote_paths + out_root + '/'

    response = raw_input('Running command:\n\n' + command + '\n\nProceed?[y/n]')
    if response == 'y':
        print('## COPY ##')
        os.system(command)
    elif response != 'n':
        response = raw_input('Running command:\n\n' + command + '\nProceed?[y/n]')
    else:
        print 'Not running command.'
        sys.exit(0)

    # here return the experiment folders where experiment.log had changed
    new_exp_log_lengths = parse_exp_log(experiments)
    experiments_to_plot = [e for e in experiments if e not in exp_log_lengths or exp_log_lengths[e] != new_exp_log_lengths[e]]

    print 'experiments to plot'
    pprint.pprint(experiments_to_plot)

    return experiments_to_plot

def plot(experiments):
    print('## PLOT ##')
    # plot
    for experiment_folder in experiments:
        try:
            experiment_folder = os.path.join(out_root, experiment_folder)
            # command = 'th plot_results.lua -hid -infolder ' + experiment_folder
            command = 'th plot_results.lua -infolder ' + experiment_folder
            print '#'*80
            print command
            os.system(command)

            visual_folder = os.path.join(experiment_folder, 'visual')
            mkdir_p(visual_folder)

            # if there exists a folder that ends with predictions, make subfolders in there.
            for f in os.listdir(experiment_folder):
                f = os.path.join(experiment_folder,f)
                if os.path.isdir(f) and 'predictions' in os.path.basename(f):
                    for j in os.listdir(f):
                        batch_folder = os.path.join(visual_folder,j[:-len('.json')])
                        mkdir_p(batch_folder)
                        print 'Made', batch_folder, 'if it did not already exist.\n' + '-'*80


            # print 'plot hidden state'
            # plot_results.plot_hid_state(experiment_folder)  # TODO! check if filepath is correct
        except KeyboardInterrupt:
            sys.exit(0)

# return np array from log file
def read_log_file(log_file):
    data = {'train':[],'val':[],'test':[]}
    with open(log_file, 'r') as f:
        raw = f.readlines()
    for t in xrange(1,len(raw)):
        [train, val, test] = raw[t].split('\t')[:3]
        data['train'].append(train)
        data['val'].append(val)
        data['test'].append(test)
    return data

def read_tva_file(log_file):
    data = {'ang_vel_loss':[],'vel_loss':[],'avg_rel_mag_error':[], 'loss': [], 'avg_ang_error': []}
    with open(log_file, 'r') as f:
        raw = f.readlines()
    for t in xrange(1,len(raw)):
        [ang_vel_loss, vel_loss, avg_rel_mag_error, loss, avg_ang_error] = raw[t].split('\t')[:5]
        data['ang_vel_loss'].append(ang_vel_loss)
        data['vel_loss'].append(vel_loss)
        data['avg_rel_mag_error'].append(avg_rel_mag_error)
        data['loss'].append(loss)
        data['avg_ang_error'].append(avg_ang_error)
    return data

# Cosine Difference   Timesteps   Magnitude Difference    MSE Error
def read_div_file(log_file):
    data = {'Cosine Difference':[],'Timesteps':[],'Magnitude Difference':[], 'MSE Error': []}
    with open(log_file, 'r') as f:
        raw = f.readlines()
    for t in xrange(1,len(raw)):
        [ang_vel_loss, t, avg_rel_mag_error, loss] = raw[t].split('\t')[:4]
        data['Cosine Difference'].append(ang_vel_loss)
        data['Timesteps'].append(t)
        data['Magnitude Difference'].append(avg_rel_mag_error)
        data['MSE Error'].append(loss)
    return data

def read_inf_log_file(log_file):
    data = {'mass':[]}
    with open(log_file, 'r') as f:
        raw = f.readlines()
    for t in xrange(1,len(raw)):
        # print(raw[t].strip())
        mass = raw[t].strip()
        data['mass'].append(mass)
    return data

def plot_experiment(exp_list, dataset, outfolder, outfile):
    ys = []
    xs = []

    fig, ax = plt.subplots()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

    # read in the experminet.logs from exp_list
    for exp, label in exp_list:
        # read exp.log
        y = read_log_file(os.path.join(*[out_root,exp,'experiment.log']))[dataset]
        x = range(len(y)) # TODO!

        ax.plot(x,y, label=label)

    plt.legend(fontsize=14)
    plt.xlabel('Iterations (x 10000)')
    plt.ylabel('Log MSE Loss')  # TODO!
    plt.savefig(os.path.join(outfolder, outfile))
    plt.close()

def custom_plot(x, means, mins, maxs, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, means, **kwargs)
    ax.fill_between(x, mins, maxs, facecolor=base_line.get_color(), alpha=0.5, linewidth=0.0)

    # if logscale:
    #     ax.set_yscale('symlog', basey=10)


def plot_experiment_error(exp_list, dataset, outfolder, outfile,two_seeds):
    ys = []
    xs = []

    fig, ax = plt.subplots()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    marker = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')) 

    # first group all the same labels together. You will use these for error bars
    exp_groups = {}

    for name, label in exp_list:
        exp_groups.setdefault(label, []).append(name)

    for label in exp_groups:
        indep_runs = exp_groups[label]

        indep_run_data = [[float(x) for x in read_log_file(os.path.join(*[out_root,exp,'experiment.log']))[dataset]] for exp in indep_runs]

        if two_seeds:
            min_length = min(len(x) for x in indep_run_data)
            min_length_index = -1
            for i in range(len(indep_run_data)):
                if len(indep_run_data[i]) == min_length:
                    min_length_index = i
                    break
            if len(indep_run_data) > 2:
                indep_run_data = [indep_run_data[i] for i in range(len(indep_run_data)) if i != min_length_index]

        # trim to the minimum length
        min_length = min(len(x) for x in indep_run_data)
        indep_run_data = np.array([x[:min_length] for x in indep_run_data])  # (num_seeds, min_length)

        # convert it from ln scale
        indep_run_data = np.exp(indep_run_data)

        # convert it to log base 10 scale
        indep_run_data = np.log10(indep_run_data)

        print label, indep_run_data, min_length

        # compute max min and average
        maxs = np.max(indep_run_data,0)
        mins = np.min(indep_run_data,0)
        means = np.mean(indep_run_data,0)

        x = range(min_length) # TODO

        custom_plot(x, means, mins, maxs, label=label, marker=marker.next())
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        if any('tower' in x for x in indep_runs):
            ax.set_ylim(-8,-1)
        else:
            ax.set_ylim(-4, -1)

    leg = plt.legend(fontsize=14, frameon=False)
    plt.xlabel('Iterations (x 100000)')
    plt.ylabel('Mean Squared Error')  # TODO!
    plt.savefig(os.path.join(outfolder, outfile))
    plt.close()


def plot_tva_error(exp_list, dataset, outfolder, outfile,two_seeds):
    ys = []
    xs = []

    fig, ax = plt.subplots()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    marker = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')) 

    # first group all the same labels together. You will use these for error bars
    if not exp_list: return
    exp_groups = {}

    for name, label in exp_list:
        exp_groups.setdefault(label, []).append(name)

    for label in exp_groups:

        # fig, ax = plt.subplots()
        # ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        # marker = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')) 

        indep_runs = exp_groups[label]

        # here get all the prediction folders
        # prediction_folder_exps = []
        for exp in indep_runs:
            prediction_folders = [x for x in os.listdir(os.path.join(out_root,exp)) if 'predictions' in x]
        #     if not prediction_folders: return
        #     prediction_folder = prediction_folders[0]  # hacky
        #     print prediction_folder
        #     print os.path.join(*[out_root,exp,prediction_folder])
        #     print os.listdir(os.path.join(*[out_root,exp,prediction_folder]))
        #     if 'tva.log' not in os.listdir(os.path.join(*[out_root,exp,prediction_folder])): 
        #         # print 'complain'
        #         # continue
        #         return 
        #     # else:
        #     #     print 'yaas' #return
        #     # prediction_folder_exps.append(prediction_folder)
        # print 'data'
        for prediction_folder in prediction_folders:

            indep_run_data = [[float(x) for x in read_tva_file(os.path.join(*[out_root,exp,prediction_folder,'tva.log']))[dataset]] for exp in indep_runs]

            if two_seeds:
                min_length = min(len(x) for x in indep_run_data)
                min_length_index = -1
                for i in range(len(indep_run_data)):
                    if len(indep_run_data[i]) == min_length:
                        min_length_index = i
                        break
                if len(indep_run_data) > 2:
                    indep_run_data = [indep_run_data[i] for i in range(len(indep_run_data)) if i != min_length_index]

            # trim to the minimum length
            min_length = min(len(x) for x in indep_run_data)
            # min_length = 11
            indep_run_data = np.array([x[:min_length] for x in indep_run_data])  # (num_seeds, min_length)

            # convert it from ln scale
            # indep_run_data = np.exp(indep_run_data)

            # convert it to log base 10 scale
            # indep_run_data = np.log10(indep_run_data)

            print label, indep_run_data, min_length

            # compute max min and average
            maxs = np.max(indep_run_data,0)
            mins = np.min(indep_run_data,0)
            means = np.mean(indep_run_data,0)

            x = range(min_length) # TODO

            print 'x',x, len(x)
            print 'means',means, len(means)
            print 'mins',mins, len(mins)
            print 'maxs',maxs, len(maxs)

            custom_plot(x, means, mins, maxs, label=label + ' ' + find_num_obj_in_substring(prediction_folder) + ' objects', marker=marker.next())
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # if any('tower' in x for x in indep_runs):
            #     ax.set_ylim(-8,-1)
            # else:
            #     ax.set_ylim(-4, -1)
            if dataset == 'avg_ang_error':
                ax.set_ylim(0.85,1)
            elif dataset == 'avg_rel_mag_error':
                ax.set_ylim(0.0,0.2)

            
    # leg = plt.legend(fontsize=20, frameon=False)
    plt.xlabel('Iterations (x 100000)')
    if dataset =='avg_ang_error':
        plt.ylabel('Cosine Distance')  # TODO!
        leg = plt.legend(fontsize=14, frameon=False, loc='lower right')
    elif dataset == 'avg_rel_mag_error':
        plt.ylabel('Relative Error in Magnitude')  # TODO!
        leg = plt.legend(fontsize=14, frameon=False, loc='upper right')
    plt.savefig(os.path.join(outfolder, outfile))
    plt.close()


def plot_div_error(exp_list, dataset, outfolder, outfile,two_seeds):
    ys = []
    xs = []

    fig, ax = plt.subplots()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    marker = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')) 

    # first group all the same labels together. You will use these for error bars
    if not exp_list: return
    exp_groups = {}

    for name, label in exp_list:
        exp_groups.setdefault(label, []).append(name)

    for label in exp_groups:

        indep_runs = exp_groups[label]

        # here get all the prediction folders
        for exp in indep_runs:
            prediction_folders = [x for x in os.listdir(os.path.join(out_root,exp)) if 'predictions' in x]

        print 'indep runs', indep_runs
        print 'prediction_folders', prediction_folders

        for prediction_folder in prediction_folders:

            indep_run_data = [[float(x) for x in read_div_file(os.path.join(*[out_root,exp,prediction_folder,'gt_divergence.log']))[dataset]] for exp in indep_runs]

            if two_seeds:
                min_length = min(len(x) for x in indep_run_data)
                min_length_index = -1
                for i in range(len(indep_run_data)):
                    if len(indep_run_data[i]) == min_length:
                        min_length_index = i
                        break
                if len(indep_run_data) > 2:
                    indep_run_data = [indep_run_data[i] for i in range(len(indep_run_data)) if i != min_length_index]

            # trim to the minimum length
            min_length = min(len(x) for x in indep_run_data)
            min_length = 51
            indep_run_data = np.array([x[:min_length] for x in indep_run_data])  # (num_seeds, min_length)

            print label, indep_run_data, min_length

            # compute max min and average
            maxs = np.max(indep_run_data,0)
            mins = np.min(indep_run_data,0)
            means = np.mean(indep_run_data,0)

            x = range(min_length) # TODO
            if label == 'Balls Generalization':

                print 'x',x, len(x)
                print 'means',means, len(means)
                print 'mins',mins, len(mins)
                print 'maxs',maxs, len(maxs)

            # custom_plot(x, means, mins, maxs, label=label + ' ' + find_num_obj_in_substring(prediction_folder) + ' objects', marker=marker.next())

            # if ',' in indep_runs[0]:
            #     custom_plot(x, means, mins, maxs, label=label + find_num_obj_in_substring(prediction_folder), marker=marker.next())
            # else:
            custom_plot(x, means, mins, maxs, label=label + find_num_obj_in_substring_single(prediction_folder), marker=marker.next())

            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # if any('dataset' in x for x in indep_runs):
            #     ax.set_ylim(-8,-1)
            # else:
            #     ax.set_ylim(-4, -1)
            # if 
            # if dataset == 'Cosine Difference':
            #     ax.set_ylim(0.8,1)
            # elif dataset == 'Magnitude Difference':
            #     ax.set_ylim(0.0,0.2)

            
    # leg = plt.legend(fontsize=20, frameon=False)
    plt.xlabel('Timesteps')
    if dataset =='Cosine Difference':
        plt.ylabel('Cosine Distance')  # TODO!
        leg = plt.legend(fontsize=14, frameon=False)
    elif dataset == 'Magnitude Difference':
        plt.ylabel('Relative Error in Magnitude')  # TODO!
        leg = plt.legend(fontsize=14, frameon=False, loc='upper left')
    # elif dataset == 'avg_rel_mag_error':
    #     plt.ylabel('Relative Error in Magnitude')  # TODO!
    #     leg = plt.legend(fontsize=20, frameon=False)
    # elif dataset == 'avg_rel_mag_error':
    #     plt.ylabel('Relative Error in Magnitude')  # TODO!
    #     leg = plt.legend(fontsize=20, frameon=False)
    plt.savefig(os.path.join(outfolder, outfile))
    plt.close()

def find_num_obj_in_substring(substring):
    num_objs = []
    for m in re.finditer('_n', substring):
        begin = m.end()
        end = begin + substring[m.end():].find('_')
        num_objs.append(substring[begin:end]) 
        return ','.join(num_objs)

def find_num_obj_in_substring_single(substring):
    begin = substring.find('_n') + len('_n')
    end = begin + substring[begin:].find('_')
    return substring[begin:end]

def extract_num_objs(exp_name):
    train_half = exp_name[:exp_name.find('__')]
    test_half = exp_name[exp_name.find('__'):exp_name.find('_layers')]

    return {'train': find_num_obj_in_substring(train_half), 'val': find_num_obj_in_substring(train_half), 'test': find_num_obj_in_substring(test_half)}

def plot_generalization_error(exp_list, outfolder, outfile,two_seeds):
    ys = []
    xs = []

    # fig, ax = plt.subplots()
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    # marker = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')) 

    # first group all the same labels together. You will use these for error bars
    if not exp_list: return
    exp_groups = {}

    for name, label in exp_list:
        exp_groups.setdefault(label, []).append(name)

    for label in exp_groups:

        fig, ax = plt.subplots()
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        marker = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')) 

        indep_runs = exp_groups[label]
        num_obj_dict = extract_num_objs(indep_runs[0])

        for dataset in ['train', 'val', 'test']:


            indep_run_data = [[float(x) for x in read_log_file(os.path.join(*[out_root,exp,'experiment.log']))[dataset]] for exp in indep_runs]

            if two_seeds:
                min_length = min(len(x) for x in indep_run_data)
                min_length_index = -1
                for i in range(len(indep_run_data)):
                    if len(indep_run_data[i]) == min_length:
                        min_length_index = i
                        break
                indep_run_data = [indep_run_data[i] for i in range(len(indep_run_data)) if i != min_length_index]

            # trim to the minimum length
            min_length = min(len(x) for x in indep_run_data)
            indep_run_data = np.array([x[:min_length] for x in indep_run_data])  # (num_seeds, min_length)

            # convert it from ln scale
            indep_run_data = np.exp(indep_run_data)

            # convert it to log base 10 scale
            indep_run_data = np.log10(indep_run_data)

            print label, indep_run_data, min_length

            # compute max min and average
            maxs = np.max(indep_run_data,0)
            mins = np.min(indep_run_data,0)
            means = np.mean(indep_run_data,0)

            x = range(min_length) # TODO

            custom_plot(x, means, mins, maxs, label=dataset+': '+num_obj_dict[dataset]+' objects', marker=marker.next())
            # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if any('tower' in x for x in indep_runs):
                ax.set_ylim(-8,-1)
            else:
                ax.set_ylim(-4, -1)


        leg = plt.legend(fontsize=16, frameon=False)
        plt.xlabel('Iterations (x 100000)')
        plt.ylabel('Mean Squared Error')  # TODO!
        plt.savefig(os.path.join(outfolder, label+'_'+outfile))
        plt.close()

def plot_inf_error(exp_list, dataset, outfolder, outfile,two_seeds):
    # print exp_list

    ys = []
    xs = []

    fig, ax = plt.subplots()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    marker = itertools.cycle(('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd')) 

    # first group all the same labels together. You will use these for error bars
    exp_groups_orig = {}

    for name, label in exp_list:
        exp_groups_orig.setdefault(label, []).append(name)

    exp_groups = {}

    for label in exp_groups_orig:
        # for name in exp_groups_orig[label]:
        #     print os.path.join(*[out_root,name,dataset+'_infer_cf.log'])
        if any(os.path.exists(os.path.join(*[out_root,name,dataset+'_infer_cf.log'])) for name in exp_groups_orig[label]):
            exp_groups[label] = exp_groups_orig[label]

    print 'Experiments with ' + dataset + ' inference', exp_groups
    if not exp_groups:
        return

    for label in exp_groups:
        indep_runs = exp_groups[label]

        indep_run_data = [[float(x) for x in read_inf_log_file(os.path.join(*[out_root,exp,dataset+'_infer_cf.log']))[dataset]] for exp in indep_runs]

        if two_seeds:
            min_length = min(len(x) for x in indep_run_data)
            min_length_index = -1
            for i in range(len(indep_run_data)):
                if len(indep_run_data[i]) == min_length:
                    min_length_index = i
                    break
            indep_run_data = [indep_run_data[i] for i in range(len(indep_run_data)) if i != min_length_index]

        # trim to the minimum length
        min_length = min(len(x) for x in indep_run_data)
        indep_run_data = np.array([x[:min_length] for x in indep_run_data])  # (num_seeds, min_length)

        print label, indep_run_data, min_length

        # compute max min and average
        maxs = np.max(indep_run_data,0)
        mins = np.min(indep_run_data,0)
        means = np.mean(indep_run_data,0)

        x = range(1,min_length+1) # TODO

        custom_plot(x, means, mins, maxs, label=label, marker=marker.next())
        ax.set_ylim(0.3, 1)
        ax.set_xlim(1, 12)

    leg = plt.legend(fontsize=14, frameon=False, loc='lower right')
    plt.xlabel('Iterations (x 100000)')
    plt.ylabel('Accuracy')  # TODO!
    plt.savefig(os.path.join(outfolder, outfile))
    plt.close()




def plot_experiments(experiments_dict, two_seeds):
    for e in experiments_dict:
        print 'Plotting', e
        # plot_experiment(experiments_dict[e], 'test', out_root, e+'.png')
        # plot_experiment_error(experiments_dict[e], 'test', out_root, e+'_rda.png',two_seeds)
        # plot_inf_error([exp for exp in experiments_dict[e] if '_m_' in exp[0]], 'mass', out_root, e+'_mass_inference_rda.png',two_seeds)
        # plot_generalization_error([exp for exp in experiments_dict[e] if ',' in exp[0]], out_root, e+'_gen.png',two_seeds)

        # plot_tva_error([exp for exp in experiments_dict[e] if 'balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0]) ], 'avg_ang_error', out_root, e+'_angle.png', two_seeds)
        # plot_tva_error([exp for exp in experiments_dict[e] if 'balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0])], 'avg_rel_mag_error', out_root, e+'_mag.png', two_seeds)

        plot_tva_error([exp for exp in experiments_dict[e] if 'balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0]) ], 'avg_ang_error', out_root, e+'_angle.png', two_seeds)
        plot_tva_error([exp for exp in experiments_dict[e] if 'balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0])], 'avg_rel_mag_error', out_root, e+'_mag.png', two_seeds)

        # plot_div_error([exp for exp in experiments_dict[e] if 'balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0]) ], 'Cosine Difference', out_root, e+'_anglesim.png', two_seeds)
        # plot_div_error([exp for exp in experiments_dict[e] if 'balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0])], 'Magnitude Difference', out_root, e+'_magsim.png', two_seeds)

        # plot_div_error([exp for exp in experiments_dict[e] if 'balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0]) ], 'Cosine Difference', out_root, e+'_anglesim.png', two_seeds)
        # plot_div_error([exp for exp in experiments_dict[e] if 'balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0])], 'Magnitude Difference', out_root, e+'_magsim.png', two_seeds)

        # plot_div_error([exp for exp in experiments_dict[e] if '_rda__balls' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0]) ], 'Cosine Difference', out_root, e+'_anglesimb.png', two_seeds)
        # plot_div_error([exp for exp in experiments_dict[e] if '_rda__balls' in exp[0] and ('modelnp' in exp[0] or 'modelbffobj' in exp[0])], 'Magnitude Difference', out_root, e+'_magsimb.png', two_seeds)




# balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda

        # plot_tva_error([exp for exp in experiments_dict[e] if 'mixed_n6_t60_ex50000_m_z_o_dras3_rda__' in exp[0]], 'avg_ang_error', out_root, e+'_angle.png', two_seeds)
        # plot_tva_error([exp for exp in experiments_dict[e] if 'mixed_n6_t60_ex50000_m_z_o_dras3_rda__' in exp[0]], 'avg_rel_mag_error', out_root, e+'_mag.png', two_seeds)
        # plot_tva_error([exp for exp in experiments_dict[e] if 'rda__tower' in exp[0]], 'vel_loss', out_root, e+'_vel.png', two_seeds)
        # plot_tva_error([exp for exp in experiments_dict[e] if 'rda__tower' in exp[0]], 'ang_vel_loss', out_root, e+'_av.png', two_seeds)




# Call Demo_minimal here
def visualize(experiments):
    print('## VISUALIZE ##')
    for experiment_folder in experiments:
        try:
            experiment_folder = os.path.join(out_root, experiment_folder)
            if any('predictions' in x for x in os.listdir(experiment_folder)):
                prediction_folders = [x for x in os.listdir(experiment_folder) if 'predictions' in x]
                # print prediction_folders
                # assert(len(prediction_folders)==1)
                # prediction_folder = prediction_folders[0]
                for prediction_folder in prediction_folders:
                    for batch in [x for x in os.listdir(os.path.join(experiment_folder, prediction_folder)) if 'batch' in x]:
                        mkdir_p(os.path.join(*[experiment_folder,'visual',os.path.splitext(batch)[0]]))
                        # print os.path.splitext(batch)[0]


                    command = 'node ' + js_root + '/Demo_minimal.js -e ' + os.path.join(experiment_folder, prediction_folder)  # maybe I need to do this in callback? If I do one it should work, but more than that I don't know.
                    # print '#'*80
                    print(command)
                    os.system(command)
                    print '#'*80

                # we could make the gif now.
        except KeyboardInterrupt:
            sys.exit(0)

# Call Demo_minimal here
def tower_stability(experiments):

    print('## TOWER STABILITY ##')
    for experiment_folder in experiments:
        experiment_folder = os.path.join(out_root, experiment_folder)
        if any('predictions' in x for x in os.listdir(experiment_folder)):
            prediction_folders = [x for x in os.listdir(experiment_folder) if 'predictions' in x]
            # assert(len(prediction_folders)==1)
            prediction_folder = prediction_folders[0]  # WILL NOT BE TRUE WHEN YOU DO GENERALIZAION!
            command = 'node ' + js_root + '/Demo_minimal.js -i -e ' + os.path.join(experiment_folder, prediction_folder)  # maybe I need to do this in callback? If I do one it should work, but more than that I don't know.
            print(command)
            os.system(command)
            print '#'*80

            # you need to get the subfolders now
            visual_folder = os.path.join(*[out_root, experiment_folder, 'visual'])
            batch_folders = []
            for batch_folder in [x for x in os.listdir(visual_folder) if os.path.isdir(os.path.join(visual_folder,x))]:
                batch_folder = os.path.join(visual_folder, batch_folder)
                batch_folders.append(batch_folder)

            # let's group the gt together and the pred together
            gt_batch_folders = sorted([f for f in batch_folders if 'gt' in f])
            pred_batch_folders = sorted([f for f in batch_folders if 'pred' in f])

            # make sure the batches correspond




            # assert False


            stability_stats_all = {}

            # get a list of batch_exs by looking at any stability_stats
            example_stability_stats_file = os.path.join(*[experiment_folder,'visual', gt_batch_folders[0], 'stability_stats.json'])

            # actually get a dictionary: batch_ex: {gt: gt_frac_stable, pred: pred_frac_stable}
            for i in range(len(gt_batch_folders)):
                print(len(gt_batch_folders))
                print(len(pred_batch_folders))

                gt_batch_folder_stability_stats_file = os.path.join(*[experiment_folder,'visual', gt_batch_folders[i], 'stability_stats.json'])
                pred_batch_folder_stability_stats_file = os.path.join(*[experiment_folder,'visual', pred_batch_folders[i], 'stability_stats.json'])


                gt_batch_folder_stability_stats_data = json.load(open(gt_batch_folder_stability_stats_file,'r'))
                pred_batch_folder_stability_stats_data = json.load(open(pred_batch_folder_stability_stats_file,'r'))

                gt_batch_exs = sorted([x[len('gt_'):] for x in gt_batch_folder_stability_stats_data.keys()])
                pred_batch_exs = sorted([x[len('pred_'):] for x in pred_batch_folder_stability_stats_data.keys()])

                assert(gt_batch_exs == pred_batch_exs)

                for be in gt_batch_exs:
                    print(gt_batch_folder_stability_stats_data)
                    print(be)
                    stability_stats_all[be] = {'gt': gt_batch_folder_stability_stats_data['gt_'+be]['frac_unstable'], 'pred': pred_batch_folder_stability_stats_data['pred_'+be]['frac_unstable']}

            pprint.pprint(stability_stats_all)

            # now get a list of the keys and get a numpy matrix of the frac_stable
            # is it frac_stable or frad_unstable?

            batch_exs = stability_stats_all.keys()
            gt_frac_unstables = [stability_stats_all[be]['gt'] for be in batch_exs]
            pred_frac_unstables = [stability_stats_all[be]['pred'] for be in batch_exs]

            # this is all you need to plot
            print batch_exs
            print gt_frac_unstables
            print pred_frac_unstables


def img_id_json(filename):
    begin = filename.rfind('step')+len('step')
    end = filename.rfind('.')
    return int(filename[begin:end])

def ex_json(fn):
    begin = fn.find('_ex')+len('_ex')
    end = begin + fn[begin:].find('_')
    ex = int(fn[begin:end])
    return ex

def create_gif_json(images_root, gifname, stability_stats=None):
    """
        writeGif(filename, images, duration=0.1, loops=0, dither=1)
            Write an animated gif from the specified images.
            images should be a list of numpy arrays of PIL images.
            Numpy images of type float should have pixels between 0 and 1.
            Numpy images of other types are expected to have values between 0 and 255.
    """
    # TODO! I'm assuming my img_id is fixed!
    # first group by ex
    exs = {}
    for fn in [fn for fn in os.listdir(images_root) if fn.endswith('.png') and 'overlay' not in fn]:
        ex =ex_json(fn)
        if ex not in exs: exs[ex] = []
        exs[ex].append(fn)

    if stability_stats:
        # sort stability_stats
        sorted_stability_stats = sorted(stability_stats.items(), key=lambda x:x[1])

    for ex in exs:
        exs[ex] = sorted(exs[ex], key=lambda x: img_id_json(x))
        # stability_stats = None
        if stability_stats:
            print(sorted_stability_stats)
            key = [i for i in range(len(sorted_stability_stats)) if 'ex'+str(ex) in sorted_stability_stats[i][0]][0]
            # print(key)
            # key = [k for k in sorted_stability_stats if 'ex'+str(ex) in k][0]
            gifname_ex = gifname[:gifname.rfind('.gif')]+ '_rank' + str(key)+'_ex'+str(ex) +'_top-block-displacement_' + str(sorted_stability_stats[key][1])  + '.gif'
        else:
            gifname_ex = gifname[:gifname.rfind('.gif')]+'_ex'+str(ex)+'.gif'
        create_gif_json_ex(images_root, exs[ex], gifname_ex)

def create_gif_json_ex(images_root, file_names, gifname):
    images = [Image.open(os.path.join(images_root,fn)) for fn in file_names]
    filename = os.path.join(images_root, gifname)
    writeGif(filename, images, duration=0.001)

def overlay_imgs(images_root, batch_name, subsample):
    # assert False, "Did you incorporate the ex numbers?"
    file_names = sorted([fn for fn in os.listdir(images_root) if fn.endswith('.png') and 'overlay' not in fn], key=lambda x: img_id_json(x))
    # here I group by example number, since we already in a batch folder
    exs = {}

    for file_name in file_names:
        # extract example number here
        exs.setdefault(ex_json(file_name), []).append(file_name)

    for ex in exs:
        sorted_filenames_for_this_ex = sorted(exs[ex], key=lambda x: img_id_json(x))

        images = [Image.open(os.path.join(images_root,fn)) for fn in sorted_filenames_for_this_ex] 
        filename = os.path.join(images_root, batch_name+'_ex' + str(ex) + '_overlay.png')
        result = images[0]
        samples = range(subsample,len(images)/2, subsample)
        for i in range(len(samples)):
            next_img = images[samples[i]]  # for some reason next_img is not skipping every 5?
            result = Image.blend(result, next_img, 0.35)
            result = Image.blend(result, next_img, 0.1)
        result = Image.blend(result, next_img, 0.1)

        result.save(filename,"PNG")
        print 'Saved overlay to',filename


def animate(experiments, remove_png):
    print('## ANIMATE ##')
    # wait until all of Demo_minimal has finished
    animated_experiments = []
    for experiment_folder in experiments:
        print '#'*80
        print 'Trying to animate', experiment_folder
        visual_folder = os.path.join(*[out_root, experiment_folder, 'visual'])
        if not os.listdir(visual_folder): 
            print 'Nothing in', visual_folder
        else:
            animated_experiments.append(experiment_folder)
            for batch_folder in [x for x in os.listdir(visual_folder) if os.path.isdir(os.path.join(visual_folder,x))]:
                print '-'*80
                batch_name = experiment_folder + '_' + batch_folder
                gifname = batch_name + '.gif'
                # overlayed_name = experiment_folder + '_' + batch_folder + '_overlay.png'
                batch_folder = os.path.join(visual_folder, batch_folder)
                if any(f.endswith('.png') for f in os.listdir(batch_folder)):
                    create_gif_json(batch_folder, gifname)
                    overlay_imgs(batch_folder, batch_name, 5)

                    if remove_png:
                        print 'Removing images from', batch_folder
                        for imgfile in [x for x in os.listdir(batch_folder) if x.endswith('.png') and 'overlay' not in x]:
                            imgfile = os.path.join(batch_folder, imgfile)
                            command = 'rm ' + imgfile
                            os.system(command)
                else:
                    print 'No .pngs found. Not creating gif for', batch_folder

    print 'Animated the following folders:'
    pprint.pprint(animated_experiments)


def animate_tower(experiments, remove_png):
    print('## ANIMATE TOWER##')
    # wait until all of Demo_minimal has finished
    animated_experiments = []
    for experiment_folder in experiments:
        print '#'*80
        print 'Trying to animate', experiment_folder
        visual_folder = os.path.join(*[out_root, experiment_folder, 'visual'])
        if not os.listdir(visual_folder): 
            print 'Nothing in', visual_folder
        else:
            animated_experiments.append(experiment_folder)
            for batch_folder in [x for x in os.listdir(visual_folder) if os.path.isdir(os.path.join(visual_folder,x))]:

                print '-'*80
                batch_name = experiment_folder + '_' + batch_folder
                gifname = batch_name + '.gif'
                # overlayed_name = experiment_folder + '_' + batch_folder + '_overlay.png'
                batch_folder = os.path.join(visual_folder, batch_folder)

                # get the stats
                stability_stats = json.loads(open(os.path.join(batch_folder,'stability_stats.json'),'r').read().strip())
                print stability_stats


                if any(f.endswith('.png') for f in os.listdir(batch_folder)):
                    create_gif_json(batch_folder, gifname, stability_stats)
                    overlay_imgs(batch_folder, batch_name, 5)

                    if remove_png:
                        print 'Removing images from', batch_folder
                        for imgfile in [x for x in os.listdir(batch_folder) if x.endswith('.png') and 'overlay' not in x]:
                            imgfile = os.path.join(batch_folder, imgfile)
                            command = 'rm ' + imgfile
                            os.system(command)
                else:
                    print 'No .pngs found. Not creating gif for', batch_folder

    print 'Animated the following folders:'
    pprint.pprint(animated_experiments)


# experiments_to_plot = copy(experiments)  # returns a list of experiments that changed

# experiments_to_plot = experiments
# plot(experiments_to_plot)
plot_experiments(experiments_dict, False)
# 
# visualize(experiments_to_visualize)
# tower_stability(experiments_to_visualize)
# animate(experiments_to_visualize, False)
# animate_tower(experiments_to_visualize, False)


# Balls Pred
# opmjlogs/balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0/visual/gt_batch335/balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0_gt_batch335_ex50000_overlay.png
# open opmjlogs/balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0/visual/pred_batch335/balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0_pred_batch335_ex50000_overlay.png




