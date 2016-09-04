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
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'No Pairwise No Lookahead'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'No Pairwise No Lookahead'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'No Pairwise No Lookahead'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE No Lookahead'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE No Lookahead'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rda__mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE No Lookahead'),
    ],

    'Mixed Generalization': [
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'No Pairwise No Lookahead'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'No Pairwise No Lookahead'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'No Pairwise No Lookahead'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE No Lookahead'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE No Lookahead'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rda,mixed_n4_t60_ex50000_z_o_dras3_rda__mixed_n5_t60_ex50000_z_o_dras3_rda,mixed_n6_t60_ex50000_z_o_dras3_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE No Lookahead'),
    ], 

    'Balls Prediction': [
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'No Pairwise No Lookahead'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'No Pairwise No Lookahead'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'No Pairwise No Lookahead'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE No Lookahead'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE No Lookahead'),
        ('balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE No Lookahead'),
    ],

    'Balls Prediction Mass': [
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'No Pairwise No Lookahead'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'No Pairwise No Lookahead'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'No Pairwise No Lookahead'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE No Lookahead'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE No Lookahead'),
        ('balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE No Lookahead'),
    ],

    'Balls Generalization': [
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'No Pairwise No Lookahead'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'No Pairwise No Lookahead'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'No Pairwise No Lookahead'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE No Lookahead'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE No Lookahead'),
        ('balls_n3_t60_ex50000_rda,balls_n4_t60_ex50000_rda,balls_n5_t60_ex50000_rda__balls_n6_t60_ex50000_rda,balls_n7_t60_ex50000_rda,balls_n8_t60_ex50000_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE No Lookahead'),

    ],

    'Balls Generalization Mass': [
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_fast_seed0_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_fast_seed1_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers3_rs_fast_seed2_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed0_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed1_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed1_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed2_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_rs_fast_seed2_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed0', 'No Pairwise No Lookahead'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed1', 'No Pairwise No Lookahead'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelnp_seed2', 'No Pairwise No Lookahead'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed0', 'NPE No Lookahead'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed1', 'NPE No Lookahead'),
        ('balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_rs_fast_nlan_lr0.0003_modelbffobj_seed2', 'NPE No Lookahead'),

    ]


}

experiments = list(set(itertools.chain.from_iterable([[x[0] for x in y] for y in experiments_dict.values()])))

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
    experiments_to_visualize = []  # TODO

    print 'experiments to plot'
    pprint.pprint(experiments_to_plot)
    print 'experiments to visualize'
    pprint.pprint(experiments_to_visualize)

    return experiments_to_plot, []

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

    plt.legend(fontsize=8)
    plt.xlabel('Iterations (x 10000)')
    plt.ylabel('Log MSE Loss')  # TODO!
    plt.savefig(os.path.join(outfolder, outfile))
    plt.close()

def custom_plot(x, means, mins, maxs, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, means, **kwargs)
    ax.fill_between(x, mins, maxs, facecolor=base_line.get_color(), alpha=0.5, linewidth=0.0)


def plot_experiment_error(exp_list, dataset, outfolder, outfile):
    ys = []
    xs = []

    fig, ax = plt.subplots()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

    # first group all the same labels together. You will use these for error bars
    exp_groups = {}

    for name, label in exp_list:
        exp_groups.setdefault(label, []).append(name)

    for label in exp_groups:
        indep_runs = exp_groups[label]

        indep_run_data = [[float(x) for x in read_log_file(os.path.join(*[out_root,exp,'experiment.log']))[dataset]] for exp in indep_runs]

        # trim to the minimum length
        min_length = min(len(x) for x in indep_run_data)
        indep_run_data = np.array([x[:min_length] for x in indep_run_data])  # (num_seeds, min_length)

        print label, indep_run_data, min_length

        # compute max min and average
        maxs = np.max(indep_run_data,0)
        mins = np.min(indep_run_data,0)
        means = np.mean(indep_run_data,0)

        x = range(min_length) # TODO

        custom_plot(x, means, mins, maxs, label=label)

    plt.legend(fontsize=8)
    plt.xlabel('Iterations (x 100000)')
    plt.ylabel('Log MSE Loss')  # TODO!
    plt.savefig(os.path.join(outfolder, outfile))
    plt.close()




def plot_experiments(experiments_dict):
    for e in experiments_dict:
        print 'Plotting', e
        # plot_experiment(experiments_dict[e], 'test', out_root, e+'.png')
        plot_experiment_error(experiments_dict[e], 'test', out_root, e+'_rda.png')


# Call Demo_minimal here
def visualize(experiments):
    print('## VISUALIZE ##')
    for experiment_folder in experiments:
        try:
            experiment_folder = os.path.join(out_root, experiment_folder)
            if any('predictions' in x for x in os.listdir(experiment_folder)):
                prediction_folders = [x for x in os.listdir(experiment_folder) if 'predictions' in x]
                assert(len(prediction_folders)==1)
                prediction_folder = prediction_folders[0]
                command = 'node ' + js_root + '/Demo_minimal.js -e ' + os.path.join(experiment_folder, prediction_folder)  # maybe I need to do this in callback? If I do one it should work, but more than that I don't know.
                print '#'*80
                print(command)
                os.system(command)

                # we could make the gif now.
        except KeyboardInterrupt:
            sys.exit(0)

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

def overlay_imgs(images_root, overlayedname):
    assert False, "Did you incorporate the ex numbers?"
    file_names = sorted([fn for fn in os.listdir(images_root) if fn.endswith('.png') and 'overlay' not in fn], key=lambda x: img_id_json(x))
    images = [Image.open(os.path.join(images_root,fn)) for fn in file_names] 
    filename = os.path.join(images_root, overlayedname)
    result = images[0]
    # unit = 1/(2*len(images))
    for i in range(1,len(images)):
        next_img = images[i]
        result = Image.blend(result, next_img, 0.5)
    result.save(filename,"PNG")


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
                gifname = experiment_folder + '_' + batch_folder + '.gif'
                overlayed_name = experiment_folder + '_' + batch_folder + '_overlay.png'
                batch_folder = os.path.join(visual_folder, batch_folder)
                if any(f.endswith('.png') for f in os.listdir(batch_folder)):
                    create_gif_json(batch_folder, gifname)
                    # overlay_imgs(batch_folder, overlayed_name)

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
                gifname = experiment_folder + '_' + batch_folder + '.gif'
                overlayed_name = experiment_folder + '_' + batch_folder + '_overlay.png'
                batch_folder = os.path.join(visual_folder, batch_folder)

                # get the stats
                stability_stats = json.loads(open(os.path.join(batch_folder,'stability_stats.json'),'r').read().strip())
                print stability_stats


                if any(f.endswith('.png') for f in os.listdir(batch_folder)):
                    create_gif_json(batch_folder, gifname, stability_stats)
                    # overlay_imgs(batch_folder, overlayed_name)

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


experiments_to_plot, experiments_to_visualize = copy(experiments)  # returns a list of experiments that changed

# experiments_to_visualize = [
#     'tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda100_modelbffobj_lambda100_batch_norm',
#     # 'tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda100_modelbffobj_lambda100_batch_norm',
#     # 'tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda10_modelbffobj_lambda10_batch_norm',
#     # 'tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda10_modelbffobj_lambda10_batch_norm',
#     # 'balls_n6_t60_ex50000_rd__balls_n6_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
# ]

plot(experiments_to_plot)
plot_experiments(experiments_dict)

# TODO: visualize only if epxerimtns_to_visualize says so.

# visualize(experiments_to_visualize)
# animate(experiments_to_visualize, False)
# animate_tower(experiments_to_visualize, True)





