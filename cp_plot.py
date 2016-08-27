import os
import sys
import plot_results
import errno   

import cv2
import os
import numpy as np
from images2gif import writeGif
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import pprint 


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

experiments = [
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_lrdecay_every10000',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_lrdecay_every5000',
                # 'balls_n2_t60_ex50000__balls_n2_t60_ex50000_batchnorm',
                # 'balls_n5_t60_ex50000__balls_n5_t60_ex50000_lrdecay_every5000',

                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_lrdecay_every2500',
                # 'balls_n5_t60_ex50000__balls_n5_t60_ex50000_lrdecay_every2500',
                # 'balls_n6_t60_ex50000__balls_n6_t60_ex50000_lrdecay_every2500',
                # 'balls_n7_t60_ex50000__balls_n7_t60_ex50000_lrdecay_every2500',
                # 'balls_n8_t60_ex50000__balls_n8_t60_ex50000_lrdecay_every2500',
                # 'balls_n9_t60_ex50000__balls_n9_t60_ex50000_lrdecay_every2500',
                # 'balls_n10_t60_ex50000__balls_n10_t60_ex50000_lrdecay_every2500',


                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_modelind',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_modelcat_lr3e-5',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_modelcat_lr3-e5_lineardecoder',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_modelind_lineardecoder',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000__balls_n5_t60_ex50000',
                # 'balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n3_t60_ex50000',
                # 'balls_n5_t60_ex50000,balls_n3_t60_ex50000__balls_n4_t60_ex50000',
                # 'balls_n3_t60_ex50000_m__balls_n3_t60_ex50000_m',
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

                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past10',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past9',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past8',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past7',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past6',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past5',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past4',
                # 'balls_n3_t60_ex50000__balls_n3_t60_ex50000_num_past3',

                # all of the above have been plotted after the experiments finished or ran out of time. Possibly not simulated.

                # bffobj initial test
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers2_nbrhd_lr0.0003_modelbffobj',  
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',  


                # ffobj with nbrhd initial test'
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers2_nbrhd_lr0.0003_modelffobj',
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers2_nbrhd_lr0.001_modelffobj',
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers3_nbrhd_lr0.0003_modelffobj',
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers3_nbrhd_lr0.001_modelffobj',

                # bffobj generalization test
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000__balls_n5_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n3_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',
                # 'balls_n5_t60_ex50000,balls_n3_t60_ex50000__balls_n4_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',

                # generalization test 2
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n7_t60_ex50000_layers3_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000_layers3_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n8_t60_ex50000_layers3_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000,balls_n7_t60_ex50000_layers3_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000,balls_n8_t60_ex50000_layers3_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000,balls_n7_t60_ex50000,balls_n8_t60_ex50000_layers3_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n7_t60_ex50000,balls_n8_t60_ex50000_layers3_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n7_t60_ex50000_layers3_nbrhd_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000_layers3_nbrhd_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n8_t60_ex50000_layers3_nbrhd_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n7_t60_ex50000,balls_n8_t60_ex50000_layers3_nbrhd_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000,balls_n7_t60_ex50000_layers3_nbrhd_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000,balls_n7_t60_ex50000,balls_n8_t60_ex50000_layers3_nbrhd_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000,balls_n8_t60_ex50000_layers3_nbrhd_lr0.0003_modelffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n7_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n8_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000,balls_n7_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000,balls_n8_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n7_t60_ex50000,balls_n8_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',
                # 'balls_n3_t60_ex50000,balls_n4_t60_ex50000,balls_n5_t60_ex50000__balls_n6_t60_ex50000,balls_n7_t60_ex50000,balls_n8_t60_ex50000_layers3_nbrhd_lr0.0003_modelbffobj',
                

                # bffobj experiments with ACTUAL nbhrd
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers3_nbrhd_nbrhdsize4_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers3_nbrhd_nbrhdsize1.5_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers3_nbrhd_nbrhdsize3_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000__balls_n4_t60_ex50000_layers3_nbrhd_nbrhdsize4.5_lr0.0003_modelbffobj',

                # testing generaliazation, but not true rd
                # 'balls_n4_t120_ex25000_m,balls_n5_t120_ex25000_m__balls_n4_t120_ex25000_m,balls_n5_t120_ex25000_m_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                # 'balls_n4_t120_ex25000_m,balls_n5_t120_ex25000_m__balls_n4_t120_ex25000_m,balls_n5_t120_ex25000_m_layers3_nbrhd_nbrhdsize4.5_lr0.0003_modelbffobj',
                # 'balls_n4_t120_ex25000_m,balls_n5_t120_ex25000_m__balls_n4_t120_ex25000_m,balls_n5_t120_ex25000_m_layers3_nbrhd_nbrhdsize4_lr0.0003_modelbffobj',

                # RD dataset
                # 'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers2_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers4_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers2_nbrhd_nbrhdsize3.5_lr0.0001_modelbffobj',
                # 'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers4_nbrhd_nbrhdsize3.5_lr0.0001_modelbffobj',
                # 'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0001_modelbffobj',
                # 'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                # 'balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers2_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers2_nbrhd_nbrhdsize3.5_lr0.0001_modelbffobj',
                # 'balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0001_modelbffobj',
                # 'balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                # 'balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_nbrhd_nbrhdsize3.5_lr0.0001_modelbffobj',

                # test im, but this includes 1e30. so just testing nbrhd size here
                # 'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj',
                # 'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize4_lr0.0003_im_modelbffobj',
                # 'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3_lr0.0003_im_modelbffobj',

                # RD dataset with 2.5 buffer
                'balls_n3_t60_ex50000_rd__balls_n3_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'balls_n5_t60_ex50000_rd__balls_n5_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'balls_n6_t60_ex50000_rd__balls_n6_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'balls_n7_t60_ex50000_rd__balls_n7_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'balls_n8_t60_ex50000_rd__balls_n8_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj',
                'balls_n3_t60_ex50000_m_rd__balls_n3_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj',
                'balls_n5_t60_ex50000_m_rd__balls_n5_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj',
                'balls_n6_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj',
                'balls_n7_t60_ex50000_m_rd__balls_n7_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj',
                'balls_n8_t60_ex50000_m_rd__balls_n8_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj',

                # try different models for mass (did not do mass inference here)
                'balls_n6_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd_layers2_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'balls_n6_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd_layers4_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',


                # tower 6 blocks (variance for 10 blocks though)
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers2_nbrhd_nbrhdsize3_lr0.0003_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers4_nbrhd_nbrhdsize3_lr0.0003_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers2_nbrhd_nbrhdsize4_lr0.0003_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers4_nbrhd_nbrhdsize4_lr0.0003_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0001_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.001_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr3e-05_val_eps0_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr3e-06_val_eps0_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr3e-07_val_eps0_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.003_val_eps0_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.03_val_eps0_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.3_val_eps0_modelbffobj',

                'tower_n8_t120_ex25000_rd__tower_n8_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                'tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj',
                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj',
                'tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj',

                # tower debugging
                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_vlambda1000_modelbffobj_lambda1000_batch_norm',
                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_vlambda1000_modelbffobj_lambda1000',

                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_vlambda0_modelbffobj_lambda1',

                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_vlambda1000_modelbffobj',
                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_vlambda100_modelbffobj',
                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_vlambda10_modelbffobj',


                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj_lambda1000',
                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj_lambda100',
                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj_lambda10',

                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj_batch_norm',

                'tower_n4_t120_ex25000_rd_stable__tower_n4_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj',
                'tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj',
                'tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj',
                'tower_n6_t120_ex25000_rd__tower_n6_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj',

                'tower_n2_t120_ex25000_rd_stable__tower_n2_t120_ex25000_rd_stable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps0_modelbffobj',






                # for towers: try batch norm

                # for balls: try doing inference on only the examples where there is a collision. But how to decide? Perhaps you can say if the reversal is > 90 degree then it is a collision? You can also test by angle
                
                # balls generalization
                'balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers2_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers4_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                
                'balls_n3_t60_ex50000_m_rd,balls_n4_t60_ex50000_m_rd,balls_n5_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd,balls_n7_t60_ex50000_m_rd,balls_n8_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                # mixed
                'mixed_n6_t60_ex50000_rd__mixed_n6_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n6_t60_ex50000_z_rd__mixed_n6_t60_ex50000_z_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n6_t60_ex50000_o_rd__mixed_n6_t60_ex50000_o_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n6_t60_ex50000_z_o_rd__mixed_n6_t60_ex50000_z_o_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                # debug mixed
                'mixed_n6_t60_ex50000_z_dras_rd__mixed_n6_t60_ex50000_z_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                # invisible
                'invisible_n5_t60_ex50000_z_o_dras_rd__invisible_n5_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'invisible_n6_t60_ex50000_z_o_dras_rd__invisible_n6_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                # mixed dras
                'mixed_n3_t60_ex50000_m_z_o_dras_rd__mixed_n3_t60_ex50000_m_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n4_t60_ex50000_m_z_o_dras_rd__mixed_n4_t60_ex50000_m_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n5_t60_ex50000_m_z_o_dras_rd__mixed_n5_t60_ex50000_m_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n6_t60_ex50000_m_z_o_dras_rd__mixed_n6_t60_ex50000_m_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                'mixed_n3_t60_ex50000_z_o_dras_rd__mixed_n3_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n4_t60_ex50000_z_o_dras_rd__mixed_n4_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n5_t60_ex50000_z_o_dras_rd__mixed_n5_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n6_t60_ex50000_z_o_dras_rd__mixed_n6_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                'mixed_n3_t60_ex50000_z_o_dras3_rd__mixed_n3_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n4_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n5_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n6_t60_ex50000_z_o_dras3_rd__mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                'mixed_n3_t60_ex50000_m_z_o_dras3_rd__mixed_n3_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n4_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n5_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',
                'mixed_n6_t60_ex50000_m_z_o_dras3_rd__mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj',

                # tower unstable and normal with higher lr
                'tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda100_modelbffobj_lambda100_batch_norm',
                'tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda100_modelbffobj_lambda100_batch_norm',
                'tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda10_modelbffobj_lambda10_batch_norm',
                'tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda10_modelbffobj_lambda10_batch_norm',

                ]

# specify paths
out_root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/opmjlogs'
in_root = '/om/user/mbchang/physics/lua/logs'
copy_prefix = 'rsync -avz --exclude \'*.t7\' mbchang@openmind7.mit.edu:'
remote_prefix = '/om/user/mbchang/physics/lua/logs/'
js_root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/demo/js'


def copy(experiments):
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
                        print 'Made', batch_folder, '\n' + '-'*80
                        mkdir_p(batch_folder)

            # print 'plot hidden state'
            # plot_results.plot_hid_state(experiment_folder)  # TODO! check if filepath is correct
        except KeyboardInterrupt:
            sys.exit(0)

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

def create_gif_json(images_root, gifname):
    """
        writeGif(filename, images, duration=0.1, loops=0, dither=1)
            Write an animated gif from the specified images.
            images should be a list of numpy arrays of PIL images.
            Numpy images of type float should have pixels between 0 and 1.
            Numpy images of other types are expected to have values between 0 and 255.
    """
    # TODO! I'm assuming my img_id is fixed!
    file_names = sorted([fn for fn in os.listdir(images_root) if fn.endswith('.png')], key=lambda x: img_id_json(x))
    images = [Image.open(os.path.join(images_root,fn)) for fn in file_names]
    filename = os.path.join(images_root, gifname)
    writeGif(filename, images, duration=0.2)


def animate(experiments, remove_png):
    print('## ANIMATE ##')
    # wait until all of Demo_minimal has finished
    animated_experiments = []
    for experiment_folder in experiments:
        print '#'*80
        print 'Trying to animate', experiment_folder
        try:
            visual_folder = os.path.join(*[out_root, experiment_folder, 'visual'])
            if not os.listdir(visual_folder): 
                print 'Nothing in', visual_folder
            else:
                animated_experiments.append(experiment_folder)
                for batch_folder in os.listdir(visual_folder):
                    print '-'*80
                    gifname = experiment_folder + '_' + batch_folder + '.gif'
                    batch_folder = os.path.join(visual_folder, batch_folder)
                    if any(f.endswith('.png') for f in os.listdir(batch_folder)):
                        create_gif_json(batch_folder, gifname)

                        if remove_png:
                            print 'Removing images from', batch_folder
                            for imgfile in [x for x in os.listdir(batch_folder) if x.endswith('.png')]:
                                imgfile = os.path.join(batch_folder, imgfile)
                                command = 'rm ' + imgfile
                                os.system(command)
                    else:
                        print 'No .pngs found. Not creating gif for', batch_folder

        except KeyboardInterrupt:
            sys.exit(0)
    print 'Animated the following folders:'
    pprint.pprint(animated_experiments)

# copy(experiments)
# plot(experiments)
# visualize(experiments)
# animate(experiments, True)




