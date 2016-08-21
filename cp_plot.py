import os
import sys
import plot_results

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



                ]

# specify paths
out_root = 'opmjlogs'
in_root = '/om/user/mbchang/physics/lua/logs'
copy_prefix = 'rsync -avz --exclude \'*.t7\' mbchang@openmind7.mit.edu:'
remote_prefix = '/om/user/mbchang/physics/lua/logs/'

# copy
if len(experiments) > 1:
    remote_paths = remote_prefix + '\{' + ','.join(['\\"' + e + '\\"' for e in experiments]) + '\} '
    command = copy_prefix + remote_paths + out_root
else:
    remote_paths = remote_prefix + experiments[0] + ' '
    command = copy_prefix + remote_paths + out_root + '/'

response = raw_input('Running command:\n\n' + command + '\n\nProceed?[y/n]')
if response == 'y':
    os.system(command)
elif response != 'n':
    response = raw_input('Running command:\n\n' + command + '\nProceed?[y/n]')
else:
    print 'Not running command.'
    sys.exit(0)

# plot
for experiment_folder in experiments:
    try:
        experiment_folder = os.path.join(out_root, experiment_folder)
        # command = 'th plot_results.lua -hid -infolder ' + experiment_folder
        command = 'th plot_results.lua -infolder ' + experiment_folder
        print command
        os.system(command)
        # print 'plot hidden state'
        # plot_results.plot_hid_state(experiment_folder)  # TODO! check if filepath is correct
    except KeyboardInterrupt:
        sys.exit(0)
