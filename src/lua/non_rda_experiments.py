experiments_dict = {
    'Balls Prediction All': [
         # RD dataset with 2.5 buffer
        ('balls_n3_t60_ex50000_rd__balls_n3_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '3 balls'),
        ('balls_n5_t60_ex50000_rd__balls_n5_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '5 balls'),
        ('balls_n6_t60_ex50000_rd__balls_n6_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '6 balls'),
        ('balls_n7_t60_ex50000_rd__balls_n7_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '7 balls'),
        ('balls_n8_t60_ex50000_rd__balls_n8_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '8 balls'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '4 balls'),
    ],

    'Balls Prediction Mass All': [
        # RD dataset with 2.5 buffer
        ('balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj', '4 balls'),
        ('balls_n3_t60_ex50000_m_rd__balls_n3_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj', '3 balls'),
        ('balls_n5_t60_ex50000_m_rd__balls_n5_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj', '5 balls'),
        ('balls_n6_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj', '6 balls'),
        ('balls_n7_t60_ex50000_m_rd__balls_n7_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj', '7 balls'),
        ('balls_n8_t60_ex50000_m_rd__balls_n8_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_im_modelbffobj', '8 balls'),
    ],

    'Balls Generalization': [
        ('balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', 'No Mass'),
        ('balls_n3_t60_ex50000_m_rd,balls_n4_t60_ex50000_m_rd,balls_n5_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd,balls_n7_t60_ex50000_m_rd,balls_n8_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', 'Mass'),
    ],

    'Mixed Prediction Dras vs Dras3': [
        ('mixed_n3_t60_ex50000_z_o_dras_rd__mixed_n3_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '3 objects 2 sizes'),
        ('mixed_n4_t60_ex50000_z_o_dras_rd__mixed_n4_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '4 objects 2 sizes'),
        ('mixed_n5_t60_ex50000_z_o_dras_rd__mixed_n5_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '5 objects 2 sizes'),
        ('mixed_n6_t60_ex50000_z_o_dras_rd__mixed_n6_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '6 objects 2 sizes'),

        ('mixed_n3_t60_ex50000_z_o_dras3_rd__mixed_n3_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '3 objects 3 sizes'),
        ('mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n4_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '4 objects 3 sizes'),
        ('mixed_n5_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '5 objects 3 sizes'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rd__mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '6 objects 3 sizes'),
    ],

    'Mixed Prediction Mass Dras vs Dras3': [
        ('mixed_n3_t60_ex50000_m_z_o_dras_rd__mixed_n3_t60_ex50000_m_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '3 objects 2 sizes'),
        ('mixed_n4_t60_ex50000_m_z_o_dras_rd__mixed_n4_t60_ex50000_m_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '4 objects 2 sizes'),
        ('mixed_n5_t60_ex50000_m_z_o_dras_rd__mixed_n5_t60_ex50000_m_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '5 objects 2 sizes'),
        ('mixed_n6_t60_ex50000_m_z_o_dras_rd__mixed_n6_t60_ex50000_m_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '6 objects 2 sizes'),
        
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rd__mixed_n3_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '3 objects 3 sizes'),
        ('mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n4_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '4 objects 3 sizes'),
        ('mixed_n5_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '5 objects 3 sizes'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rd__mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', '6 objects 3 sizes'),
    ],

    'Mixed Generalization': [
        ('mixed_n3_t60_ex50000_m_z_o_dras_rd,mixed_n4_t60_ex50000_m_z_o_dras_rd__mixed_n5_t60_ex50000_m_z_o_dras_rd,mixed_n6_t60_ex50000_m_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', 'Mass 2 sizes'),
        ('mixed_n3_t60_ex50000_z_o_dras_rd,mixed_n4_t60_ex50000_z_o_dras_rd__mixed_n5_t60_ex50000_z_o_dras_rd,mixed_n6_t60_ex50000_z_o_dras_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', 'No Mass 2 sizes'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rd,mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd,mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', 'No Mass 3 sizes'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rd,mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd,mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelbffobj', 'Mass 3 sizes'),
    ],

    'Tower': [
        ('tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda100_modelbffobj_lambda100_batch_norm', 'Lambda 100'),
        ('tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda100_modelbffobj_lambda100_batch_norm', 'Unstable Lambda 100'),
        ('tower_n4_t120_ex25000_rd__tower_n4_t120_ex25000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda10_modelbffobj_lambda10_batch_norm', 'Lambda 10'),
        ('tower_n4_t120_ex25000_rd_unstable__tower_n4_t120_ex25000_rd_unstable_layers3_nbrhd_nbrhdsize3.5_lr0.0003_val_eps1e-09_vlambda10_modelbffobj_lambda10_batch_norm', 'Unstable Lambda 10'),
    ],

    'Balls Prediction Fast': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', 'NPE Neighborhood'),
    ],

    'Balls Prediction Mass Fast': [
        ('balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_fast_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_fast_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_fast_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', 'NPE'),
    ],

    'Balls Generalization Fast': [
        ('balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers3_fast_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers3_fast_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers3_fast_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', 'NPE'),
    ],

    'Balls Generalization Mass Fast': [
        ('balls_n3_t60_ex50000_m_rd,balls_n4_t60_ex50000_m_rd,balls_n5_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd,balls_n7_t60_ex50000_m_rd,balls_n8_t60_ex50000_m_rd_layers3_fast_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_m_rd,balls_n4_t60_ex50000_m_rd,balls_n5_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd,balls_n7_t60_ex50000_m_rd,balls_n8_t60_ex50000_m_rd_layers3_fast_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_m_rd,balls_n4_t60_ex50000_m_rd,balls_n5_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd,balls_n7_t60_ex50000_m_rd,balls_n8_t60_ex50000_m_rd_layers3_fast_lr0.0003_modelind', 'Independent'),
        ('balls_n3_t60_ex50000_m_rd,balls_n4_t60_ex50000_m_rd,balls_n5_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd,balls_n7_t60_ex50000_m_rd,balls_n8_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', 'NPE'),
    ],

    'Mixed Prediction Fast': [
        ('mixed_n6_t60_ex50000_z_o_dras3_rd__mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_fast_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rd__mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_fast_lr0.0003_modelind', 'Independent'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rd__mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_fast_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rd__mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', 'NPE'),
    ],

    'Mixed Prediction Mass Fast': [
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rd__mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_fast_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rd__mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_fast_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rd__mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_fast_lr0.0003_modelind', 'Independent'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rd__mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', 'NPE'),
    ],


    'Mixed Generalization Fast': [
        ('mixed_n3_t60_ex50000_z_o_dras3_rd,mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd,mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_fast_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rd,mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd,mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_fast_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rd,mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd,mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_fast_lr0.0003_modelind', 'Independent'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rd,mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd,mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', 'NPE'),
    ],

    'Mixed Generalization Mass Fast': [
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rd,mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd,mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_fast_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rd,mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd,mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_fast_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rd,mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd,mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_fast_lr0.0003_modelind', 'Independent'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rd,mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd,mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', 'NPE'),
    ],

    'Mixed Prediction Slow': [
        ('mixed_n6_t60_ex50000_z_o_dras3_rd__mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_lr0.0003_modelind', 'Independent'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rd__mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n6_t60_ex50000_z_o_dras3_rd__mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
    ],

    'Mixed Prediction Mass Slow': [
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rd__mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_lr0.0003_modelind', 'Independent'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rd__mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n6_t60_ex50000_m_z_o_dras3_rd__mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
    ],

    'Balls Prediction Slow': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
    ],
    
    'Balls Prediction Mass Slow': [
        ('balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_lr0.0003_modelind', 'Independent'),
        ('balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n4_t60_ex50000_m_rd__balls_n4_t60_ex50000_m_rd_layers3_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
    ],

    'Balls Generalization Slow': [
        ('balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers3_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers3_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_rd,balls_n4_t60_ex50000_rd,balls_n5_t60_ex50000_rd__balls_n6_t60_ex50000_rd,balls_n7_t60_ex50000_rd,balls_n8_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelind', 'Independent'),
    ],

    'Balls Generalization Mass Slow': [
        ('balls_n3_t60_ex50000_m_rd,balls_n4_t60_ex50000_m_rd,balls_n5_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd,balls_n7_t60_ex50000_m_rd,balls_n8_t60_ex50000_m_rd_layers3_lr0.0003_modellstmcat', 'LSTM'),
        ('balls_n3_t60_ex50000_m_rd,balls_n4_t60_ex50000_m_rd,balls_n5_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd,balls_n7_t60_ex50000_m_rd,balls_n8_t60_ex50000_m_rd_layers3_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('balls_n3_t60_ex50000_m_rd,balls_n4_t60_ex50000_m_rd,balls_n5_t60_ex50000_m_rd__balls_n6_t60_ex50000_m_rd,balls_n7_t60_ex50000_m_rd,balls_n8_t60_ex50000_m_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelind', 'Independent'),    
    ],

    'Mixed Generalization Slow': [
        ('mixed_n3_t60_ex50000_z_o_dras3_rd,mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd,mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rd,mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd,mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n3_t60_ex50000_z_o_dras3_rd,mixed_n4_t60_ex50000_z_o_dras3_rd__mixed_n5_t60_ex50000_z_o_dras3_rd,mixed_n6_t60_ex50000_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelind', 'Independent'),
    ],

    'Mixed Generalization Mass Slow': [
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rd,mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd,mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_lr0.0003_modelbffobj', 'NPE No Neighborhood'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rd,mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd,mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_lr0.0003_modellstmcat', 'LSTM'),
        ('mixed_n3_t60_ex50000_m_z_o_dras3_rd,mixed_n4_t60_ex50000_m_z_o_dras3_rd__mixed_n5_t60_ex50000_m_z_o_dras3_rd,mixed_n6_t60_ex50000_m_z_o_dras3_rd_layers3_nbrhd_nbrhdsize3.5_lr0.0003_modelind', 'Independent'),
    ],

    'LSTM Search': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0001_modellstmcat', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.001_modellstmcat', '4 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0001_modellstmcat', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.001_modellstmcat', '5 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0001_modellstmcat', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.001_modellstmcat', '6 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0001_modellstmcat', '7 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.001_modellstmcat', '7 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0003_modellstmcat', '4 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0003_modellstmcat', '5 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0003_modellstmcat', '6 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0003_modellstmcat', '7 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.001_modellstmcat', '3 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0001_modellstmcat', '3 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modellstmcat', '3 Layers, Learning Rate 0.0003'),

    ],

    'Independent Search': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0001_modelind', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.001_modelind', '4 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0001_modelind', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.001_modelind', '5 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0001_modelind', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.001_modelind', '6 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0001_modelind', '7 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.001_modelind', '7 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0003_modelind', '4 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0003_modelind', '7 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0003_modelind', '5 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0003_modelind', '6 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0001_modelind', '3 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.001_modelind', '3 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modelind', '3 Layers, learning Rate 0.0003'),
    ],

    'LSTM Search Learning Rate 0.0003': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0003_modellstmcat', '4 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0003_modellstmcat', '5 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0003_modellstmcat', '6 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0003_modellstmcat', '7 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modellstmcat', '3 Layers, Learning Rate 0.0003'),
    ],

    'LSTM Search Learning Rate 0.0001': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0001_modellstmcat', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0001_modellstmcat', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0001_modellstmcat', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0001_modellstmcat', '7 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0001_modellstmcat', '3 Layers, Learning Rate 0.0001'),
    ],

    'LSTM Search Learning Rate 0.001': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.001_modellstmcat', '4 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.001_modellstmcat', '5 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.001_modellstmcat', '6 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.001_modellstmcat', '7 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.001_modellstmcat', '3 Layers, Learning Rate 0.001'),
    ],

    'Independent Search Learning Rate 0.0003': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0003_modelind', '4 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0003_modelind', '7 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0003_modelind', '5 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0003_modelind', '6 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modelind', '3 Layers, learning Rate 0.0003'),
    ],

    'Independent Search Learnign Rate 0.0001': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0001_modelind', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0001_modelind', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0001_modelind', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0001_modelind', '7 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0001_modelind', '3 Layers, Learning Rate 0.0001'),
    ],

    'Independent Search 0.001': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.001_modelind', '4 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.001_modelind', '5 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.001_modelind', '6 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.001_modelind', '7 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.001_modelind', '3 Layers, Learning Rate 0.001'),
    ],

    'LSTM Search Layers 3': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.001_modellstmcat', '3 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0001_modellstmcat', '3 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modellstmcat', '3 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_rs_fast_lr3e-05_modellstmcat', '3 Layers, Learning Rate 0.00003'),
    ],

    'LSTM Search Layers 4': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0001_modellstmcat', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.001_modellstmcat', '4 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0003_modellstmcat', '4 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_rs_fast_lr3e-05_modellstmcat', '4 Layers, Learning Rate 0.00003'),
    ],

    'LSTM Search Layers 5': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0001_modellstmcat', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.001_modellstmcat', '5 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0003_modellstmcat', '5 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_rs_fast_lr3e-05_modellstmcat', '5 Layers, Learning Rate 0.00003'),
    ],

    'LSTM Search Layers 6': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0001_modellstmcat', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.001_modellstmcat', '6 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0003_modellstmcat', '6 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_rs_fast_lr3e-05_modellstmcat', '6 Layers, Learning Rate 0.00003'),
    ],

    'LSTM Search Layers 7': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0001_modellstmcat', '7 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.001_modellstmcat', '7 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0003_modellstmcat', '7 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_rs_fast_lr3e-05_modellstmcat', '7 Layers, Learning Rate 0.00003'),
    ],

    'Independent Search Layers 3': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0001_modelind', '3 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.001_modelind', '3 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modelind', '3 Layers, learning Rate 0.0003'),
    ],

    'Independent Search Layers 4': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0001_modelind', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.001_modelind', '4 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0003_modelind', '4 Layers, Learning Rate 0.0003'),
    ],

    'Independent Search Layers 5': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0001_modelind', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.001_modelind', '5 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0003_modelind', '5 Layers, Learning Rate 0.0003'),
    ],

    'Independent Search Layers 6': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0001_modelind', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.001_modelind', '6 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0003_modelind', '6 Layers, Learning Rate 0.0003'),
    ],

    'Independent Search Layers 7': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0001_modelind', '7 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.001_modelind', '7 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0003_modelind', '7 Layers, Learning Rate 0.0003'),
    ],

    'NPE No Neighborhood Layers 4': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0003_modelbffobj', '4 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0001_modelbffobj', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.001_modelbffobj', '4 Layers, Learning Rate 0.001'),
    ],

    'NPE No Neighborhood Layers 5': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0003_modelbffobj', '5 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0001_modelbffobj', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.001_modelbffobj', '5 Layers, Learning Rate 0.001'),
    ],

    'NPE No Neighborhood Layers 6': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0003_modelbffobj', '6 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0001_modelbffobj', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.001_modelbffobj', '6 Layers, Learning Rate 0.001'),
    ],

    'NPE No Neighborhood Layers 7': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0003_modelbffobj', '7 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0001_modelbffobj', '7 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.001_modelbffobj', '7 Layers, Learning Rate 0.001'),
    ],

    'NPE No Neighborhood Learning Rate 0.0003': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_fast_lr0.0003_modelbffobj', '3 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0003_modelbffobj', '4 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0003_modelbffobj', '5 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0003_modelbffobj', '6 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0003_modelbffobj', '7 Layers, Learning Rate 0.0003'),
    ],

    'NPE No Neighborhood Learning Rate 0.0001': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.0001_modelbffobj', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.0001_modelbffobj', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.0001_modelbffobj', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.0001_modelbffobj', '7 Layers, Learning Rate 0.0001'),
    ],

    'NPE No Neighborhood Learning Rate 0.001': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_fast_lr0.001_modelbffobj', '4 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_fast_lr0.001_modelbffobj', '5 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_fast_lr0.001_modelbffobj', '6 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_fast_lr0.001_modelbffobj', '7 Layers, Learning Rate 0.001'),
    ],

    'NPE Layers 4': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_nbrhd_nbrhdsize3.5_fast_lr0.0001_modelbffobj', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', '4 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_nbrhd_nbrhdsize3.5_fast_lr0.001_modelbffobj', '4 Layers, Learning Rate 0.001'),
    ],

    'NPE Layers 5': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_nbrhd_nbrhdsize3.5_fast_lr0.0001_modelbffobj', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', '5 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_nbrhd_nbrhdsize3.5_fast_lr0.001_modelbffobj', '5 Layers, Learning Rate 0.001'),
    ],

    'NPE Layers 6': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_nbrhd_nbrhdsize3.5_fast_lr0.0001_modelbffobj', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', '6 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_nbrhd_nbrhdsize3.5_fast_lr0.001_modelbffobj', '6 Layers, Learning Rate 0.001'),
    ],

    'NPE Layers 7': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_nbrhd_nbrhdsize3.5_fast_lr0.0001_modelbffobj', '7 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', '7 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_nbrhd_nbrhdsize3.5_fast_lr0.001_modelbffobj', '7 Layers, Learning Rate 0.001'),
    ],

    'NPE Learning Rate 0.0001': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_nbrhd_nbrhdsize3.5_fast_lr0.0001_modelbffobj', '4 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_nbrhd_nbrhdsize3.5_fast_lr0.0001_modelbffobj', '5 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_nbrhd_nbrhdsize3.5_fast_lr0.0001_modelbffobj', '6 Layers, Learning Rate 0.0001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_nbrhd_nbrhdsize3.5_fast_lr0.0001_modelbffobj', '7 Layers, Learning Rate 0.0001'),
    ],

    'NPE Learning Rate 0.001': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_nbrhd_nbrhdsize3.5_fast_lr0.001_modelbffobj', '4 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_nbrhd_nbrhdsize3.5_fast_lr0.001_modelbffobj', '5 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_nbrhd_nbrhdsize3.5_fast_lr0.001_modelbffobj', '6 Layers, Learning Rate 0.001'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_nbrhd_nbrhdsize3.5_fast_lr0.001_modelbffobj', '7 Layers, Learning Rate 0.001'),
    ],

    'NPE Learning Rate 0.0003': [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', '3 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers4_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', '4 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers5_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', '5 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers6_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', '6 Layers, Learning Rate 0.0003'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers7_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', '7 Layers, Learning Rate 0.0003'),

    ],

    'NPE Random vs Priority' : [
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_rs_fast_lr0.0003_modelbffobj', 'RS'),
        ('balls_n4_t60_ex50000_rd__balls_n4_t60_ex50000_rd_layers3_nbrhd_nbrhdsize3.5_fast_lr0.0003_modelbffobj', 'PS'),
    ]
}