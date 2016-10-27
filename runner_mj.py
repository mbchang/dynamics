import copy
import os
import sys
import pprint

def create_jobs(dry_run, mode, ext):
    # dry_run = '--dry-run' in sys.argv
    # local   = '--local' in sys.argv
    # detach  = '--detach' in sys.argv

    # dry_run = False
    local = False
    detach = True

    if not os.path.exists("slurm_logs"):
        os.makedirs("slurm_logs")

    if not os.path.exists("slurm_scripts"):
        os.makedirs("slurm_scripts")


    jobs = [
            # {'dataset_folders':"{'balls_n3_t60_ex50000'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n10_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n5_t60_ex50000','balls_n7_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex50000'}"},
            # {'dataset_folders':"{'tower_n10_t60_ex50000'}", 'test_dataset_folders': "{'tower_n10_t60_ex50000'}"}
            # {'dataset_folders':"{'tower_n10_t120_ex50000'}", 'test_dataset_folders': "{'tower_n10_t120_ex50000'}"},

            # {'dataset_folders':"{'balls_n7_t60_ex50000'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex10000_gf'}", 'test_dataset_folders': "{'balls_n5_t60_ex10000_gf'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex10000_fr'}", 'test_dataset_folders': "{'balls_n5_t60_ex10000_fr'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex10000'}", 'test_dataset_folders': "{'balls_n5_t60_ex10000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000,balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000,balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000,balls_n5_t60_ex50000,balls_n7_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex5000'}"},

            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n4_t60_ex50000'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n4_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n3_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex50000','balls_n3_t60_ex50000'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000'}"},

            # generalization 2
            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n4_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n4_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n4_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n8_t60_ex50000'}"},

            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n4_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000','balls_n7_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n4_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000','balls_n8_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n4_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000','balls_n8_t60_ex50000'}"},

            # {'dataset_folders':"{'balls_n3_t60_ex50000','balls_n4_t60_ex50000','balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000','balls_n7_t60_ex50000','balls_n8_t60_ex50000'}"},

            # {'dataset_folders':"{'balls_n3_t60_ex50000'}", 'test_dataset_folders': "{'balls_n3_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n4_t60_ex50000'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n6_t60_ex50000'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n7_t60_ex50000'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n8_t60_ex50000'}", 'test_dataset_folders': "{'balls_n8_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n9_t60_ex50000'}", 'test_dataset_folders': "{'balls_n9_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n10_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex50000'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000_m'}", 'test_dataset_folders': "{'balls_n3_t60_ex50000_m'}"},

            # 1,15,30 mass
            # {'dataset_folders':"{'balls_n4_t120_ex25000_m','balls_n5_t120_ex25000_m'}", 'test_dataset_folders': "{'balls_n4_t120_ex25000_m','balls_n5_t120_ex25000_m'}"},
            
            # test prediction and mass
            # these are for the rd without 2.5 buffer
            # {'dataset_folders':"{'balls_n4_t60_ex50000_rd'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000_rd'}"},
            # {'dataset_folders':"{'balls_n4_t60_ex50000_m_rd'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000_m_rd'}"},

            # actual experiments
            # {'dataset_folders':"{'balls_n3_t60_ex50000_rd'}", 'test_dataset_folders': "{'balls_n3_t60_ex50000_rd'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000_m_rd'}", 'test_dataset_folders': "{'balls_n3_t60_ex50000_m_rd'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex50000_rd'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000_rd'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex50000_m_rd'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000_m_rd'}"},
            # {'dataset_folders':"{'balls_n6_t60_ex50000_rd'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_rd'}"},
            # {'dataset_folders':"{'balls_n6_t60_ex50000_m_rd'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_m_rd'}"},
            # {'dataset_folders':"{'balls_n7_t60_ex50000_rd'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000_rd'}"},
            # {'dataset_folders':"{'balls_n7_t60_ex50000_m_rd'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000_m_rd'}"},
            # {'dataset_folders':"{'balls_n8_t60_ex50000_rd'}", 'test_dataset_folders': "{'balls_n8_t60_ex50000_rd'}"},
            # {'dataset_folders':"{'balls_n8_t60_ex50000_m_rd'}", 'test_dataset_folders': "{'balls_n8_t60_ex50000_m_rd'}"},

            # {'dataset_folders':"{'tower_n4_t120_ex25000_rd'}", 'test_dataset_folders': "{'tower_n4_t120_ex25000_rd'}"},
            # {'dataset_folders':"{'tower_n4_t120_ex25000_rd_stable'}", 'test_dataset_folders': "{'tower_n4_t120_ex25000_rd_stable'}"},
            # {'dataset_folders':"{'tower_n4_t120_ex25000_rd_unstable'}", 'test_dataset_folders': "{'tower_n4_t120_ex25000_rd_unstable'}"},
            # {'dataset_folders':"{'tower_n2_t120_ex25000_rd_stable'}", 'test_dataset_folders': "{'tower_n2_t120_ex25000_rd_stable'}"},


            # {'dataset_folders':"{'tower_n6_t120_ex25000_rd'}", 'test_dataset_folders': "{'tower_n6_t120_ex25000_rd'}"},
            # {'dataset_folders':"{'tower_n8_t120_ex25000_rd'}", 'test_dataset_folders': "{'tower_n8_t120_ex25000_rd'}"},
            # {'dataset_folders':"{'tower_n10_t120_ex25000_rd'}", 'test_dataset_folders': "{'tower_n10_t120_ex25000_rd'}"},

            # {'dataset_folders':"{'balls_n3_t60_ex50000_rd','balls_n4_t60_ex50000_rd','balls_n5_t60_ex50000_rd'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_rd','balls_n7_t60_ex50000_rd','balls_n8_t60_ex50000_rd'}"},
            # {'dataset_folders':"{'balls_n3_t60_ex50000_m_rd','balls_n4_t60_ex50000_m_rd','balls_n5_t60_ex50000_m_rd'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_m_rd','balls_n7_t60_ex50000_m_rd','balls_n8_t60_ex50000_m_rd'}"},

            # mixed
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_rd'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_rd'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_z_rd'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_z_rd'}"},  # this does not use dras sizing
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_o_rd'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_o_rd'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_z_o_rd'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_z_o_rd'}"},  # this does not use dras sizing
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_z_dras_rd'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_z_dras_rd'}"},

            # # invisible
            # {'dataset_folders':"{'invisible_n6_t60_ex50000_z_o_dras_rd'}", 'test_dataset_folders': "{'invisible_n6_t60_ex50000_z_o_dras_rd'}"},
            # {'dataset_folders':"{'invisible_n5_t60_ex50000_z_o_dras_rd'}", 'test_dataset_folders': "{'invisible_n5_t60_ex50000_z_o_dras_rd'}"},

            # mixed dras
            # {'dataset_folders':"{'mixed_n3_t60_ex50000_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n3_t60_ex50000_z_o_dras_rd'}"},
            # {'dataset_folders':"{'mixed_n4_t60_ex50000_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n4_t60_ex50000_z_o_dras_rd'}"},
            # {'dataset_folders':"{'mixed_n5_t60_ex50000_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_z_o_dras_rd'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_z_o_dras_rd'}"},
            # {'dataset_folders':"{'mixed_n3_t60_ex50000_z_o_dras_rd','mixed_n4_t60_ex50000_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_z_o_dras_rd','mixed_n6_t60_ex50000_z_o_dras_rd'}"},


            # # mixed dras3
            # {'dataset_folders':"{'mixed_n3_t60_ex50000_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n3_t60_ex50000_z_o_dras3_rd'}"},
            # {'dataset_folders':"{'mixed_n4_t60_ex50000_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n4_t60_ex50000_z_o_dras3_rd'}"},
            # {'dataset_folders':"{'mixed_n5_t60_ex50000_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_z_o_dras3_rd'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_z_o_dras3_rd'}"},
            # {'dataset_folders':"{'mixed_n3_t60_ex50000_z_o_dras3_rd','mixed_n4_t60_ex50000_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_z_o_dras3_rd','mixed_n6_t60_ex50000_z_o_dras3_rd'}"},

            # {'dataset_folders':"{'mixed_n3_t60_ex50000_m_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n3_t60_ex50000_m_z_o_dras3_rd'}"},
            # {'dataset_folders':"{'mixed_n4_t60_ex50000_m_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n4_t60_ex50000_m_z_o_dras3_rd'}"},
            # {'dataset_folders':"{'mixed_n5_t60_ex50000_m_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_m_z_o_dras3_rd'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_m_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_m_z_o_dras3_rd'}"},
            # {'dataset_folders':"{'mixed_n3_t60_ex50000_m_z_o_dras3_rd','mixed_n4_t60_ex50000_m_z_o_dras3_rd'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_m_z_o_dras3_rd','mixed_n6_t60_ex50000_m_z_o_dras3_rd'}"},

            # {'dataset_folders':"{'mixed_n3_t60_ex50000_m_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n3_t60_ex50000_m_z_o_dras_rd'}"},
            # {'dataset_folders':"{'mixed_n4_t60_ex50000_m_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n4_t60_ex50000_m_z_o_dras_rd'}"},
            # {'dataset_folders':"{'mixed_n5_t60_ex50000_m_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_m_z_o_dras_rd'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_m_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_m_z_o_dras_rd'}"},  
            # {'dataset_folders':"{'mixed_n3_t60_ex50000_m_z_o_dras_rd','mixed_n4_t60_ex50000_m_z_o_dras_rd'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_m_z_o_dras_rd','mixed_n6_t60_ex50000_m_z_o_dras_rd'}"},


            # rda experiments
            # {'dataset_folders':"{'mixed_n3_t60_ex50000_z_o_dras3_rda','mixed_n4_t60_ex50000_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_z_o_dras3_rda','mixed_n6_t60_ex50000_z_o_dras3_rda'}"},
            # {'dataset_folders':"{'mixed_n3_t60_ex50000_m_z_o_dras3_rda','mixed_n4_t60_ex50000_m_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_m_z_o_dras3_rda','mixed_n6_t60_ex50000_m_z_o_dras3_rda'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_z_o_dras3_rda'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_m_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_m_z_o_dras3_rda'}"},  # blstm


            # {'dataset_folders':"{'balls_n3_t60_ex50000_rda','balls_n4_t60_ex50000_rda','balls_n5_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_rda','balls_n7_t60_ex50000_rda','balls_n8_t60_ex50000_rda'}"},  # blstm
            # {'dataset_folders':"{'balls_n3_t60_ex50000_m_rda','balls_n4_t60_ex50000_m_rda','balls_n5_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_m_rda','balls_n7_t60_ex50000_m_rda','balls_n8_t60_ex50000_m_rda'}"},

            # {'dataset_folders':"{'balls_n4_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000_rda'}"},
            # {'dataset_folders':"{'balls_n4_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000_m_rda'}"},  # blstm


            # {'dataset_folders':"{'balls_n3_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n3_t60_ex50000_rda'}"},
            # {'dataset_folders':"{'balls_n4_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000_rda'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000_rda'}"},
            # {'dataset_folders':"{'balls_n6_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_rda'}"},
            # {'dataset_folders':"{'balls_n7_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000_rda'}"},
            # {'dataset_folders':"{'balls_n8_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n8_t60_ex50000_rda'}"},


            # {'dataset_folders':"{'balls_n3_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n3_t60_ex50000_m_rda'}"},
            # {'dataset_folders':"{'balls_n4_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000_m_rda'}"},
            # {'dataset_folders':"{'balls_n5_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000_m_rda'}"},
            # {'dataset_folders':"{'balls_n6_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_m_rda'}"},
            # {'dataset_folders':"{'balls_n7_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000_m_rda'}"},
            # {'dataset_folders':"{'balls_n8_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n8_t60_ex50000_m_rda'}"},

            # {'dataset_folders':"{'mixed_n4_t60_ex50000_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n4_t60_ex50000_z_o_dras3_rda'}"},
            # {'dataset_folders':"{'mixed_n5_t60_ex50000_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_z_o_dras3_rda'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_z_o_dras3_rda'}"},


            # {'dataset_folders':"{'mixed_n3_t60_ex50000_m_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n3_t60_ex50000_m_z_o_dras3_rda'}"},
            # {'dataset_folders':"{'mixed_n4_t60_ex50000_m_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n4_t60_ex50000_m_z_o_dras3_rda'}"},
            # {'dataset_folders':"{'mixed_n5_t60_ex50000_m_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n5_t60_ex50000_m_z_o_dras3_rda'}"},
            # {'dataset_folders':"{'mixed_n6_t60_ex50000_m_z_o_dras3_rda'}", 'test_dataset_folders': "{'mixed_n6_t60_ex50000_m_z_o_dras3_rda'}"},

            # {'dataset_folders':"{'tower_n5_t120_ex25000_rda'}", 'test_dataset_folders': "{'tower_n5_t120_ex25000_rda'}"},
            # {'dataset_folders':"{'tower_n6_t120_ex25000_rda'}", 'test_dataset_folders': "{'tower_n6_t120_ex25000_rda'}"},
            # {'dataset_folders':"{'tower_n7_t120_ex25000_rda'}", 'test_dataset_folders': "{'tower_n7_t120_ex25000_rda'}"},
            # {'dataset_folders':"{'tower_n8_t120_ex25000_rda'}", 'test_dataset_folders': "{'tower_n8_t120_ex25000_rda'}"},
            # {'dataset_folders':"{'tower_n5_t120_ex25000_rda','tower_n6_t120_ex25000_rda'}", 'test_dataset_folders': "{'tower_n7_t120_ex25000_rda','tower_n8_t120_ex25000_rda'}"},

            # {'dataset_folders':"{'walls_n2_t60_ex100_wU_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex100_wU_rda'}"},
            # {'dataset_folders':"{'walls_n2_t60_ex50000_wI_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wI_rda'}"},
            # {'dataset_folders':"{'walls_n2_t60_ex50000_wO_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wO_rda'}"},
            # {'dataset_folders':"{'walls_n2_t60_ex50000_wL_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wL_rda'}"},
            # {'dataset_folders':"{'walls_n2_t60_ex50000_wU_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wU_rda'}"},
            {'dataset_folders':"{'walls_n2_t60_ex50000_wO_rda','walls_n2_t60_ex50000_wL_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wU_rda','walls_n2_t60_ex50000_wI_rda'}"},
            ]


    actual_jobs = []
    for job in jobs:
        job['name'] = job['dataset_folders'] + '__' + job['test_dataset_folders']
        job['name'] = job['name'].replace('{','').replace('}', '').replace("'","").replace('\\"','')
        for model in ['lstm']:
            for nbrhd in [True]:  
                for nbhrdsize in [3.5]:  # [3, 3.5, 4, 4.5]
                    for layers in [3]:  # [2,3,4]
                        for lr in [3e-4,1e-3]:  # [1e-4, 3e-4, 1e-2]
                            for cuda in [False]:
                                for im in [False]:
                                    # for veps in [1e-9]:
                                    #     for lda in [100]:
                                    #         for vlda in [100]:
                                    #             for bnorm in [False]:
                                                    for of in [True]:
                                                        for duo in [True]:
                                                            for f in [True]:
                                                                for rs in [True]:
                                                                    for seed in [0,1,2]:
                                                                        for nlan in [True]:
                                                                            for rnn_dim in [50,100]:
                                                                                job['model'] = model
                                                                                job['nbrhd'] = nbrhd
                                                                                job['layers'] = layers
                                                                                job['lr'] = lr
                                                                                job['nbrhdsize'] = nbhrdsize
                                                                                job['im'] = im
                                                                                job['fast'] = f
                                                                                job['rs'] = rs
                                                                                job['seed'] = seed
                                                                                job['nlan'] = nlan
                                                                                job['cuda'] = cuda
                                                                                # job['dropout'] = dropout
                                                                                # job['val_eps'] = veps
                                                                                # job['lambda'] = lda
                                                                                # job['vlambda'] = vlda
                                                                                # job['batch_norm'] = bnorm
                                                                                job['rnn_dim'] = rnn_dim
                                                                                job['of'] = of
                                                                                job['duo'] = duo
                                                                                actual_jobs.append(copy.deepcopy(job))
    jobs = actual_jobs


    if dry_run:
        print "NOT starting jobs:"
    else:
        print "Starting jobs:"

    for job in jobs:
        jobname = job['name']
        flagstring = ""
        for flag in job:
            if isinstance(job[flag], bool):
                if job[flag]:
                    jobname = jobname + "_" + flag
                    if not (mode == 'sim' or mode == 'minf' or mode == 'sinf' or mode == 'oinf' or mode == 'tva' or mode == 'sa' or mode == 'oia'):
                        flagstring = flagstring + " -" + flag
                else:
                    print "WARNING: Excluding 'False' flag " + flag
            else:
                if flag in ['dataset_folders', 'test_dataset_folders']:
                    # eval.lua does not have a 'dataset_folders' flag
                    if not(mode == 'sim' and flag == 'dataset_folders') and not(mode == 'minf' and flag == 'dataset_folders') and not(mode == 'sinf' and flag == 'dataset_folders') and not(mode == 'oinf' and flag == 'dataset_folders') and not(mode == 'tva' and flag == 'dataset_folders') and not(mode == 'sa' and flag == 'dataset_folders') and not(mode == 'oia' and flag == 'dataset_folders'):
                        flagstring = flagstring + " -" + flag + ' \"' + str(job[flag] + '\"')                        
                else:
                    if flag not in ['name']:
                        jobname = jobname + "_" + flag  + str(job[flag])
                        if (mode == 'sim' or mode == 'minf' or mode == 'sinf' or mode == 'oinf' or mode == 'tva' or mode == 'sa' or mode == 'oia') and flag not in ['test_dataset_folders', 'name']:
                            pass
                        else:
                            flagstring = flagstring + " -" + flag + " " + str(job[flag])

        # print flagstring
        # jobname = jobname.replace('lr0.01_modelbl_seed0', 'seed0_lr0.01_modelbl')
        # jobname = jobname.replace('lr0.01_modelbl_seed1', 'seed1_lr0.01_modelbl')
        # jobname = jobname.replace('lr0.01_modelbl_seed2', 'seed2_lr0.01_modelbl')

        # jobname = jobname.replace('lr0.0003_modelbl_seed0', 'seed0_lr0.0003_modelbl')
        # jobname = jobname.replace('lr0.0003_modelbl_seed1', 'seed1_lr0.0003_modelbl')
        # jobname = jobname.replace('lr0.0003_modelbl_seed2', 'seed2_lr0.0003_modelbl')

        # jobname = jobname.replace('lr1e-05_modelbl_seed0', 'seed0_lr1e-05_modelbl')
        # jobname = jobname.replace('lr1e-05_modelbl_seed1', 'seed1_lr1e-05_modelbl')
        # jobname = jobname.replace('lr1e-05_modelbl_seed2', 'seed2_lr1e-05_modelbl')

        # print flagstring

        flagstring = flagstring + " -name " + jobname + " -mode " + mode 

        if mode == 'exp' or mode == 'expload' or mode == 'save':
            prefix = 'th main.lua'
        elif mode == 'sim' or mode == 'minf' or mode == 'sinf' or mode == 'oinf' or mode == 'tva' or mode == 'sa' or mode == 'oia':
            prefix = 'th eval.lua'
        else:
            assert False, 'Unknown mode'

        jobcommand = prefix + flagstring

        # print(jobcommand + '\n')
        if local and not dry_run:
            if detach:
                os.system(jobcommand + ' 2> slurm_logs/' + jobname + '.err 1> slurm_logs/' + jobname + '.out &')
            else:
                os.system(jobcommand)

        else:
            blacklist = [
                'balls_n4_t60_ex50000_m_rda__balls_n4_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_of_rnn_dim200_fast_nlan_lr0.001_modellstm_seed0',
                'balls_n3_t60_ex50000_m_rda,balls_n4_t60_ex50000_m_rda,balls_n5_t60_ex50000_m_rda__balls_n6_t60_ex50000_m_rda,balls_n7_t60_ex50000_m_rda,balls_n8_t60_ex50000_m_rda_layers5_nbrhd_nbrhdsize3.5_rs_of_rnn_dim100_fast_nlan_lr0.001_modellstm_seed2',
                'balls_n4_t60_ex50000_rda__balls_n4_t60_ex50000_rda_layers5_rs_rnn_dim200_fast_seed0_lr0.0003_modelbl',
                'walls_n2_t60_ex50000_wO_rda,walls_n2_t60_ex50000_wL_rda__walls_n2_t60_ex50000_wU_rda,walls_n2_t60_ex50000_wI_rda_layers5_nbrhd_nbrhdsize3.5_rs_of_rnn_dim100_fast_nlan_lr0.0003_modellstm_seed0',  
            ]

            if jobname not in blacklist:
                to_slurm(jobname + ext, jobcommand, dry_run)
            else:
                print '*'*80
                print 'THIS IS IN THE BLACKLIST:', jobname
                print '*'*80

def run_experiment(dry_run):
    create_jobs(dry_run=dry_run, mode='exp', ext='')

def save(dry_run):
    create_jobs(dry_run=dry_run, mode='save', ext='_save')

def run_experimentload(dry_run):
    create_jobs(dry_run=dry_run, mode='expload', ext='_expload')

def predict(dry_run):
    create_jobs(dry_run=dry_run, mode='pred', ext='_predict')

def sim(dry_run):
    create_jobs(dry_run=dry_run, mode='sim', ext='_sim')

def minf(dry_run):
    create_jobs(dry_run=dry_run, mode='minf', ext='_minf')

def sinf(dry_run):
    create_jobs(dry_run=dry_run, mode='sinf', ext='_sinf')

def oinf(dry_run):
    create_jobs(dry_run=dry_run, mode='oinf', ext='_oinf')

def sa(dry_run):
    create_jobs(dry_run=dry_run, mode='sa', ext='_sa')

def oia(dry_run):
    create_jobs(dry_run=dry_run, mode='oia', ext='_oia')

def tva(dry_run):
    create_jobs(dry_run=dry_run, mode='tva', ext='_tva')

def to_slurm(jobname, jobcommand, dry_run):
    # jobname_formatted = jobname.replace('{','\{').replace('}','\}').replace("'","\\'")
    # jobname_formatted2 = jobname_formatted.replace('\\"','')
    print '################'
    print 'name:', jobname
    print 'command:', jobcommand
    print '\n'

    with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH -c 1\n")
        # slurmfile.write("#SBATCH --gres=gpu:tesla-k20:1\n")
        slurmfile.write("#SBATCH --mem=30000\n")
        slurmfile.write("#SBATCH --time=6-23:00:00\n")
        slurmfile.write(jobcommand)

    if not dry_run:
        print "sbatch slurm_scripts/" + jobname + ".slurm &"
        os.system("sbatch slurm_scripts/" + jobname + ".slurm &")

dry_run = '--rd' not in sys.argv # real deal
run_experiment(dry_run)
# run_experimentload(dry_run)
# sim(dry_run)
# minf(dry_run)
# sinf(dry_run)
# oinf(dry_run)
# save(dry_run)
# tva(dry_run)
# sa(dry_run)
# oia(dry_run)


