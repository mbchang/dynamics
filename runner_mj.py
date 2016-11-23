import copy
import os
import sys
import pprint

def create_jobs(dry_run, mode, ext):
    local = False
    detach = True

    if not os.path.exists("slurm_logs"):
        os.makedirs("slurm_logs")

    if not os.path.exists("slurm_scripts"):
        os.makedirs("slurm_scripts")


    jobs = [
            # {'dataset_folders':"{'balls_n3_t60_ex50000_rda','balls_n4_t60_ex50000_rda','balls_n5_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_rda','balls_n7_t60_ex50000_rda','balls_n8_t60_ex50000_rda'}"},  # blstm
            {'dataset_folders':"{'balls_n3_t60_ex50000_m_rda','balls_n4_t60_ex50000_m_rda','balls_n5_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n6_t60_ex50000_m_rda','balls_n7_t60_ex50000_m_rda','balls_n8_t60_ex50000_m_rda'}"},
            # {'dataset_folders':"{'balls_n4_t60_ex50000_rda'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000_rda'}"},
            # {'dataset_folders':"{'balls_n4_t60_ex50000_m_rda'}", 'test_dataset_folders': "{'balls_n4_t60_ex50000_m_rda'}"},  # blstm

            # {'dataset_folders':"{'walls_n2_t60_ex100_wU_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex100_wU_rda'}"},
            # {'dataset_folders':"{'walls_n2_t60_ex50000_wI_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wI_rda'}"},
            # {'dataset_folders':"{'walls_n2_t60_ex50000_wO_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wO_rda'}"},
            # {'dataset_folders':"{'walls_n2_t60_ex50000_wL_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wL_rda'}"},
            # {'dataset_folders':"{'walls_n2_t60_ex50000_wU_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wU_rda'}"},
            # {'dataset_folders':"{'walls_n2_t60_ex50000_wO_rda','walls_n2_t60_ex50000_wL_rda'}", 'test_dataset_folders': "{'walls_n2_t60_ex50000_wU_rda','walls_n2_t60_ex50000_wI_rda'}"},
            ]


    actual_jobs = []
    for job in jobs:
        job['name'] = job['dataset_folders'] + '__' + job['test_dataset_folders']
        job['name'] = job['name'].replace('{','').replace('}', '').replace("'","").replace('\\"','')
        for model in ['bffobj','np']:
            for nbrhd in [True]:  
                for nbhrdsize in [3.5]:  # [3, 3.5, 4, 4.5]
                    for layers in [5]:  # [2,3,4]
                        for lr in [3e-4]:  # [1e-4, 3e-4, 1e-2]
                            for cuda in [False]:
                                for im in [False]:
                                    # for veps in [1e-9]:
                                    #     for lda in [100]:
                                    #         for vlda in [100]:
                                    #             for bnorm in [False]:
                                    # for z in [True]:
                                                    for of in [False]:
                                                        for duo in [False]:
                                                            for f in [True]:
                                                                for rs in [True]:
                                                                    for seed in [0,1,2]:
                                                                        for nlan in [True]:
                                                                            for rnn_dim in [100]:
                                                                                job['model'] = model
                                                                                job['nbrhd'] = nbrhd
                                                                                job['layers'] = layers
                                                                                job['lr'] = lr
                                                                                # job['nbrhdsize'] = nbhrdsize
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
                                                                                # job['rnn_dim'] = rnn_dim
                                                                                job['of'] = of
                                                                                job['duo'] = duo
                                                                                # job['zero'] = z
                                                                                # job['num_past'] = 1
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

                    iseval = mode == 'sim' or mode == 'minf' or mode == 'sinf' or mode == 'oinf' or mode == 'tva' or mode == 'sa' or mode == 'oia'

                    # eval.lua does not have a 'dataset_folders' flag
                    if not(flag == 'dataset_folders' and iseval):
                        if flag == 'test_dataset_folders' and iseval and job['dataset_folders'] != job['test_dataset_folders']:
                            # generalization mode: join dataset_folders and test_dataset_folders
                            train_worlds = str(job['dataset_folders'])[1:]
                            test_worlds = str(job['test_dataset_folders'])[:-1]
                            flagstring = flagstring + " -" + flag + ' \"' + test_worlds + ',' + train_worlds + '\"'

                            # hacks
                            # flagstring = flagstring + " -" + flag + ' \"' + str(job['dataset_folders']) + '\"'
                            # if '_m_' in str(job['test_dataset_folders']):
                            #     flagstring = flagstring + " -" + flag + ' \"' + str("{'balls_n3_t60_ex50000_m_rda'}") + '\"'
                            # else:
                            #     flagstring = flagstring + " -" + flag + ' \"' + str("{'balls_n3_t60_ex50000_rda'}") + '\"'
                        else:
                            flagstring = flagstring + " -" + flag + ' \"' + str(job[flag]) + '\"'
                else:
                    if flag not in ['name']:
                        jobname = jobname + "_" + flag  + str(job[flag])
                        if (mode == 'sim' or mode == 'minf' or mode == 'sinf' or mode == 'oinf' or mode == 'tva' or mode == 'sa' or mode == 'oia') and flag not in ['test_dataset_folders', 'name']:
                            pass
                        else:
                            flagstring = flagstring + " -" + flag + " " + str(job[flag])

        flagstring = flagstring + " -name " + jobname.replace('_zero','') + " -mode " + mode 

        if mode == 'exp' or mode == 'expload' or mode == 'save':
            prefix = 'th main.lua'
        elif mode == 'sim' or mode == 'minf' or mode == 'sinf' or mode == 'oinf' or mode == 'tva' or mode == 'sa' or mode == 'oia':
            prefix = 'th eval.lua'
        else:
            assert False, 'Unknown mode'

        jobcommand = prefix + flagstring

        if local and not dry_run:
            if detach:
                os.system(jobcommand + ' 2> slurm_logs/' + jobname + '.err 1> slurm_logs/' + jobname + '.out &')
            else:
                os.system(jobcommand)

        else:
            blacklist = []

            if jobname not in blacklist:
                jobname = jobname.replace('_zero_','z') # ZERO CHANGED
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
# run_experiment(dry_run)
# run_experimentload(dry_run)
sim(dry_run)
# minf(dry_run)
# sinf(dry_run)
# oinf(dry_run)
# save(dry_run)
# tva(dry_run)
# sa(dry_run)
# oia(dry_run)


