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

            {'dataset_folders':"{'tower_n10_t120_ex50000'}", 'test_dataset_folders': "{'tower_n10_t120_ex50000'}"},
            {'dataset_folders':"{'balls_n7_t60_ex50000'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000'}"},
            {'dataset_folders':"{'balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000'}"},
            {'dataset_folders':"{'balls_n5_t60_ex10000_gf'}", 'test_dataset_folders': "{'balls_n5_t60_ex10000_gf'}"},
            {'dataset_folders':"{'balls_n5_t60_ex10000_fr'}", 'test_dataset_folders': "{'balls_n5_t60_ex10000_fr'}"},
            {'dataset_folders':"{'balls_n5_t60_ex10000'}", 'test_dataset_folders': "{'balls_n5_t60_ex10000'}"},
            {'dataset_folders':"{'balls_n3_t60_ex50000'}", 'test_dataset_folders': "{'balls_n5_t60_ex50000'}"},
            {'dataset_folders':"{'balls_n3_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex50000'}"},
            {'dataset_folders':"{'balls_n3_t60_ex50000,balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n7_t60_ex50000'}"},
            {'dataset_folders':"{'balls_n3_t60_ex50000,balls_n5_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex50000'}"},
            {'dataset_folders':"{'balls_n3_t60_ex50000,balls_n5_t60_ex50000,balls_n7_t60_ex50000'}", 'test_dataset_folders': "{'balls_n10_t60_ex5000'}"},
            {'dataset_folders':"{'balls_n3_t60_ex50000'}", 'test_dataset_folders': "{'balls_n3_t60_ex50000'}"},

            ]

    for job in jobs:
        job['name'] = job['dataset_folders'] + '__' + job['test_dataset_folders']


    if dry_run:
        print "NOT starting jobs:"
    else:
        print "Starting jobs:"

    for job in jobs:
        jobname = ""
        flagstring = ""
        for flag in job:
            if isinstance(job[flag], bool):
                if job[flag]:
                    jobname = jobname + "_" + flag
                    flagstring = flagstring + " -" + flag
                else:
                    print "WARNING: Excluding 'False' flag " + flag
            elif flag == 'import':
                imported_network_name = job[flag]
                if imported_network_name in base_networks.keys():
                    network_location = base_networks[imported_network_name]
                    jobname = jobname + "_" + flag + "_" + str(imported_network_name)
                    flagstring = flagstring + " -" + flag + " " + str(network_location)
                else:
                    jobname = jobname + "_" + flag + "_" + str(job[flag])
                    flagstring = flagstring + " -" + flag + " " + str(job[flag])
            else:
                if flag in ['dataset_folders', 'test_dataset_folders']:
                    # eval.lua does not have a 'dataset_folders' flag
                    if not(mode == 'sim' and flag == 'dataset_folders'):
                        jobname = jobname + "_\"" + flag + "\"_" + str(job[flag])
                        flagstring = flagstring + " -" + flag + ' \"' + str(job[flag] + '\"')
                else:
                    if flag in ['name']:
                        job[flag] = job[flag].replace('{','').replace('}', '').replace("'","").replace('\\"','')
                    jobname = jobname + "_" + flag + "_" + str(job[flag])
                    flagstring = flagstring + " -" + flag + " " + str(job[flag])
        flagstring = flagstring + " -mode " + mode

        if mode == 'exp':
            prefix = 'th main.lua'
        elif mode == 'sim':
            prefix = 'th eval.lua'
        else:
            assert False, 'Unknown mode'

        jobcommand = prefix + flagstring

        print(jobcommand + '\n')
        if local and not dry_run:
            if detach:
                os.system(jobcommand + ' 2> slurm_logs/' + jobname + '.err 1> slurm_logs/' + jobname + '.out &')
            else:
                os.system(jobcommand)

        else:
            to_slurm(job['name'] + ext, jobcommand, dry_run)

def run_experiment(dry_run):
    create_jobs(dry_run=dry_run, mode='exp', ext='')

def predict(dry_run):
    create_jobs(dry_run=dry_run, mode='pred', ext='_predict')

def sim(dry_run):
    create_jobs(dry_run=dry_run, mode='sim', ext='_sim')

def to_slurm(jobname, jobcommand, dry_run):
    # jobname_formatted = jobname.replace('{','\{').replace('}','\}').replace("'","\\'")
    # jobname_formatted2 = jobname_formatted.replace('\\"','')

    print jobname
    print jobcommand
    # assert False

    # jobname_formatted = jobname.replace('{','').replace('}','').replace("'","")
    # jobname_formatted2 = jobname_formatted.replace('\\"','')
    #
    # with open('slurm_scripts/' + jobname.replace('\\"','') + '.slurm', 'w') as slurmfile:
    #     slurmfile.write("#!/bin/bash\n")
    #     slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
    #     slurmfile.write("#SBATCH --output=slurm_logs/" + jobname_formatted + ".out\n")
    #     slurmfile.write("#SBATCH -N 1\n")
    #     slurmfile.write("#SBATCH -c 1\n")
    #     slurmfile.write("#SBATCH --gres=gpu:tesla-k20:1\n")
    #     slurmfile.write("#SBATCH --mem=3000\n")
    #     slurmfile.write("#SBATCH --time=6-23:00:00\n")
    #     slurmfile.write(jobcommand)
    #
    # if not dry_run:
    #     print "sbatch slurm_scripts/" + jobname_formatted2 + ".slurm &"
    #     os.system("sbatch slurm_scripts/" + jobname_formatted2 + ".slurm &")

    # jobname_formatted = jobname.replace('{','').replace('}','').replace("'","")
    # jobname_formatted2 = jobname_formatted.replace('\\"','')

    with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH -c 1\n")
        slurmfile.write("#SBATCH --gres=gpu:tesla-k20:1\n")
        slurmfile.write("#SBATCH --mem=3000\n")
        slurmfile.write("#SBATCH --time=6-23:00:00\n")
        slurmfile.write(jobcommand)

    if not dry_run:
        print "sbatch slurm_scripts/" + jobname + ".slurm &"
        os.system("sbatch slurm_scripts/" + jobname + ".slurm &")

dry_run = '--rd' not in sys.argv # real deal
# run_experiment(dry_run)
predict(dry_run)
# sim(dry_run)
