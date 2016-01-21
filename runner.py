# From Will Whitney

import os
import sys
import pprint


def run_experiment(dry_run):
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


    base_networks = {
        }


    # Don't give it a save name - that gets generated for you
    # jobs = [
    #         {
    #             'lr': 5e-4,
    #             'opt': 'rmsprop'
    #         },
    #     ]

    jobs = [{'lr': r, 'opt': o}
                for r in [5e-4, 5e-5, 1e-3]
                    for o in ['rmsprop', 'optimrmsprop', 'adam']]


    if dry_run:
        print "NOT starting jobs:"
    else:
        print "Starting jobs:"

    jobs = [jobs[6], jobs[7], jobs[8]]

    for job in jobs:
        jobname = "baselinesubsampled"
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
                jobname = jobname + "_" + flag + "_" + str(job[flag])
                flagstring = flagstring + " -" + flag + " " + str(job[flag])
        flagstring = flagstring + " -name " + jobname + " -mode exp"

        jobcommand = "th main.lua" + flagstring

        print(jobcommand)
        if local and not dry_run:
            if detach:
                os.system(jobcommand + ' 2> slurm_logs/' + jobname + '.err 1> slurm_logs/' + jobname + '.out &')
            else:
                os.system(jobcommand)

        else:
            to_slurm(jobname, jobcommand, dry_run)
            # with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
            #     slurmfile.write("#!/bin/bash\n")
            #     slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
            #     slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
            #     slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
            #     slurmfile.write("#SBATCH -N 1\n")
            #     slurmfile.write("#SBATCH -c 1\n")
            #     slurmfile.write("#SBATCH -p gpu\n")
            #     slurmfile.write("#SBATCH --gres=gpu:1\n")
            #     slurmfile.write("#SBATCH --mem=30000\n")
            #     slurmfile.write(jobcommand)
            #
            # if not dry_run:
            #     # if 'gpuid' in job and job['gpuid'] >= 0:
            #     #     os.system("sbatch -N 1 -c 1 --gres=gpu:1 -p gpu --mem=10000 slurm_scripts/" + jobname + ".slurm &")
            #     # else:
            #     # print('RUNNING')
            #     # os.system("sbatch -N 1 -c 1 --gres=gpu:1 -p gpu --mem=10000 slurm_scripts/" + jobname + ".slurm &")
            #     os.system("sbatch slurm_scripts/" + jobname + ".slurm &")

def predict(dry_run):
    folder = 'logs2'
    for exp in os.listdir(folder):
        command = 'th main.lua -name ' + exp +  " -mode exp"
        to_slurm(exp + '_predict', command, dryrun)


def to_slurm(jobname, jobcommand, dry_run):
    with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
        slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH -c 1\n")
        slurmfile.write("#SBATCH -p gpu\n")
        slurmfile.write("#SBATCH --gres=gpu:1\n")
        slurmfile.write("#SBATCH --mem=30000\n")
        slurmfile.write(jobcommand)

    if not dry_run:
        # if 'gpuid' in job and job['gpuid'] >= 0:
        #     os.system("sbatch -N 1 -c 1 --gres=gpu:1 -p gpu --mem=10000 slurm_scripts/" + jobname + ".slurm &")
        # else:
        # print('RUNNING')
        # os.system("sbatch -N 1 -c 1 --gres=gpu:1 -p gpu --mem=10000 slurm_scripts/" + jobname + ".slurm &")
        os.system("sbatch slurm_scripts/" + jobname + ".slurm &")

dryrun = True
# run_experiment(dryrun)
# predict(dryrun)
