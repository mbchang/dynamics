# From Will Whitney

import os
import sys
import pprint

def run_experiment(dry_run):
    create_jobs(dry_run=dry_run, mode='exp', ext='')


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


    base_networks = {
        }

    jobs = [{'lr': r, 'opt': o, 'batch_size': b, 'shuffle': s, 'lrdecay': d}
                for r in [5e-3, 1e-3]
                    for o in ['optimrmsprop', 'adam']
                        for b in [65, 260]
                            for s in ['true', 'false']
                                for d in [0.95, 0.99, 1]]
    for job in jobs:
        job['traincfgs'] = '[:-2:2-:]'
        job['testcfgs'] = '[:-2:2-:]'
        # job['lrdecay'] = 1

    if dry_run:
        print "NOT starting jobs:"
    else:
        print "Starting jobs:"

    # jobs = [jobs[6], jobs[7], jobs[8]]

    for job in jobs:
        jobname = "baselinesubsampledcontigdense3"
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
        flagstring = flagstring + " -name " + jobname + " -mode " + mode

        jobcommand = "th main.lua" + flagstring #+ '-traincfgs [:-2:2-:] -testcfgs [:-2:2-:]'  # TODO put it in slurm script?

        print(jobcommand)
        if local and not dry_run:
            if detach:
                os.system(jobcommand + ' 2> slurm_logs/' + jobname + '.err 1> slurm_logs/' + jobname + '.out &')
            else:
                os.system(jobcommand)

        else:
            to_slurm(jobname + ext, jobcommand, dry_run)

def predict(dry_run):
    create_jobs(dry_run=dry_run, mode='pred', ext='_predict')


def to_slurm(jobname, jobcommand, dry_run):
    with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name"+"=" + jobname + "\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
        # slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write("#SBATCH -N 1\n")
        slurmfile.write("#SBATCH -c 1\n")
        slurmfile.write("#SBATCH -p gpu\n")
        slurmfile.write("#SBATCH --gres=gpu:1\n")
        slurmfile.write("#SBATCH --mem=30000\n")
        slurmfile.write(jobcommand)

    if not dry_run:
        os.system("sbatch slurm_scripts/" + jobname + ".slurm &")

dryrun = False
run_experiment(dryrun)
# predict(dryrun)
