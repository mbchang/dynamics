# have a method to do tower, and cloth

# have a method to parse flags

# have a method to run slurm jobs?

# have a method to generate from mj

# have a method to do torch stuff


# TODO: add in masses!


import os
import sys
import pprint
from collections import OrderedDict

def create_jobs(dry_run, ext):
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

    # paths
    # mj_root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/physics_worlds/'
    # torch_root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/'

    mj_root = '/om/user/mbchang/physics/js/'
    toch_root = '/om/user/mbchang/physics/lua'

    # world parameters
    num_objs = [4]  # how far should we go? Let's say the max is 20. Should we include 1?
    friction = [False]
    gravity = [False]
    masses = [False]  # TODO
    sizes = [True]
    num_obstacles = [False]
    envs = ['mixed']
    drastic_size = [True]

    # mj data generation
    steps = 60
    samples = 50000

    # generate json files
    generator = 'node ' + mj_root + 'demo/js/Demo.js'

    # generate balls
    # make this a method
    jobs = []
    for n in num_objs:
        for m in masses:
            for z in sizes:
                for o in num_obstacles:
                    for d in drastic_size:
                        for g in gravity:
                            for f in friction:
                                for e in envs:
                                    job = OrderedDict()
                                    job['e'] = e
                                    job['n'] = n
                                    job['t'] = steps
                                    job['s'] = samples
                                    job['m'] = m
                                    job['g'] = g
                                    job['f'] = f 
                                    job['z'] = z
                                    job['o'] = o
                                    job['d'] = d
                                    jobs.append(job)
                                    # jobs.append(OrderedDict({'e':e,'n':n, 't':samples, 'ex':steps,'g':g,'f':f}))

    if dry_run:
        print "NOT starting jobs:"
    else:
        print "Starting jobs:"

    # TODO: have an ordering in the flags in the jobame

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
            else:
                if flag != 'e': jobname = jobname + "_" + flag + "" + str(job[flag])
                flagstring = flagstring + " -" + flag + " " + str(job[flag])
        flagstring = flagstring
        jobname = job['e']+jobname

        jobcommand = generator + flagstring

        if local and not dry_run:
            if detach:
                os.system(jobcommand + ' 2> slurm_logs/' + jobname + '.err 1> slurm_logs/' + jobname + '.out &')
            else:
                os.system(jobcommand)

        else:
            to_slurm(jobname + ext, jobcommand, dry_run)

def generate_data(dry_run):
    create_jobs(dry_run=dry_run, ext='_js2')  # the js extension is for positionIterations and velocityIterations = 100, and runner.isFixed

def to_slurm(jobname, jobcommand, dry_run):
    print jobname
    print jobcommand + '\n'


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
generate_data(dry_run)

