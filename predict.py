import os

dryrun = True
folder = 'logs2'


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


for exp in os.listdir(folder):
    # tosave = os.path.join(folder,exp)
    command = 'th main.lua -name ' + exp +  " -mode exp"
    # if dryrun:
    #     print command
    # else:
    #     to_slurm(exp + '_predict', command)
    #     # os.system(command)
    to_slurm(exp + '_predict', command, dryrun)
