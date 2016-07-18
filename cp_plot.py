import os

experiments = [
                # 'balls_n3_t60_ex50000',
                # 'balls_n3_t60_ex50000,balls_n5_t60_ex50000,balls_n7_t60_ex50000__balls_n10_t60_ex50000',
                # 'balls_n3_t60_ex50000,balls_n5_t60_ex50000__balls_n10_t60_ex50000',
                # 'balls_n3_t60_ex50000,balls_n5_t60_ex50000__balls_n7_t60_ex50000',
                # 'balls_n3_t60_ex50000__balls_n10_t60_ex50000',
                # 'balls_n3_t60_ex50000__balls_n5_t60_ex50000',
                # 'balls_n5_t60_ex10000',
                # 'balls_n5_t60_ex10000_fr',
                # 'balls_n5_t60_ex10000_gf',
                # 'balls_n5_t60_ex50000',
                # 'balls_n7_t60_ex50000',
                # 'tower_n10_t120_ex50000__tower_n10_t120_ex50000',
                'balls_n2_t60_ex50000__balls_n2_t60_ex50000',
                'balls_n2_t60_ex50000__balls_n2_t60_ex50000_batchnorm',
                'balls_n3_t60_ex50000__balls_n3_t60_ex50000_batchnorm',
                'balls_n3_t60_ex50000__balls_n3_t60_ex50000'
                ]

# specify paths
out_root = 'opmjlogs'
in_root = '/om/user/mbchang/physics/lua/logs'
copy_prefix = 'rsync -avz --exclude \'*.t7\' mbchang@openmind7.mit.edu:'
remote_prefix = '/om/user/mbchang/physics/lua/logs/'

# copy
remote_paths = remote_prefix + '\{' + ','.join(['\\"' + e + '\\"' for e in experiments]) + '\} '
# remote_paths = remote_prefix + ','.join(['\\"' + e + '\\"' for e in experiments])# + '\} '
command = copy_prefix + remote_paths + out_root

response = raw_input('Running command:\n\n' + command + '\n\nProceed?[y/n]')
if response == 'y':
    os.system(command)
elif response != 'n':
    response = raw_input('Running command:\n\n' + command + '\nProceed?[y/n]')
else:
    print 'Not running command.'

# plot
for experiment_folder in experiments:
    experiment_folder = os.path.join(out_root, experiment_folder)
    command = 'th plot_results.lua -i ' + experiment_folder
    print command
    os.system(command)
