import os
import pprint
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import utils as u

root = '/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/dynamics/oplogs'


# fig,ax = plt.subplots
#
# # assuming they all have the same domain
# domain = 100 # TODO CHANGE ME
# for curve in curves:
#     ax.plot(domain, curve)

def plot_experiments(exp_list, dataset, outfolder, outfile):
    ys = []
    xs = []
    print_every = 100

    fig, ax = plt.subplots()
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

    # read in the experminet.logs from exp_list
    for exp, num_batches, label in exp_list:
        # read exp.log
        y = read_log_file(os.path.join(*[root,exp,'experiment.log']))[dataset]
        x = range(num_batches, num_batches*(len(y)+1), num_batches)

        # get label
        # end = exp.find('balls')
        # begin = exp[:end].rfind('_')+1
        # label = exp[begin:end] + ' balls'

        ax.plot(x,y, label=label)
        # plt.setp(ax.get_xticklabels(), fontsize=10, rotation=60)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Log MSE Loss')
    plt.savefig(os.path.join(outfolder, outfile))


# return np array from log file
def read_log_file(log_file):
    data = {'train':[],'val':[],'test':[]}
    with open(log_file, 'r') as f:
        raw = f.readlines()
    for t in xrange(1,len(raw)):
        [train, val, test] = raw[t].split('\t')[:3]
        data['train'].append(train)
        data['val'].append(val)
        data['test'].append(test)
    return data

def plot_lstm():
    exp_list = [('18_layers_3_lrdecay_0.99_dataset_folder_14_2balls_lr_0.0003_sharpen_1_model_lstmobj', 3600, '2 balls lstm'),  # 3600
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_3balls_lr_0.0003_sharpen_1_model_lstmobj', 5400, '3 balls lstm'), # 5400
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_4balls_lr_0.0003_sharpen_1_model_lstmobj', 7200, '4 balls lstm')]  # 7200
    plot_experiments(exp_list, 'test', root, '18_layers_3_lrdecay_0.99_lr=0.0003_sharpen_1_model_lstm.png')

def plot_ff():
    exp_list = [('18_layers_3_lrdecay_0.99_dataset_folder_14_2balls_lr_0.0003_sharpen_1_model_ffobj', 3600, '2 balls mlp'),  # 3600
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_3balls_lr_0.0003_sharpen_1_model_ffobj', 5400, '3 balls mlp'), # 5400
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_4balls_lr_0.0003_sharpen_1_model_ffobj', 7200, '4 balls mlp')]  # 7200
    plot_experiments(exp_list, 'test', root, '18_layers_3_lrdecay_0.99_lr=0.0003_sharpen_1_model_mlp.png')

def plot_ff_lstm():
    exp_list = [('18_layers_3_lrdecay_0.99_dataset_folder_14_2balls_lr_0.0003_sharpen_1_model_ffobj', 3600, '2 balls mlp'),  # 3600
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_3balls_lr_0.0003_sharpen_1_model_ffobj', 5400, '3 balls mlp'), # 5400
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_4balls_lr_0.0003_sharpen_1_model_ffobj', 7200, '4 balls mlp'),  # 7200
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_2balls_lr_0.0003_sharpen_1_model_lstmobj', 3600, '2 balls lstm'),  # 3600
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_3balls_lr_0.0003_sharpen_1_model_lstmobj', 5400, '3 balls lstm'), # 5400
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_4balls_lr_0.0003_sharpen_1_model_lstmobj', 7200, '4 balls lstm')]  # 7200
    plot_experiments(exp_list, 'test', root, '18_layers_3_lrdecay_0.99_lr=0.0003_sharpen_1_model_mlp_lstm.png')

def plot_2balls():
    exp_list = [('18_layers_3_lrdecay_0.99_dataset_folder_14_2balls_lr_0.0003_sharpen_1_model_ffobj', 3600, '2 balls mlp'),  # 3600
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_2balls_lr_0.0003_sharpen_1_model_lstmobj', 3600, '2 balls lstm')]  # 3600
    plot_experiments(exp_list, 'test', root, '18_layers_3_lrdecay_0.99_lr=0.0003_sharpen_1_2balls.png')

def plot_3balls():
    exp_list = [('18_layers_3_lrdecay_0.99_dataset_folder_14_3balls_lr_0.0003_sharpen_1_model_ffobj', 5400, '3 balls mlp'), # 5400
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_3balls_lr_0.0003_sharpen_1_model_lstmobj', 5400, '3 balls lstm')] # 5400
    plot_experiments(exp_list, 'test', root, '18_layers_3_lrdecay_0.99_lr=0.0003_sharpen_1_3balls.png')

def plot_4balls():
    exp_list = [('18_layers_3_lrdecay_0.99_dataset_folder_14_4balls_lr_0.0003_sharpen_1_model_ffobj', 7200, '4 balls mlp'),  # 7200
                ('18_layers_3_lrdecay_0.99_dataset_folder_14_4balls_lr_0.0003_sharpen_1_model_lstmobj', 7200, '4 balls lstm')]  # 7200
    plot_experiments(exp_list, 'test', root, '18_layers_3_lrdecay_0.99_lr=0.0003_sharpen_1_4balls.png')


def plot_hid_state(h5_file_folder):
    fname = 'hidden_state_all_testfolders'
    # TODO! The constants here are hard coded!!!
    data = u.load_dict_from_hdf5(os.path.join(h5_file_folder, fname))
    # keys: 'euc_dist', 'euc_dist_diff', 'effects_norm'
    all_euc_dist = data['euc_dist']
    all_euc_dist_diff = data['euc_dist_diff']
    all_effects_norm = data['effects_norm']

    neg_vel_idx = np.argwhere(all_euc_dist_diff < 0)
    pos_vel_idx = np.argwhere(all_euc_dist_diff >= 0)

    neg_vel = all_euc_dist_diff[neg_vel_idx]
    pos_vel = all_euc_dist_diff[pos_vel_idx]

    euc_dist_neg_vel = all_euc_dist[neg_vel_idx]
    euc_dist_pos_vel = all_euc_dist[pos_vel_idx]

    norm_neg_vel = all_effects_norm[neg_vel_idx]
    norm_pos_vel = all_effects_norm[pos_vel_idx]

    outfile = 's'

    plot_hid_state_helper(os.path.join(h5_file_folder, fname)+'_toward.png', euc_dist_neg_vel, norm_neg_vel, neg_vel)
    plot_hid_state_helper(os.path.join(h5_file_folder, fname)+'_away.png', euc_dist_pos_vel, norm_pos_vel, pos_vel)  # do the color gradient based on neg_vel


    # -- plot_hidden_state(pp.infolder..'/'..fname..'.png', all_euc_dist, all_effects_norm, pp.infolder)
    # plot_hid_state(pp.infolder..'/'..fname..'_toward.png', euc_dist_neg_vel, norm_neg_vel)
    # plot_hid_state(pp.infolder..'/'..fname..'_away.png', euc_dist_pos_vel, norm_pos_vel)

def plot_hid_state_helper(outfile, x, y, c):
    plt.scatter(x,y, c=c, marker='.')
    plt.legend()
    plt.title('Pairwise Hidden State as a Function of Distance from Focus Object')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Hidden State Norm')
    plt.savefig(outfile)
    # plt.show()



if __name__ == "__main__":
    # plot_lstm()
    # plot_ff()
    # plot_ff_lstm()
    # plot_2balls()
    # plot_3balls()
    # plot_4balls()
    plot_hid_state('logs/balls_n3_t60_ex20,balls_n6_t60_ex20,balls_n5_t60_ex20/hidden_state_all_testfolders.h5')
