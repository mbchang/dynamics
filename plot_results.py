import os
import pprint
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    plot_lstm()
    plot_ff()
    plot_ff_lstm()
    plot_2balls()
    plot_3balls()
    plot_4balls()
