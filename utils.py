import h5py
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def num_to_one_hot(array, discrete_values):
    """
        The values in the array come from the list discrete_values.
        For example, if discrete_values is [0.33, 1.0, 3.0] then all the values in this
        array are in [0.33, 1.0, 3.0].

        This method adds another axis (last axis) and makes these values into a one-hot
        encoding of those discrete values. For example, if the array was shape (4,10)
        and len(discrete_values) was 3, then this method will produce an array
        with shape (4,10,3)
    """
    n_values = len(discrete_values)
    broadcast = tuple([1 for i in xrange(array.ndim)] + [n_values])
    array = np.tile(np.expand_dims(array,array.ndim+1), broadcast)
    for i in xrange(n_values): array[...,i] = array[...,i] == discrete_values[i]
    return array

def one_hot_to_num(one_hot_vector, discrete_values):
    """
        one_hot_vector: (n,) one hot vector
        discrete_values is a list of values that the onehot represents

        assumes that the one_hot_vector only as one 1

        return the VALUE in discrete_values that the one_hot_vector refers to
    """
    # print one_hot_vector
    # TODO: this should return the actual value, not the index!
    assert sum(one_hot_vector) == 1  # it had better have one 1
    return discrete_values[int(np.nonzero(one_hot_vector)[0])]

def stack(list_of_nparrays):
    """
        input
            :nparray: list of numpy arrays
        output
            :stack each numpy array along a new dimension: axis=0
    """
    st = lambda a: np.vstack(([np.expand_dims(x,axis=0) for x in a]))
    stacked = np.vstack(([np.expand_dims(x,axis=0) for x in list_of_nparrays]))
    assert stacked.shape[0] == len(list_of_nparrays)
    assert stacked.shape[1:] == list_of_nparrays[0].shape
    return stacked

def save_dict_to_hdf5(dataset, dataset_name, dataset_folder):
    print '\nSaving', dataset_name
    h = h5py.File(os.path.join(dataset_folder, dataset_name + '.h5'), 'w')
    print dataset.keys()
    for k, v in dataset.items():
        print 'Saving', k
        h.create_dataset(k, data=v, dtype='float64')
    h.close()
    print 'Reading saved file'
    g = h5py.File(os.path.join(dataset_folder, dataset_name + '.h5'), 'r')
    for k in g.keys():
        print k
        print g[k][:].shape
    g.close()

def load_hdf5(filename, datapath):
    """
        Loads the data stored in the datapath stored in filename as a numpy array
    """
    data = load_dict_from_hdf5(filename)
    return data[datapath]

def load_dict_from_hdf5(filepath):
    data = {}
    g = h5py.File(filepath, 'r')
    for k in g.keys():
        data[k] = g[k][:]
    return data

def subtensor_equal(subtensor, tensor, dim):
    """
        Return if subtensor, when broaadcasted along dim
    """
    num_copies = tensor.shape[dim]

    subtensor_stack = np.concatenate([subtensor for s in num_copies], dim=dim)

    return subtensor_stack == tensor


if __name__ == "__main__":
    G_w_width, G_w_height = 640.0,480.0
    G_max_velocity, G_min_velocity = 2*4500.0, -2*4500.0

    a = load_dict_from_hdf5('/Users/MichaelChang/Documents/Researchlink/SuperUROP/Code/opdata/3/trainset.h5')
    b = a['worldm1_np=2_ng=0particles']
    ex = b[3]
    state = ex[:,:,:4]
    print state.shape
    pos = state[:,:,:2]*G_w_width  # (numobjects, timesteps, [px,py])
    vel = state[:,:,2:]*G_max_velocity/1000*5  # (numobjects, timesteps, [vx,vy])

    predpos = np.copy(pos)

    for i in range(0,19):
        predpos[0,i+1,:] = predpos[0,i,:] + vel[0,i+1,:]  # so this method is not completely correct because of subsampling.

    print pos[0]
    print predpos[0]
    # try plotting position
    # plot_pos = np.vstack((pos[0,:,0],np.arange(len(pos[0]))))
    # print (pos[0])
    # print(pos[0,:].shape)
    # plt.plot(vel[0,:,1])
    plt.plot(pos[0,:,0],pos[0,:,1])
    plt.plot(predpos[0,:,0],predpos[0,0:,1])
    # plt.plot(vel[0,:,0],vel[0,:,1])
    plt.show()







# ok, there are certain cases where this fails. Where?

# [[[ 0.2076019   0.3418827  -0.01643823 -0.26593644]
#   [ 0.20644609  0.32318405 -0.01643823 -0.26593644]
#   [ 0.20529027  0.30448539 -0.01643823 -0.26593644]]
#
#  [[ 0.15053652  0.09872539  0.24891007 -0.09015184]
#   [ 0.16453771  0.09606309  0.24891007  0.08113666]
#   [ 0.1820392   0.10176801  0.24891007  0.08113666]]]
