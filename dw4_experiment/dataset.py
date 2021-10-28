import numpy as np
import torch

from deprecated.eqnode.particle_utils import remove_mean
from deprecated.eqnode.test_systems import MultiDoubleWellPotential
from deprecated.eqnode.train_utils import IndexBatchIterator, BatchIterator


def get_data(args, partition, batch_size):
    if args.data == 'dw4':
        return get_data_dw4(args.n_data, partition, batch_size)
    elif args.data == 'lj13':
        return get_data_lj13(args.n_data, partition, batch_size)
    else:
        raise ValueError


def get_data_dw4(n_data, partition, batch_size, n_particles_test=1000, n_particles_val=1000):
    dim = 8
    n_particles = 4
    # DW parameters
    a = 0.9
    b = -4
    c = 0
    offset = 4

    # variable 'target' and function 'MultiDoubleWellPotential' not used in this experiment
    target = MultiDoubleWellPotential(dim, n_particles, a, b, c, offset)

    # shape data: [1000000, 8], idx is ignored
    data, idx = np.load("dw4_experiment/data/dw4-dataidx.npy", allow_pickle=True)

    # used once to reshuffle for cutting the data (he just uses 1000 samples)


    # The mean is computed per each set of nodes in each sample
    data = remove_mean(data, n_particles, dim // n_particles)

    data = data.reshape(-1, dim)
    data_train = data[0:len(data) - n_particles_test - n_particles_val]
    data_val = data[len(data) - n_particles_test - n_particles_val: len(data) - n_particles_test]
    data_test = data[len(data) - n_particles_val: len(data)]
    #idx = np.random.choice(len(data_train), len(data_train), replace=False)

    if partition == 'train':
        assert n_data <= len(data_train)
        data_smaller = data_train[:n_data].clone()
    elif partition == 'val':
        data_smaller = data_val
    elif partition == 'test':
        data_smaller = data_test


    n_batch = batch_size
    batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

    return data_smaller, batch_iter


def get_data_lj13(n_data, partition, batch_size, n_particles_val=1000):
    n_particles = 13
    n_dimension = 3
    dim = n_particles * n_dimension



    if partition == 'train':
        data = np.load("dw4_experiment/data/holdout_data_LJ13.npy")
        idx = np.load("dw4_experiment/data/idx_LJ13.npy")
    elif partition == 'val':
        data = np.load("dw4_experiment/data/all_data_LJ13.npy")[1000:2000]
    elif partition == 'test':
        data = np.load("dw4_experiment/data/all_data_LJ13.npy")[0:1000]
    else:
        raise Exception("Wrong partition")

    data = data.reshape(-1, dim)
    data = torch.Tensor(data)
    data = remove_mean(data, n_particles, dim // n_particles)

    if partition == 'train':
        data = data[idx[:n_data]].clone()

    batch_iter = BatchIterator(len(data), batch_size)

    return data, batch_iter


def plot_data(sample):
    import matplotlib.pyplot as plt
    x = sample[:, 0]
    y = sample[:, 1]
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    n_samples = 100
    data, _ = get_data_dw4(n_samples, 'train')
    data = data.view(n_samples, 4, 2)
    for i in range(n_samples):
        plot_data(data[i])
