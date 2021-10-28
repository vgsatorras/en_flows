import numpy as np
import torch
from .densenet import is_list_or_tuple
import matplotlib.pyplot as plt

class BatchIterator(object):
    def __init__(self, n_elems, n_batch):
        self._indices = np.arange(n_elems)
        self._n_elems = n_elems
        self._n_batch = n_batch
        self._pos = 0
        self._reset()

    def _reset(self):
        self._pos = 0
        np.random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self):
        # subtract n_batch to have a constant batch size
        if self._pos > self._n_elems - self._n_batch:
            self._reset()
            raise StopIteration
        n_collected = min(self._n_batch, self._n_elems - self._pos)
        batch = self._indices[self._pos : self._pos + n_collected]
        self._pos = self._pos + n_collected
        return batch

    def next(self):
        return self.__next__()

    def __len__(self):
        return self._n_elems // self._n_batch

        
    
def linlogcut_torch(x, a=10, b=100):
    """ Function which is linear until a, logarithmic until b and then constant.
    y = x                  x <= a
    y = a + log(x-a)   a < x < b
    y = a + log(b-a)   b < x
    """
    # cutoff x after b - this should also cutoff infinities
    x = torch.where(x < b, x, b * torch.ones(x.shape).to(x))
    # log after a
    y = a + torch.where(x < a, x - a, torch.log(x - a + 1))
    # make sure everything is finite ?notwendig?
    #y = tf.where(torch.is_finite(y), y, b * torch.ones(tf.shape(y)))
    return y

def assert_numpy(x, arr_type=None):
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
    if is_list_or_tuple(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    if arr_type is not None:
        x = x.astype(arr_type)
    return x

class IndexBatchIterator(object):
    def __init__(self, n_elems, n_batch):
        """
            Produces batches of length `n_batch` of an index set
            `[1, ..., n_elems]` which are sampled randomly without
            replacement.

            If `n_elems` is not a multiple of `n_batch` the last sampled
            batch will be truncated.

            After the iteration throw `StopIteration` its random seed
            will be reset.

            Parameters:
            -----------
            n_elems : Integer
                Number of elements in the index set.
            n_batch : Integer
                Number of batch elements sampled.

        """
        self._indices = np.arange(n_elems)
        self._n_elems = n_elems
        self._n_batch = n_batch
        self._pos = 0
        self._reset()

    def _reset(self):
        self._pos = 0
        np.random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos > self._n_elems - self._n_batch:
            self._reset()
            raise StopIteration
        n_collected = min(self._n_batch, self._n_elems - self._pos)
        batch = self._indices[self._pos : self._pos + n_collected]
        self._pos = self._pos + n_collected
        return batch

    def __len__(self):
        return self._n_elems // self._n_batch

    def next(self):
        return self.__next__()

class LossReporter:
    """
        Simple reporter use for reporting losses and plotting them.
    """

    def __init__(self, *labels):
        self._labels = labels
        self._n_reported = len(labels)
        self._raw = [[] for _ in range(self._n_reported)]

    def report(self, *losses):
        assert len(losses) == self._n_reported
        for i in range(self._n_reported):
            self._raw[i].append(assert_numpy(losses[i]))

    def plot(self, n_smooth=10):
        fig, axes = plt.subplots(self._n_reported, sharex=True)
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        fig.set_size_inches((8, 4 * self._n_reported), forward=True)
        for i, (label, raw, axis) in enumerate(zip(self._labels, self._raw, axes)):
            raw = assert_numpy(raw).reshape(-1)
            kernel = np.ones(shape=(n_smooth,)) / n_smooth
            smoothed = np.convolve(raw, kernel, mode="valid")
            axis.plot(smoothed)
            axis.set_ylabel(label)
            if i == self._n_reported - 1:
                axis.set_xlabel("Iteration")

    def show(self):
        plt.show()

    def recent(self, n_recent=1):
        return np.array([raw[-n_recent:] for raw in self._raw])