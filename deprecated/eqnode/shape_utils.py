import torch
import numpy as np

from functools import lru_cache

@lru_cache(maxsize=32)
def order_index(init_dim, n_tile, cuda_device=None):
    order_index = np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
    )
    order_index = torch.LongTensor(order_index)
    if cuda_device is not None:
        order_index = order_index.to(cuda_device)
    return order_index
    

def tile(a, dim, n_tile, reindex=False):
    """
    Tiles a pytorch tensor along one an arbitrary dimension.

    Parameters
    ----------
    a : PyTorch tensor
        the tensor which is to be tiled
    dim : integer like
        dimension along the tensor is tiled
    n_tile : integer like
        number of tiles

    Returns
    -------
    b : PyTorch tensor
        the tensor with dimension `dim` tiled `n_tile` times
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    if reindex:
        index = order_index(init_dim, n_tile, a.device)
        a = torch.index_select(a, dim, index)
    return a
