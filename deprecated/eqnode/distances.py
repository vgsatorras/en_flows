import torch
from .shape_utils import tile

from functools import lru_cache

@lru_cache(maxsize=32)
def diagonal_filter(n, m, cuda_device=None):
    filt = (torch.eye(n, m) == 0).view(n, m)
    if cuda_device is not None:
        filt = filt.to(cuda_device)
    return filt

def distance_vectors(x, remove_diagonal=True):
    """
    Computes the matrix `d` of all distance vectors between
    given input points where

        ``d_{ij} = x_{i} - x{j}``

    Parameters
    ----------
    x : PyTorch tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - x_i` should be included in the result

    Returns
    -------
    d : PyTorch tensor
        All-distnance matrix d.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.
    """
    r = tile(x.unsqueeze(2), 2, x.shape[1])
#     print(r.shape)
    r = r - r.permute([0, 2, 1, 3])
#     print(x.shape)
    filt = diagonal_filter(x.shape[1], x.shape[1], r.device)
    if remove_diagonal:
# # #         r = r.masked_select(filt).view(-1, x.shape[1], x.shape[1] - 1, x.shape[2])
        r = r[:, filt].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


def distance_vectors_v2(x, y, remove_diagonal=True):
    """
    Computes the matrix `d` of all distance vectors between
    given input points where

        ``d_{ij} = x_{i} - x{j}``

    Parameters
    ----------
    x : PyTorch tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - x_i` should be included in the result

    Returns
    -------
    d : PyTorch tensor
        All-distnance matrix d.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.
    """
    r1 = tile(x.unsqueeze(2), 2, x.shape[1])
    r2 = tile(y.unsqueeze(2), 2, y.shape[1])
    r = r1 - r2.permute([0, 2, 1, 3])
    if remove_diagonal:
        r = r[:, torch.eye(x.shape[1], x.shape[1]) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


def distances_from_vectors(r, eps=1e-6):
    return (r.pow(2).sum(dim=-1) + eps).sqrt()
