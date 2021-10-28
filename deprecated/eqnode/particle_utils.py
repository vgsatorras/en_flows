import torch


def compute_mean(samples, n_particles, n_dimensions, keepdim=False):
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        mean = torch.mean(samples, dim=1, keepdim=keepdim)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        mean = samples.mean(axis=1, keepdims=keepdim)
    return mean


def remove_mean(samples, n_particles, n_dimensions):
    shape = samples.shape
    # if isinstance(samples, torch.Tensor):
    samples = samples.view(-1, n_particles, n_dimensions)
    samples = samples - torch.mean(samples, dim=1, keepdim=True)
    samples = samples.view(*shape)
    # else:
    #     samples = samples.reshape(-1, n_particles, n_dimensions)
    #     samples = samples - samples.mean(axis=1, keepdims=True)
    #     samples = samples.reshape(*shape)
    return samples
