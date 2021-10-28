import numpy as np
import torch

from .distances import distances_from_vectors, distance_vectors


class NormalPrior(object):
    
    def __init__(self, n_particles, n_dimensions, scale=1.):
        self._n_particles = n_particles
        self._n_dimensions = n_dimensions
        self._scale = scale
        
    def sample(self, n_samples):
        return torch.Tensor(n_samples, self._n_particles * self._n_dimensions).normal_() * self._scale

    def likelihood(self, xs):
        return 0.5 * xs.pow(2).view(xs.shape[0], -1).sum(dim=-1) / (self._scale ** 2)

class RepulsionPrior(object):
    
    def __init__(self, n_particles, n_dimensions, scale=1., eps=1., power=2.):
        self._n_particles = n_particles
        self._n_dimensions = n_dimensions
        self._scale = scale
        self._eps = eps
        self._power = power
        
    def _squash(self, x, eps=1., power=12):
        return (x / eps).pow(power)
    
    def _log_accept(self, ds, eps=1., power=12):
        return torch.log(torch.min(self._squash(ds, eps=eps, power=power), torch.ones_like(ds))).sum(-1).sum(-1)
    
    def sample(self, n_samples):
        n_kept = 0
        
        collected = []
        while(n_kept < n_samples):
            zs = torch.Tensor(n_samples, self._n_particles, self._n_dimensions).normal_() * self._scale
            ds = distances_from_vectors(distance_vectors(zs))
            log_accept = self._log_accept(ds, self._eps * self._scale, self._power)
            r = torch.Tensor(n_samples).uniform_()
            keep = np.log(r) < log_accept
            zs = zs[keep]
            collected.append(zs)
            n_kept += len(zs)
        return torch.cat(collected, dim=0)[:n_samples]

    def likelihood(self, xs):
        ds = distances_from_vectors(distance_vectors(xs))
        log_accept = self._log_accept(ds, self._eps * self._scale, self._power)
        return 0.5 * xs.pow(2).sum(dim=-1).sum(dim=-1) - log_accept

class Sampler(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def _sample_with_temperature(self, n_samples, temperature):
        raise NotImplementedError()

    def _sample(self, n_samples):
        raise NotImplementedError()

    def sample(self, n_samples, temperature=None):
        if temperature is not None:
            return self._sample_with_temperature(n_samples, temperature)
        else:
            return self._sample(n_samples)

class Energy(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x)[0]


class MeanFreeNormalPrior(Energy, Sampler):
    def __init__(self, dim, n_particles):
        super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._spacial_dims = dim // n_particles
    
    def _energy(self, x):
        x = self._remove_mean(x)
        return 0.5 * x.pow(2).sum(dim=-1, keepdim=True)

    def sample(self, n_samples, temperature=1.):
        x = torch.Tensor(n_samples, self._n_particles, self._spacial_dims).normal_()
        return self._remove_mean(x)

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._spacial_dims)
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = x.view(-1, self.dim)

        if torch.cuda.is_available():
            x = x.cuda()

        return x

