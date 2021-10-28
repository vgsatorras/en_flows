import torch
import numpy as np
# import torchani

from .metropolis import MetropolisGauss

from .distances import distance_vectors, distances_from_vectors


class DeepBoltzmannEnergy(object):
    def __init__(self, energy_fn):
        self._energy_fn = energy_fn

    def energy(self, x):
        return self._energy_fn(x)


def lennard_jones_energy_torch(r, eps=1.0, rm=1.0):
    lj = eps * ( (rm / r)**12 - 2 * (rm / r)**6 )
    return lj

def lennard_jones_energy_np(r, eps=1.0, rm=1.0):
    lj = eps * ( (rm / r)**12 - 2 * (rm / r)**6 )
    return lj

    
    
def double_well_energy_torch(x, a=1.0, b=-6.0, c=1.0):
    double_well = a * x[:, 0] + b * (x[:, 0] ** 2) + c * (x[:, 0] ** 4)
    harmonic_oscillator = 0.5 * x[:, 1:].pow(2).sum(dim=-1)
    return double_well + harmonic_oscillator


def double_well_energy_np(x, a=1.0, b=-6.0, c=1.0):
    double_well = a * x[:, 0] + b * (x[:, 0] ** 2) + c * (x[:, 0] ** 4)
    harmonic_oscillator = 0.5 * np.sum(x[:, 1:] ** 2, axis=-1)
    return double_well + harmonic_oscillator


def cos_torch(x, a=1.0, b=-6.0, c=1.0):
    return (
        torch.abs(x) * (1 + torch.cos(x ** 2) ** 2) - 1 / (x.pow(4) + 1e-3)
    ).sum(dim=-1)


def cos_np(x, a=1.0, b=-6.0, c=1.0):
    return np.sum(
        np.abs(x) * (1 + np.cos(x ** 2) ** 2 - 1 / (x ** 4 + 1e-3)), axis=-1
    )


def multi_well_torch(x, a=1.0, b=-6.0, c=1.0):
    return -torch.sum(
        torch.exp(-(x - torch.linspace(-6, 6, 4).view(1, 4)) ** 2), dim=-1
    )


def multi_well_np(x, a=1.0, b=-6.0, c=1.0):
    return -np.sum(
        np.exp(-(x - np.linspace(-6, 6, 4).reshape(1, 4)) ** 2), axis=-1
    )

def lennard_jones_energy_torch( r, eps=1.0, rm=1.0):
    lj = eps * ( (rm / r)**12 - 2 * (rm / r)**6 )
    return lj

class LennardJonesPotential(object):
    def __init__(
        self, n_particles, n_dims, eps=1.0, rm=1.0, temperature=1., oscillator=True, oscillator_scale=1.
    ):
        self._n_particles = n_particles
        self._n_dims = n_dims

        self._eps = eps
        self._rm = rm
        self.T = temperature
        self.oscillator = oscillator
        self._oscillator_scale = oscillator

    def _energy_torch(self, x):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )
        dists = dists.view(-1, 1)


        lj_energies = lennard_jones_energy_torch(dists, self._eps, self._rm)
        lj_energies =  lj_energies.view(n_batch, -1).sum(-1)

        if self.oscillator==True:
            osc_energies = 0.5 * self._remove_mean(x).pow(2).sum(-1).sum(-1)
            return (lj_energies + osc_energies * self._oscillator_scale) / self.T
        else:
            return lj_energies / self.T
        
    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dims)
        x = x - torch.mean(x, dim=1, keepdim=True)
        return x.view(-1, self._n_particles, self._n_dims)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy_torch(x).cpu().numpy()

    def energy(self, x):
        return self._energy_numpy(x)


class MultiDimerPotential(object):
    def __init__(
        self, n_particles, n_dims, distance_offset=0.0, double_well_coeffs=None,
        temperature=1.
    ):
        self._n_particles = n_particles
        self._n_dims = n_dims

        if double_well_coeffs is None:
            double_well_coeffs = {"a": 1.0, "b": -6.0, "c": 1.0}
        self._double_well_coeffs = double_well_coeffs

        self._distance_offset = distance_offset
        self._temperature = temperature

    def _energy_torch(self, x):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dims)

        dists = distances_from_vectors(
            distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )
        dists = dists.view(-1, 1)

        dists = dists - self._distance_offset

        energies = double_well_energy_torch(dists, **self._double_well_coeffs) / self._temperature
        return energies.view(n_batch, -1).sum(-1)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy_torch(x).cpu().numpy()

    def energy(self, x):
        return self._energy_numpy(x)


class ANIPotential(object):
    def __init__(
            self, n_particles, n_dims, atom_types, use_cuda=False
    ):
        self._n_particles = n_particles
        self._n_dims = n_dims
        if use_cuda:
            self.model = torchani.models.ANI1ccx().cuda()
        else:
            self.model = torchani.models.ANI1ccx()
        # Methan
        self.species = self.model.species_to_tensor(atom_types).unsqueeze(0)


    def _energy_torch(self, x):
        n_batch = x.shape[0]
        # print(x, n_batch)
        x = x.view(n_batch, self._n_particles, self._n_dims)

        _, energy = self.model((self.species.repeat(n_batch, 1), x))
        # print(energy)
        # TODO change this to generell form
        return energy + 40 # Hardocoded for Methan to get energies closer to 0

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy_torch(x).detach().cpu().numpy()

    def energy(self, x):
        return self._energy_numpy(x)
    
    def move_cuda(self):
        self.model.cuda()

def generate_samples(energy, n_dof, n_steps=10000, noise=0.5, init_state=None, init_state_scale=1.):
    traj = []
    if init_state is None:
        init_state = np.random.normal(size=n_dof) * init_state_scale
    sampler = MetropolisGauss(
        DeepBoltzmannEnergy(energy), init_state, noise=noise
    )
    sampler.run(nsteps=n_steps)
    traj.append(sampler.traj.copy())
    traj = np.concatenate(traj, axis=0)
    return traj


class TestSystem(object):
    def __init__(self, potential, n_dof):
        self._potential = potential
        self._n_dof = n_dof

    def sample_mcmc(self, n_steps, n_burnin=0, noise=0.1, n_redraws=1, n_stride=10, init_state_scale=1., init_state=None):
        subsets = []
        for i in range(n_redraws):
            subsets.append(
                generate_samples(
                    self._potential._energy_numpy,
                    noise=noise,
                    n_steps=n_steps,
                    n_dof=self._n_dof,
                    init_state_scale=init_state_scale,
                    init_state=init_state
                )[n_burnin::n_stride]
            )
        return np.concatenate(subsets)


class MultiDoubleWellPotential(torch.nn.Module):
    
    def __init__(self, dim, n_particles, a, b, c, offset):
        super().__init__()
        self._dim = dim
        self._n_particles = n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset
        
    @property
    def dim(self):
        return self._dim
    
    def _energy(self, x):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, -1)

        dists = distances_from_vectors(
            distance_vectors(x.view(n_batch, self._n_particles, -1))
        )
        dists = dists.view(-1, 1)

        dists = dists - self._offset

        energies = self._a * dists**4 + self._b * dists**2 + self._c
        return energies.view(n_batch, -1).sum(-1) / 2
        
    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x).view(-1, 1) / temperature
    
    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x)[0]