import torch
import numpy
from .test_systems import *
from .particle_utils import remove_mean, compute_mean


class _Structure(torch.nn.Module):
    
    def __init__(self, structure):
        super().__init__()
        self._structure = torch.nn.Parameter(structure)
    
    def forward(self):
        self._structure
    
    @property
    def structure(self):
        return self._structure
    

def minimize_structure(energy, structure, n_iter=1000, lr=5e-3):
    structure = structure.clone()
    s = _Structure(structure).to(structure)
    optim = torch.optim.Adam(s.parameters(), lr=lr)
    for i in range(n_iter):
        optim.zero_grad()
        loss = energy(s.structure)
        loss.sum().backward()
        optim.step()
    return loss, s.structure.data
        


class LangevinIntegrator():
    
    def __init__(self, potential, n_particles, n_dimension, k=1., temperature=1., dt=1., friction=0.1, mass=1.):
        self._potential = potential
        self._k = k
        self._temperature = temperature
        self._dt = dt
        self._friction = friction
        self._mass = mass
        self._n_particles = n_particles
        self._n_dimension = n_dimension
        
    def _draw_maxwell_boltzmann_velocities(self, x):
        self._mass = 1.
        sigma = self._k * self._temperature / self._mass
        velocities = torch.Tensor(x.shape).to(x).normal_() * np.sqrt(sigma)
        return velocities
        
    def _propagate_R(self, x, v, h): 
        x += h * v
        return x, v

    def _propagate_V(self, x, v, h):
        self._state.requires_grad_(True)
        potential_energy = self._potential(self._state).sum()
        grad = -torch.autograd.grad(potential_energy, self._state)[0]
        self._state.requires_grad_(False)
        v += h * grad / self._mass
        return x, v

    def _propagate_O(self, x, v, h):
        a, b = np.exp(-self._friction * h), np.sqrt(1 - np.exp(-2 * self._friction * h))
        v *= a
        v += b * self._draw_maxwell_boltzmann_velocities(x)
        return x, v
    
    def _step_ovrvo(self):
        self._propagate_O(self._state, self._velocity, self._dt / 2)
        self._propagate_V(self._state, self._velocity, self._dt / 2)
        self._propagate_R(self._state, self._velocity, self._dt)
        self._propagate_V(self._state, self._velocity, self._dt / 2)
        self._propagate_O(self._state, self._velocity, self._dt / 2)
        self._state = remove_mean(self._state, self._n_particles, self._n_dimension)
        
    def _step_vrorv(self):
        self._propagate_V(self._state, self._velocity, self._dt / 2)
        self._propagate_R(self._state, self._velocity, self._dt / 2)
        self._propagate_O(self._state, self._velocity, self._dt)
        self._propagate_R(self._state, self._velocity, self._dt / 2)
        self._propagate_V(self._state, self._velocity, self._dt / 2)
        self._state = remove_mean(self._state, self._n_particles, self._n_dimension)

    def run(self, state, n_steps, n_report_every=10, burnin=0, scheme="vrorv"):
        self._states = []
        self._state = state.clone()
        self._state = self._state
        self._velocity = self._draw_maxwell_boltzmann_velocities(self._state).to(state)
        
        if scheme == "vrorv":
            step = self._step_vrorv
        elif scheme == "ovrvo":
            step = self._step_ovrvo
        else:
            raise ValueError("scheme must be either `vrorv` or `ovrvo`")
            
        for i in range(n_steps):
            step()
            if i % n_report_every == 0:
                self._states.append(self._state.clone())
        return torch.stack(self._states[burnin//n_report_every:], dim=1)