from collections import Counter
import numpy as np
from .distances import distance_vectors, distances_from_vectors
import torch


def configuration_distance(conf1, conf2, n_particles, n_dimension, p=1):
    dists1 = distances_from_vectors(
        distance_vectors(conf1.view(-1, n_particles, n_dimension))
    ).cpu().detach().numpy()
    dists1.reshape(-1).sort()
    dists2 = distances_from_vectors(
        distance_vectors(conf2.view(-1, n_particles, n_dimension))
    ).cpu().detach().numpy()
    dists2.reshape(-1).sort()
    conf_distance = np.linalg.norm(dists1.reshape(-1) - dists2.reshape(-1), ord=p)
    return conf_distance


def count_states(minima, n_particles=13, n_dimension=3, minimum_states=None, threshold=3):
    add = False
    if minimum_states is None:
        minimum_states = [minima[0]]
        add = True
    counter = Counter()

    for config in minima:
        for i, minimum in enumerate(minimum_states):
            if configuration_distance(config, minimum, n_particles, n_dimension) < threshold:
                counter[i] += 1
                break
        else:
            if add:
                minimum_states.append(config)
            counter[i+1] = 1
    return counter, minimum_states

def is_minimum(x, energy, threshold):
    x = x.clone()
    x = x.requires_grad_(True)
    e = energy(x)
    J = torch.autograd.grad(e.sum(), x, create_graph=True, retain_graph=True)[0]
    jtol = J.pow(2).sum(dim=-1)
    necessary = jtol < threshold
    outs =[J[:, i].sum() for i in range(J.shape[-1])]
    H = [torch.autograd.grad(out, x, create_graph=True, retain_graph=True)[0] for out in outs]
    H = torch.stack(H, dim=-1)
    _, s, _ = torch.svd(H, compute_uv=False)
    sel = s.min(-1)[0] > 1e-9
    sufficient = torch.zeros_like(necessary).bool()
    eig = torch.symeig(H[sel])[0]
    sufficient[sel] = torch.all((eig > 0), -1)
    return necessary & sufficient

def optimize(x, energy, step_size=1e-3, max_steps=int(1e5), print_iter=None, threshold=1e-6, noise=0.,
             jac_check_steps=1):
    y = x.clone()
    y = y.requires_grad_(True)
    if noise > 0:
        nu = torch.Tensor(*x.shape).normal_()
    def _newton(z):
        e = energy(z)
        if z.grad is not None and z.grad.data is not None:
            z.grad.data.zero_()
        e.sum().backward()
        z.data.add_(-step_size, z.grad.data)
        if noise > 0:
            z.data.add_(noise * nu)
        return y, e, z.grad.data
    for i in range(max_steps):
        #print(y)
        y, e, J = _newton(y)
        if i % jac_check_steps == 0:
            jtol = J.pow(2).sum(dim=-1)
            done = jtol.max() < threshold
        if print_iter and i % print_iter == 0:
            print(done.item(), e.mean().item(), jtol.max().item())
        if done:
            break
    return done, y, e, jtol

def bin_minima(minima_configurations, runs, potential):
    n_80 = np.empty(runs)
    n_70 = np.empty(runs)
    n_60 = np.empty(runs)
    for i, run in enumerate(minima_configurations):
        n = 0
        run_energy = potential._energy_torch(run)
        n_80[i] = len(run_energy[run_energy < -80])
        n += n_80[i]
        n_70[i] = len(run_energy[run_energy < -70]) - n
        n += n_70[i]
        n_60[i] = len(run_energy[run_energy < -60]) - n
    return n_80, n_70, n_60