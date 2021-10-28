# %%


# %%

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from eqnode.train_utils import assert_numpy, IndexBatchIterator, LossReporter, \
    BatchIterator
from eqnode.densenet import DenseNet
from eqnode.flows2 import HutchinsonEstimator

from eqnode.test_systems import MultiDoubleWellPotential
from eqnode.particle_utils import remove_mean
from eqnode.bg import DiffEqFlow
from eqnode.densenet import DenseNet
from eqnode.dynamics import SchNet, SimpleEqDynamics
from eqnode.kernels import RbfEncoder, kernelize_with_rbf, compute_gammas
from eqnode.priors import MeanFreeNormalPrior

# %%

# first define system dimensionality and a target energy/distribution

dim = 8
n_particles = 4

# DW parameters
a = 0.9
b = -4
c = 0
offset = 4

# %%

target = MultiDoubleWellPotential(dim, n_particles, a, b, c, offset)

# %%

data, idx = np.load("MCMC_data/dw4-dataidx.npy", allow_pickle=True)

# %%

idx = np.random.choice(len(data), len(data), replace=False)

# %%

data = data.reshape(-1, dim)
data = remove_mean(data, n_particles, dim // n_particles)

# %%

# now set up a prior

prior = MeanFreeNormalPrior(dim, n_particles)

# %%

n_rbfs = 50
d_max = 16
n_features = 16

kernel_mus = torch.linspace(0., 8., n_rbfs).cuda()
kernel_gammas = torch.ones(n_rbfs).cuda() * 0.5
rbf_encoder = RbfEncoder(kernel_mus, kernel_gammas.log(), trainable=False)

schnet = SchNet(
    transformation=DenseNet([n_features, 8, 4, 1], activation=torch.nn.Tanh()),
    cfconv1=DenseNet([n_rbfs, 32, 32, n_features], activation=torch.nn.Tanh()),
    cfconv2=DenseNet([n_rbfs, 32, 32, n_features], activation=torch.nn.Tanh()),
    cfconv3=DenseNet([n_rbfs, 32, 32, n_features], activation=torch.nn.Tanh()),
    feature_encoding=DenseNet([1, n_features], activation=torch.nn.Tanh()),
    rbf_encoder=rbf_encoder,
    n_particles=n_particles,
    n_dimesnion=dim // n_particles,
    features=n_features,
).cuda()

# simple_dynamics = SimpleEqDynamics(
#     transformation=DenseNet([n_rbfs, 64, 32, 1], activation=torch.nn.Tanh()),
#     rbf_encoder=rbf_encoder,
#     n_particles=n_particles,
#     n_dimesnion=dim // n_particles,
#     n_rbfs = n_rbfs,
# ).cuda()


# %% md

# choose
# between
# SchNet and simple
# gradient
# flow

# %%

# gradient field with SchNet
dynamics = HutchinsonEstimator(schnet, brute_force=False)

# simple gradient field
# dynamics=HutchinsonEstimator(simple_dynamics, brute_force=False)


# %%

flow = DiffEqFlow(
    dynamics=dynamics
).cuda()

# %%

n_data = 1000
data_smaller = data[idx[:n_data]].clone()

# %%

# initial training with likelihood maximization on data set


n_batch = 64
batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

optim = torch.optim.AdamW(flow.parameters(), lr=5e-3, amsgrad=True,
                          weight_decay=0.01)

n_epochs = 100
n_report_steps = 1

reporter = LossReporter("NLL")

# %%

dynamics._brute_force = False

# %%

for epoch in range(n_epochs):
    for it, idxs in enumerate(batch_iter):
        batch = data_smaller[idxs].cuda()

        optim.zero_grad()
        dynamics.reset_noise()
        z, delta_logp = flow(batch, inverse=True)
        nll = (prior.energy(z).view(-1) + delta_logp.view(-1)).mean()

        nll.backward()

        reporter.report(nll)

        optim.step()

        if it % n_report_steps == 0:
            print("\repoch: {0}, iter: {1}/{2}, NLL: {3:.4}".format(
                epoch,
                it,
                len(batch_iter),
                *reporter.recent(1).ravel()
            ), end="")

# %%

reporter.plot()

# %%

# use OTD in the evaluation process
flow._use_checkpoints = False

# %%

dynamics._brute_force = True

# %%

data_nll = 0.
batch_iter = BatchIterator(len(data_smaller), n_batch)
for it, batch_idxs in enumerate(batch_iter):
    if it > 100:
        break
    x = torch.Tensor(data_smaller[batch_idxs]).cuda()
    z, delta_logp = flow(x, inverse=True)
    nll = (prior.energy(z).view(-1) + delta_logp.view(-1)).mean()
    print("\r{}".format(it), nll, end="")
    data_nll += nll
data_nll = data_nll / (it + 1)

# %%

holdout_nll = 0.