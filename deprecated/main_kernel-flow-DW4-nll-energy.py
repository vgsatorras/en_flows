import torch
import numpy as np
import matplotlib.pyplot as plt

from eqnode.train_utils import IndexBatchIterator, LossReporter, BatchIterator
from eqnode.densenet import DenseNet
from eqnode.dynamics import KernelDynamics
#from se3_dynamics.dynamics import OurDynamics

from eqnode.flows2 import HutchinsonEstimator
from eqnode.kernels import RbfEncoder

from eqnode.test_systems import MultiDoubleWellPotential
from eqnode.particle_utils import remove_mean
from eqnode.bg import DiffEqFlow
from eqnode.priors import MeanFreeNormalPrior
import argparse

parser = argparse.ArgumentParser(description='SE3')
parser.add_argument('--model', type=str, default='kernel_dynamics',
                    help='our_dynamics | kernel_dynamics')
args = parser.parse_args()

#check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# first define system dimensionality and a target energy/distribution
dim = 8
n_particles = 4

# DW parameters
a = 0.9
b = -4
c = 0
offset = 4

# Target energy function
target = MultiDoubleWellPotential(dim, n_particles, a, b, c, offset)

# Load dataset
data, idx = np.load("MCMC_data/dw4-dataidx.npy", allow_pickle=True)
idx = np.random.choice(len(data), len(data), replace=False)
data = data.reshape(-1, dim)
data  = remove_mean(data, n_particles, dim // n_particles)

# now set up a prior distribution
prior =  MeanFreeNormalPrior(dim, n_particles)

# set up some variables
n_dimension = dim // n_particles
d_max = 8

n_rbfs = 50
mus = torch.linspace(0, d_max, n_rbfs).to(device)
mus.sort()
gammas = 0.5 * torch.ones(n_rbfs).to(device)
mus_time = torch.linspace(0, 1, 10).to(device)
gammas_time = 0.3 * torch.ones(10).to(device)

# initialize model
if args.model == "our_dynamics":
    # raise Exception("%d dynamics not implemented yet" % args.model)
    # TO DO: include our dynamics here

    rbf_encoder = RbfEncoder(mus, gammas.log(), trainable=False)
    is_ode = True
    net_dynamics = OurDynamics(
        transformation=DenseNet([n_rbfs, 64, 32, 1], activation=torch.nn.Tanh()),
        rbf_encoder=rbf_encoder,
        n_particles=n_particles,
        n_dimesnion=dim // n_particles,
        n_rbfs=n_rbfs,
    ).to(device)
    dynamics = HutchinsonEstimator(net_dynamics, brute_force=False)
elif args.model == "kernel_dynamics":
    is_ode = False
    dynamics = KernelDynamics(n_particles, n_dimension, mus, gammas, optimize_d_gammas=True, optimize_t_gammas=True,
                          mus_time=mus_time, gammas_time=gammas_time).to(device)

else:
    raise Exception("Wrong model name %s" % args.model)

flow = DiffEqFlow(dynamics=dynamics).to(device)


# Truncate dataset
n_data = 1000
data_smaller = data[idx[:n_data]].clone()


def training_nll():
    '''
    Kholer samples from the distribution and optimizes the NLL on the prior.
    I think this is more like a unit test, the plots provided in the paper are computed in the other function -->
    '''

    # initial training with likelihood maximization on data set
    n_batch = 64
    batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

    optim = torch.optim.AdamW(flow.parameters(), lr=5e-3, amsgrad=True, weight_decay=1e-12)
    n_epochs = 50
    n_report_steps = 1

    reporter = LossReporter("NLL")

    # use DTO in the training process
    # flow._use_checkpoints = True
    #
    # #Anode options
    # options = {
    #     "Nt": 20,
    #     "method": "RK4"
    # }
    # flow._kwargs = options
    if is_ode:
       dynamics._brute_force = False

    for epoch in range(n_epochs):
        for it, idxs in enumerate(batch_iter):
            batch = data_smaller[idxs].to(device)
            batch = batch

            optim.zero_grad()
            if is_ode:
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
    reporter.plot()
    reporter.show() # TO DO: change by savefig

    # use OTD in the evaluation process
    flow._use_checkpoints = False

    if is_ode:
       dynamics._brute_force = True

    #victor: here I encapsulated the code inside "evaluate_nll" to avoid copypasting as in the original script
    def evaluate_nll(n_batch):
        # use OTD in the evaluation process
        flow._use_checkpoints = False

        data_nll = 0.
        batch_iter = BatchIterator(len(data_smaller), n_batch)
        for it, batch_idxs in enumerate(batch_iter):
            if it > 100:
                break
            x = torch.Tensor(data_smaller[batch_idxs]).to(device)
            z, delta_logp = flow(x, inverse=True)
            nll = (prior.energy(z).view(-1) + delta_logp.view(-1)).mean()
            print("\r{}".format(it), nll, end="")
            data_nll += nll
        data_nll = data_nll / (it + 1)
        return data_nll

    data_nll = evaluate_nll(n_batch)
    holdout_nll = evaluate_nll(n_batch)

    print("\nnll %.4f \t holdout nll %.4f " % (data_nll.item(), holdout_nll.item()))


def train_generating_flows():
    n_batch = 64
    batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

    optim = torch.optim.Adam(flow.parameters(), lr=5e-3)

    n_epochs = 2  # 50
    n_report_steps = 1

    lambdas = torch.linspace(1., 0.5, n_epochs).to(device)

    reporter = LossReporter("NLL", "KLL")

    # use DTO in the training process
    flow._use_checkpoints = True

    # Anode options
    options = {
        "Nt": 20,
        "method": "RK4"
    }
    flow._kwargs = options

    for epoch, lamb in enumerate(lambdas):
        for it, idxs in enumerate(batch_iter):
            batch = data_smaller[idxs].to(device)
            batch = batch

            optim.zero_grad()
            z, delta_logp = flow(batch, inverse=True)
            nll = (prior.energy(z).view(-1) + delta_logp.view(-1)).mean()

            (lamb * nll).backward()

            # kl divergence to the target
            latent = prior.sample(n_batch)
            x, d_logp = flow(latent, inverse=False)
            kll = (target.energy(x) + d_logp.view(-1)).mean()

            # aggregate weighted gradient
            ((1. - lamb) * kll).backward()

            reporter.report(nll, kll)

            optim.step()

            if it % n_report_steps == 0:
                print("\repoch: {0}, iter: {1}/{2}, lambda: {3}, NLL: {4:.4}, KLL: {5:.4}".format(
                    epoch,
                    it,
                    len(batch_iter),
                    lamb,
                    *reporter.recent(1).ravel()
                ), end="")

    #plot curves
    reporter.plot()
    reporter.show()  # change by savefig

    plot_generating_flow()


def plot_generating_flow():
    # use OTD in the evaluation process
    flow._use_checkpoints = False

    latent = prior.sample(10000)
    #latent = prior.sample(500)
    x, dlogp = flow(latent, brute_force=True)

    energies_data = target.energy(data).numpy()
    energies_bg = target.energy(x).cpu().detach().view(-1).numpy()
    energies_prior = target.energy(latent).cpu().detach().numpy()
    min_energy = min(energies_data.min(), energies_bg.min())

    log_w = target.energy(x).view(-1) - prior.energy(latent).view(-1) + dlogp.view(-1)
    log_w = log_w.view(-1).cpu().detach()

    plt.hist(log_w, bins=100, density=True, )

    plt.figure(figsize=(13, 8))
    efac = 1

    plt.hist(energies_bg, bins=100, density=True, range=(min_energy, 0), alpha=0.4, histtype='step', linewidth=4,
             color="r", label="samples");

    plt.hist(energies_data, bins=100, density=True, range=(min_energy, 0), alpha=0.4, color="g", histtype='step',
             linewidth=4,
             label="test data");
    plt.hist(energies_bg, bins=100, density=True, range=(min_energy, 0), alpha=0.4, histtype='step', linewidth=4,
             color="b", label="weighted samples", weights=np.exp(-log_w));

    plt.xlabel("u(x)", fontsize=45)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45);
    plt.legend(fontsize=25);
    plt.show()  # change by savefig

def main():
    training_nll()
    train_generating_flows()


if __name__ == "__main__":
    main()