

import torch
import numpy as np

from eqnode.train_utils import IndexBatchIterator, LossReporter, \
    BatchIterator
from eqnode.flows2 import HutchinsonEstimator, RegularizedHutchinsonEstimator

from eqnode.test_systems import MultiDoubleWellPotential
from eqnode.particle_utils import remove_mean
from eqnode.bg import DiffEqFlow, RegularizedDiffEqFlow
from eqnode.densenet import DenseNet
from eqnode.dynamics import SchNet, SimpleEqDynamics
from se3_dynamics.dynamics import OurDynamics
from eqnode.kernels import RbfEncoder
from eqnode.priors import MeanFreeNormalPrior
import argparse
# %%

# this try statement allows for automatic switching into debug mode in case
# of an error
def main():

    parser = argparse.ArgumentParser(description='SE3')
    parser.add_argument('--model', type=str, default='our_dynamics', help='our_dynamics | schnet | simple_dynamics | se3_coupling')
    parser.add_argument('--se3_layers', type=int, default=3)
    parser.add_argument('--se3_channels', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_data', type=int, default=1000, help="Number of training samples")
    parser.add_argument('--brute_force', type=bool, default=False,
                        help='True | False')
    parser.add_argument('--ode_regularization', type=float, default=1e-2)
    args = parser.parse_args()
    print(args)

    # first define system dimensionality and a target energy/distribution

    dim = 8
    n_particles = 4

    # DW parameters
    a = 0.9
    b = -4
    c = 0
    offset = 4

    # variable 'target' and function 'MultiDoubleWellPotential' not used in this experiment
    target = MultiDoubleWellPotential(dim, n_particles, a, b, c, offset)

    # shape data: [1000000, 8], idx is ignored
    data, idx = np.load("MCMC_data/dw4-dataidx.npy", allow_pickle=True)

    # used once to reshuffle for cutting the data (he just uses 1000 samples)
    idx = np.random.choice(len(data), len(data), replace=False)

    data = data.reshape(-1, dim)
    data = remove_mean(data, n_particles, dim // n_particles)

    data_smaller = data[idx[:args.n_data]].clone()


    # now set up a prior
    prior = MeanFreeNormalPrior(dim, n_particles)

    n_rbfs = 50
    d_max = 16
    n_features = 16

    kernel_mus = torch.linspace(0., 8., n_rbfs)
    kernel_gammas = torch.ones(n_rbfs) * 0.5

    if torch.cuda.is_available():
        kernel_mus = kernel_mus.cuda()
        kernel_gammas = kernel_gammas.cuda()

    rbf_encoder = RbfEncoder(kernel_mus, kernel_gammas.log(), trainable=False)

    model_name = args.model # schnet | simple_dynamics | our_dynamics

    if model_name == 'schnet':
        is_ode = True
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
        )

        if torch.cuda.is_available():
            schnet = schnet.cuda()
        net_dynamics = schnet

        if args.ode_regularization > 0:
            dynamics = RegularizedHutchinsonEstimator(net_dynamics,
                                                      brute_force=args.brute_force)
            flow = RegularizedDiffEqFlow(dynamics)
        else:
            dynamics = HutchinsonEstimator(net_dynamics, brute_force=args.brute_force)
            flow = DiffEqFlow(dynamics=dynamics)

    elif model_name == 'simple_dynamics':
        is_ode = True
        net_dynamics = SimpleEqDynamics(
            transformation=DenseNet([n_rbfs, 64, 32, 1], activation=torch.nn.Tanh()),
            rbf_encoder=rbf_encoder,
            n_particles=n_particles,
            n_dimesnion=dim // n_particles,
            n_rbfs = n_rbfs,
        )
        if torch.cuda.is_available():
            net_dynamics = net_dynamics.cuda()

        if args.ode_regularization > 0:
            dynamics = RegularizedHutchinsonEstimator(net_dynamics,
                                                      brute_force=args.brute_force)
            flow = RegularizedDiffEqFlow(dynamics)
        else:
            dynamics = HutchinsonEstimator(net_dynamics, brute_force=args.brute_force)
            flow = DiffEqFlow(dynamics=dynamics)

    elif model_name == 'our_dynamics':
        is_ode = True
        net_dynamics = OurDynamics(
            n_particles=n_particles,
            n_dimension=dim // n_particles,
            se3_layers=args.se3_layers,
            se3_channels=args.se3_channels
        )
        if torch.cuda.is_available():
            net_dynamics = net_dynamics.cuda()

        if args.ode_regularization > 0:
            dynamics = RegularizedHutchinsonEstimator(net_dynamics,
                                                      brute_force=args.brute_force)
            flow = RegularizedDiffEqFlow(dynamics)
        else:
            dynamics = HutchinsonEstimator(net_dynamics, brute_force=args.brute_force)
            flow = DiffEqFlow(dynamics=dynamics)
    elif model_name == 'our_dynamics_reimplementation':
        is_ode = True
        net_dynamics = OurDynamics(
            n_particles=n_particles,
            n_dimension=dim // n_particles,
            se3_layers=args.se3_layers,
            se3_channels=args.se3_channels
        )
        if torch.cuda.is_available():
            net_dynamics = net_dynamics.cuda()

        #
        # if args.ode_regularization > 0:
        #     dynamics = RegularizedHutchinsonEstimator(net_dynamics,
        #                                               brute_force=args.brute_force)
        #     flow = RegularizedDiffEqFlow(dynamics)
        # else:
        #     dynamics = HutchinsonEstimator(net_dynamics, brute_force=args.brute_force)
        #     flow = DiffEqFlow(dynamics=dynamics)

    else:
        raise ValueError


    if torch.cuda.is_available():
        flow = flow.cuda()


    # initial training with likelihood maximization on data set


    n_batch = 64
    batch_iter = IndexBatchIterator(len(data_smaller), n_batch)

    optim = torch.optim.AdamW(flow.parameters(), lr=5e-3, amsgrad=True,
                              weight_decay=0.01)

    n_report_steps = 1

    reporter = LossReporter("NLL")

    # %%
    if is_ode:
       dynamics._brute_force = False

    # %%


    for epoch in range(args.n_epochs):
        for it, idxs in enumerate(batch_iter):
            batch = data_smaller[idxs]

            if not is_ode:
                # The coupling flow already expects data in shape B, N, D=2/3
                batch = batch.view(batch.size(0), 4, 2)

            if torch.cuda.is_available():
                batch = batch.cuda()

            optim.zero_grad()
            if is_ode:
                dynamics.reset_noise()
            # transform batch through flow
            if args.ode_regularization > 0:
                z, delta_logp, reg_frob, reg_dx2 = flow(batch, inverse=True)
                z = z.view(z.size(0), 8)
                nll = (prior.energy(z).view(-1) + delta_logp.view(-1)).mean()
                reg_term = (reg_frob.mean() + reg_dx2.mean())
                loss = nll + args.ode_regularization * reg_term
            else:
                if is_ode:
                    z, delta_logp = flow(batch, inverse=True)
                    Ez = prior.energy(z)
                    nll = (Ez.view(-1) + delta_logp.view(-1)).mean()
                else:
                    z, delta_logp = flow(batch)
                    z = z.view(z.size(0), 8)
                    Ez = prior.energy(z)
                    nll = (Ez.view(-1) - delta_logp.view(-1)).mean()

                print(torch.mean(torch.abs(z)).item())

                reg_term = torch.tensor([0.])
                loss = nll
            # standard nll from forward KL

            loss.backward()

            reporter.report(nll)

            optim.step()

            if it % n_report_steps == 0:
                print("\repoch: {0}, iter: {1}/{2}, NLL: {3:.4} Reg term: {4:.3f}".format(
                    epoch,
                    it,
                    len(batch_iter),
                    *reporter.recent(1).ravel(),
                    reg_term.item()
                ), end="\n")
        print()  # Clear line

    reporter.plot()

    # use OTD in the evaluation process
    flow._use_checkpoints = False

    #flow.odefunc.method = 'exact'
    if is_ode:
       dynamics._brute_force = True

    data_nll = 0.
    batch_iter = BatchIterator(len(data_smaller), n_batch)
    for it, batch_idxs in enumerate(batch_iter):
        if it > 100:
            break
        x = torch.Tensor(data_smaller[batch_idxs])
        if torch.cuda.is_available():
            x = x.cuda()
        z, delta_logp, _, _ = flow(x, inverse=True)
        nll = (prior.energy(z).view(-1) + delta_logp.view(-1)).mean()
        print("\r{}".format(it), nll, end="")
        data_nll += nll
    data_nll = data_nll / (it + 1)

    print()
    print(f'Final nll {data_nll}')

    # TODO: no evaluation on hold out data yet
    holdout_nll = 0.


main()

# try:
#     main()
# except:
# #except:
#   import pdb
#   pdb.post_mortem()