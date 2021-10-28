import torch
from egnn.models import EGNN_dynamics
from deprecated.eqnode.kernels import compute_gammas
from deprecated.eqnode.flows2 import HutchinsonEstimator, RegularizedHutchinsonEstimator
from deprecated.eqnode.bg import DiffEqFlow, RegularizedDiffEqFlow
from deprecated.eqnode.flows import ContinuousNormalizingFlow
from deprecated.eqnode.densenet import DenseNet
from deprecated.eqnode.dynamics import SchNet, SimpleEqDynamics
from flows.ffjord import FFJORD
from deprecated.eqnode.kernels import RbfEncoder
from deprecated.eqnode.dynamics import KernelDynamics


def get_model(args, dim, n_particles):
    n_rbfs = 50
    d_max = 16
    n_features = 16
    n_rbfs = 50

    kernel_mus = torch.linspace(0., 8., n_rbfs)
    kernel_gammas = torch.ones(n_rbfs) * 0.5

    if torch.cuda.is_available():
        kernel_mus = kernel_mus.cuda()
        kernel_gammas = kernel_gammas.cuda()

    rbf_encoder = RbfEncoder(kernel_mus, kernel_gammas.log(), trainable=False)

    if args.model == 'schnet':
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

    elif args.model == 'simple_dynamics':
        net_dynamics = SimpleEqDynamics(
            transformation=DenseNet([n_rbfs, 64, 32, 1],
                                    activation=torch.nn.Tanh()),
            rbf_encoder=rbf_encoder,
            n_particles=n_particles,
            n_dimesnion=dim // n_particles,
            n_rbfs=n_rbfs,
        )
        flow = FFJORD(net_dynamics, trace_method='hutch', hutch_noise=args.hutch_noise)

    # elif args.model == 'our_dynamics':
    #     net_dynamics = OurDynamics(
    #         n_particles=n_particles,
    #         n_dimension=dim // n_particles,
    #         se3_layers=args.se3_layers,
    #         se3_channels=args.se3_channels,
    #         num_degrees=args.se3_num_degrees
    #     )
    #     if torch.cuda.is_available():
    #         net_dynamics = net_dynamics.cuda()
    #
    #     if args.ode_regularization > 0:
    #         dynamics = RegularizedHutchinsonEstimator(net_dynamics,
    #                                                   brute_force=args.brute_force)
    #         flow = RegularizedDiffEqFlow(dynamics)
    #     else:
    #         dynamics = HutchinsonEstimator(net_dynamics, brute_force=args.brute_force)
    #         flow = DiffEqFlow(dynamics=dynamics)

    elif args.model == 'egnn_dynamics' or args.model == 'gnn_dynamics':
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        net_dynamics = EGNN_dynamics(n_particles=n_particles, device=device,  n_dimension=dim // n_particles, hidden_nf=args.nf,
            act_fn=torch.nn.SiLU(), n_layers=args.n_layers, recurrent=True, tanh=args.tanh, attention=args.attention, condition_time=args.condition_time, mode=args.model, agg=args.x_aggregation)

        flow = FFJORD(net_dynamics, trace_method='hutch', hutch_noise=args.hutch_noise, ode_regularization=args.ode_regularization)

    # elif args.model == 'egnn_dynamics_dirty':
    #     device = 'cpu'
    #     if torch.cuda.is_available():
    #         device = 'cuda'
    #     net_dynamics = EGNN_dynamics(n_particles=n_particles, device=device,  n_dimension=dim // n_particles, hidden_nf=64,
    #         act_fn=torch.nn.SiLU(), n_layers=3, recurrent=True)
    #
    #     if args.ode_regularization > 0:
    #         dynamics = RegularizedHutchinsonEstimator(net_dynamics,
    #                                                   brute_force=args.brute_force)
    #         flow = RegularizedDiffEqFlow(dynamics)
    #     else:
    #         dynamics = HutchinsonEstimator(net_dynamics, brute_force=args.brute_force)
    #         flow = DiffEqFlow(dynamics=dynamics)

    elif args.model == "kernel_dynamics":
        n_dimension = dim // n_particles
        d_max = 8
        n_rbfs = 50
        mus = torch.linspace(0, d_max, n_rbfs)
        mus.sort()
        gammas = 0.5 * torch.ones(n_rbfs)
        mus_time = torch.linspace(0, 1, 10)
        gammas_time = 0.3 * torch.ones(10)
        dynamics = KernelDynamics(n_particles, n_dimension, mus, gammas,
                                  optimize_d_gammas=True,
                                  optimize_t_gammas=True,
                                  mus_time=mus_time,
                                  gammas_time=gammas_time)
    # elif args.model == 'our_dynamics_reimplementation':
    #     net_dynamics = OurDynamics(
    #         n_particles=n_particles,
    #         n_dimension=dim // n_particles,
    #         se3_layers=args.se3_layers,
    #         se3_channels=args.se3_channels
    #     )
    #     if torch.cuda.is_available():
    #         net_dynamics = net_dynamics.cuda()

        #
        # if args.ode_regularization > 0:
        #     dynamics = RegularizedHutchinsonEstimator(net_dynamics,
        #                                               brute_force=args.brute_force)
        #     flow = RegularizedDiffEqFlow(dynamics)
        # else:
        #     dynamics = HutchinsonEstimator(net_dynamics, brute_force=args.brute_force)
        flow = DiffEqFlow(dynamics=dynamics)
        #flow = FFJORD(dynamics, trace_method='hutch', hutch_noise=args.hutch_noise)

    elif args.model == "kernel_dynamics_lj13":
        n_dimension = dim // n_particles
        d_max = 16
        mus = torch.linspace(0, d_max, 50)
        gain = 0.2 + 10 * ((mus / d_max - 1.5 / d_max) / 2).pow(2)
        mus = 2 * ((mus / d_max - 1.5 / d_max) / 2).pow(2) * d_max
        mus = mus.sort()[0]
        gammas = (0.3 + 3 * ((mus / d_max - 1.5 / d_max) / 2).pow(2) * d_max)

        mus_time = torch.linspace(0, 1, 10)
        gain = 0.3 + 2 * ((mus_time - 0.5) / 2).pow(2)
        gammas_time = compute_gammas(mus_time, gain=gain)
        kdyn = KernelDynamics(n_particles, n_dimension, mus, gammas, optimize_d_gammas=True, optimize_t_gammas=True,
                              mus_time=mus_time, gammas_time=gammas_time)

        flow = ContinuousNormalizingFlow(kdyn, integrator="dopri5", n_time_steps=2, step_size=1. / 100)

    else:
        raise ValueError

    return flow
