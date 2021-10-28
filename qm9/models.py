import torch
from torch.distributions.categorical import Categorical
from qm9 import analyze
import numpy as np
from egnn.models import EGNN_dynamics_QM9
from flows.ffjord import FFJORD
from flows.dequantize import UniformDequantizer, \
    VariationalDequantizer, ArgmaxAndVariationalDequantizer
from flows import Flow
from flows.actnorm import ActNormPositionAndFeatures
from flows.distributions import PositionFeaturePrior


def get_model(args, device):
    in_node_nf = 6
    if args.dataset == 'qm9_positional':
        raise NotImplementedError
        # in_node_nf = 1

    prior = PositionFeaturePrior(n_dim=3, in_node_nf=in_node_nf)  # set up prior

    if args.dequantization == 'uniform':
        dequantizer = UniformDequantizer()
    elif args.dequantization == 'variational':
        dequantizer = VariationalDequantizer(in_node_nf, device)
    elif args.dequantization == 'argmax_variational':
        dequantizer = ArgmaxAndVariationalDequantizer(in_node_nf, device)
    else:
        raise ValueError()

    if args.condition_time:
        dynamics_in_node_nf = in_node_nf + 1
    else:
        print('Warning: dynamics model is _not_ conditioned on time.')
        dynamics_in_node_nf = in_node_nf

    net_dynamics = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=args.context_node_nf,
        n_dims=3, device=device, hidden_nf=args.nf,
        act_fn=torch.nn.SiLU(), n_layers=args.n_layers, recurrent=True,
        attention=args.attention, tanh=args.tanh, mode=args.model)

    flow_transforms = []

    if args.actnorm:
        actnorm = ActNormPositionAndFeatures(in_node_nf, n_dims=3)
        flow_transforms.append(actnorm)

    ffjord = FFJORD(net_dynamics, trace_method='hutch',
                    ode_regularization=args.ode_regularization)
    flow_transforms.append(ffjord)

    flow = Flow(transformations=flow_transforms)

    flow.set_trace(args.trace)

    nodes_dist = DistributionNodes()

    return prior, flow, dequantizer, nodes_dist


def get_optim(args, flow, dequantizer):
    optim = torch.optim.AdamW(
        list(dequantizer.parameters()) + list(flow.parameters()),
        lr=args.lr, amsgrad=True,
        weight_decay=1e-12)

    return optim


class DistributionNodes:
    def __init__(self, histogram=analyze.analyzed['n_nodes']):

        self.n_nodes = []
        prob = []
        self.keys = {}
        for i, nodes in enumerate(histogram):
            self.n_nodes.append(nodes)
            self.keys[nodes] = i
            prob.append(histogram[nodes])

        prob = np.array(prob)
        prob = prob/np.sum(prob)

        self.prob = torch.from_numpy(prob).float()

        entropy = torch.sum(self.prob * torch.log(self.prob + 1e-30))
        print("Entropy of n_nodes: H[N]", entropy.item())

        self.m = Categorical(torch.tensor(prob))

    def sample(self):
        idx = self.m.sample()
        return self.n_nodes[idx]

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1

        idcs = [self.keys[i.item()] for i in batch_n_nodes]
        idcs = torch.tensor(idcs).to(batch_n_nodes.device)

        log_p = torch.log(self.prob + 1e-30)

        log_p = log_p.to(batch_n_nodes.device)

        log_probs = log_p[idcs]

        return log_probs



if __name__ == '__main__':
    dist_nodes = DistributionNodes()
    print(dist_nodes.n_nodes)
    print(dist_nodes.prob)
    for i in range(10):
        print(dist_nodes.sample())
