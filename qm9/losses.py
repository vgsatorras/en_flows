import torch


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8


# class UnnormalizedPrior(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self._dim = dim
#         # self._n_particles = n_particles
#         # self._spacial_dims = dim // n_particles
#         print("Deprecated, do not use!")
#
#     def forward(self, z_x, z_h, node_mask=None):
#         assert len(z_x.size()) == 3
#         z_x = self._remove_mean(z_x)
#         z = torch.cat([z_x, z_h], dim=2)
#         if node_mask is not None:
#             z = z * node_mask
#         log_pz = sum_except_batch(-0.5 * z.pow(2))
#         return log_pz

    # def sample(self, n_samples, temperature=1.):
    #     x = torch.Tensor(n_samples, self._n_particles,
    #                      self._spacial_dims).normal_()
    #     return self._remove_mean(x)
    #
    # def _remove_mean(self, x):
    #     x = x - torch.mean(x, dim=1, keepdim=True)
    #     return x


def compute_loss_and_nll(args, dequantizer, flow, prior, nodes_dist, x, h, node_mask, edge_mask, context):
    # if args.ode_regularization > 0:
    #     z, delta_logp, reg_frob, reg_dx2 = flow(batch)
    #     z = z.view(z.size(0), 8)
    #     nll = (prior(z).view(-1) + delta_logp.view(-1)).mean()
    #     reg_term = (reg_frob.mean() + reg_dx2.mean())
    #     loss = nll + args.ode_regularization * reg_term
    # else:
    bs, n_nodes, n_dims = x.size()

    if args.dataset == 'qm9':
        h, log_qh_x = dequantizer(h, node_mask, edge_mask, x)

        h = torch.cat([h['categorical'], h['integer']], dim=2)
        # TO DO: Change dequantizer
    elif args.dataset == 'qm9_positional':
        h = torch.ones(bs, n_nodes, 0).to(x.device)
        log_qh_x = 0
    else:
        raise ValueError

    xh = torch.cat([x, h], dim=2)

    assert_correctly_masked(xh, node_mask)

    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    z, delta_logp, reg_term = flow(xh, node_mask, edge_mask, context)

    z_x, z_h = z[:, :, 0:n_dims].clone(), z[:, :, n_dims:].clone()

    assert_correctly_masked(z_x, node_mask)
    assert_correctly_masked(z_h, node_mask)

    N = node_mask.squeeze(2).sum(1).long()

    log_pN = nodes_dist.log_prob(N)

    log_pz = prior(z_x, z_h, node_mask)
    assert log_pz.size() == delta_logp.size()
    log_px = (log_pz + delta_logp - log_qh_x + log_pN).mean()  # Average over batch.
    reg_term = reg_term.mean()  # Average over batch.
    nll = -log_px

    mean_abs_z = torch.mean(torch.abs(z)).item()

    return nll, reg_term, mean_abs_z
