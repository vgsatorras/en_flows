import torch

def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(dim=-1)


# class UnnormalizedPrior(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self._dim = dim
#         # self._n_particles = n_particles
#         # self._spacial_dims = dim // n_particles
#
#     def forward(self, x):
#         assert len(x.size()) == 3
#         x = self._remove_mean(x)
#         log_px = sum_except_batch(-0.5 * x.pow(2).sum(dim=-1, keepdim=True))
#         return log_px
#
#     # def sample(self, n_samples, temperature=1.):
#     #     x = torch.Tensor(n_samples, self._n_particles,
#     #                      self._spacial_dims).normal_()
#     #     return self._remove_mean(x)
#
#     def _remove_mean(self, x):
#         x = x - torch.mean(x, dim=1, keepdim=True)
#         return x


def compute_loss_and_nll(args, flow, prior, batch):
    # if args.ode_regularization > 0:
    #     z, delta_logp, reg_frob, reg_dx2 = flow(batch)
    #     z = z.view(z.size(0), 8)
    #     nll = (prior(z).view(-1) + delta_logp.view(-1)).mean()
    #     reg_term = (reg_frob.mean() + reg_dx2.mean())
    #     loss = nll + args.ode_regularization * reg_term
    # else:
    z, delta_logp, reg_term = flow(batch)

    log_pz = prior(z)
    log_px = (log_pz + delta_logp.view(-1)).mean()
    nll = -log_px

    mean_abs_z = torch.mean(torch.abs(z)).item()
    # print(f"mean(abs(z)): {mean_abs_z:.2f}")

    #reg_term = torch.tensor([0.])
    reg_term = reg_term.mean()  # Average over batch.
    loss = nll

    return loss, nll, reg_term, mean_abs_z



def compute_loss_and_nll_kerneldynamics(args, flow, prior, batch, n_particles, n_dims):
    bs = batch.size(0)
    z, delta_logp = flow(batch.view(bs, -1))
    z = z.view(bs, n_particles, n_dims)
    nll = -(prior(z).view(-1) - delta_logp.view(-1)).mean()
    loss = nll
    reg_term, mean_abs_z = torch.tensor([0.]), 0
    return loss, nll, reg_term.to(z.device), mean_abs_z
