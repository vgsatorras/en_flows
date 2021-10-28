import torch


def masked_mean(x, node_mask, dim, keepdim=False):
    return x.sum(dim=dim, keepdim=keepdim) / node_mask.sum(dim=dim, keepdim=keepdim)


def masked_stdev(x, node_mask, dim, keepdim=False):
    mean = masked_mean(x, node_mask, dim, keepdim=True)

    diff = (x - mean) * node_mask
    diff_2 = diff.pow(2).sum(dim=dim, keepdim=keepdim)

    diff_div_N = diff_2 / node_mask.sum(dim=dim, keepdim=keepdim)
    return torch.sqrt(diff_div_N + 1e-5)


class ActNormPositionAndFeatures(torch.nn.Module):
    def __init__(self, in_node_nf, n_dims):
        super().__init__()
        self.in_node_nf = in_node_nf
        self.n_dims = n_dims

        self.x_log_s = torch.nn.Parameter(torch.zeros(1, 1))

        self.h_t = torch.nn.Parameter(torch.zeros(1, in_node_nf))
        self.h_log_s = torch.nn.Parameter(torch.zeros(1, in_node_nf))
        self.register_buffer('initialized', torch.tensor(0))

    def initialize(self, x, h, node_mask):
        print('initializing')
        with torch.no_grad():
            h_mean = masked_mean(h, node_mask, dim=0, keepdim=True)
            h_stdev = masked_stdev(h, node_mask, dim=0, keepdim=True)
            h_log_stdev = torch.log(h_stdev + 1e-8)

            self.h_t.data.copy_(h_mean.detach())
            self.h_log_s.data.copy_(h_log_stdev.detach())

            x_stdev = masked_stdev(x, node_mask, dim=(0, 1), keepdim=True)
            x_log_stdev = torch.log(x_stdev + 1e-8)

            self.x_log_s.data.copy_(x_log_stdev.detach())

            self.initialized.fill_(1)

    def forward(self, xh, node_mask, edge_mask, context=None, reverse=False):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        # edges = self.get_adj_matrix(n_nodes, bs, self.device)
        node_mask = node_mask.view(bs * n_nodes, 1)
        xh = xh.view(bs * n_nodes, -1).clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1).to(xh.device)
        else:
            h = xh[:, self.n_dims:].clone()

        # TODO ENABLE INIT.
        if not self.initialized:
            self.initialize(x, h, node_mask)

        h_log_s = self.h_log_s.expand_as(h)
        h_t = self.h_t.expand_as(h)
        x_log_s = self.x_log_s.expand_as(x)

        h_d_ldj = -(h_log_s * node_mask).sum(1)
        x_d_ldj = -(x_log_s * node_mask).sum(1)
        d_ldj = h_d_ldj + x_d_ldj
        d_ldj = d_ldj.view(bs, n_nodes).sum(1)

        if not reverse:
            h = (h - h_t) / torch.exp(h_log_s) * node_mask
            x = x / torch.exp(x_log_s) * node_mask

        else:
            h = (h * torch.exp(h_log_s) + h_t) * node_mask
            x = x * torch.exp(x_log_s) * node_mask

        x = x.view(bs, n_nodes, self.n_dims)
        h = h.view(bs, n_nodes, h_dims)
        xh = torch.cat([x, h], dim=2)

        if not reverse:
            return xh, d_ldj, 0
        else:
            return xh

    def reverse(self, xh, node_mask, edge_mask, context=None):
        assert self.initialized
        return self(xh, node_mask, edge_mask, context, reverse=True)
