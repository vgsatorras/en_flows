import torch
from torch import nn
from flows.utils import \
    standard_gaussian_log_likelihood_with_mask, \
    sample_gaussian_with_mask, sum_except_batch, \
    assert_correctly_masked
import torch.nn.functional as F
from egnn.models import EGNN


class EGNN_output_h(nn.Module):
    def __init__(self, in_node_nf, out_node_nf, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, recurrent=True,
                 attention=False, agg='sum'):
        super().__init__()
        self.egnn = EGNN(in_node_nf=in_node_nf, in_edge_nf=0,
                         hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                         n_layers=n_layers, recurrent=recurrent,
                         attention=attention,
                         out_node_nf=out_node_nf, agg=agg)

        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.device = device
        # self.n_dims = None
        self._edges_dict = {}

    def forward(self, x, h, node_mask, edge_mask):
        bs, n_nodes, dims = x.shape
        assert self.in_node_nf == h.size(2)

        node_mask = node_mask.view(bs * n_nodes, 1)
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        x = x.view(bs*n_nodes, dims) * node_mask

        h = h.view(bs*n_nodes, self.in_node_nf) * node_mask

        h_final, x_final = self.egnn(
            h, x, edges, node_mask=node_mask, edge_mask=edge_mask)

        h_final = h_final.view(bs, n_nodes, self.out_node_nf)

        return h_final

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

        edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
        return edges


class UniformDequantizer(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(UniformDequantizer, self).__init__()

    def forward(self, tensor, node_mask, edge_mask, context):
        category, integer = tensor['categorical'], tensor['integer']
        zeros = torch.zeros(integer.size(0), device=integer.device)

        out_category = category + torch.rand_like(category) - 0.5
        out_integer = integer + torch.rand_like(integer) - 0.5

        if node_mask is not None:
            out_category = out_category * node_mask
            out_integer = out_integer * node_mask

        out = {'categorical': out_category, 'integer': out_integer}
        return out, zeros

    def reverse(self, tensor):
        categorical, integer = tensor['categorical'], tensor['integer']
        categorical, integer = torch.round(categorical), torch.round(integer)
        return {'categorical': categorical, 'integer': integer}


def sigmoid(x, node_mask):
    z = torch.sigmoid(x)
    ldj = sum_except_batch(node_mask * (F.logsigmoid(x) + F.logsigmoid(-x)))
    return z, ldj


def affine(x, translation, log_scale):
    z = translation + x * log_scale.exp()
    ldj = sum_except_batch(log_scale)
    return z, ldj


def transform_to_hypercube_partition(integer, interval_noise):
    assert interval_noise.min().item() >= 0., interval_noise.max().item() <= 1.
    return integer + interval_noise


def transform_to_argmax_partition(onehot, u, node_mask):
    assert torch.allclose(
        onehot.sum(-1, keepdims=True) * node_mask,
        torch.ones_like(onehot[..., 0:1]) * node_mask)

    T = (onehot * u).sum(-1, keepdim=True)
    z = onehot * u + node_mask * (1 - onehot) * (T - F.softplus(T - u))
    ldj = (1 - onehot) * F.logsigmoid(T - u) * node_mask

    assert_correctly_masked(z, node_mask)
    assert_correctly_masked(ldj, node_mask)

    ldj = sum_except_batch(ldj)

    return z, ldj


class VariationalDequantizer(nn.Module):
    def __init__(self, node_nf, device, agg='sum'):
        super().__init__()
        self.net_fn = EGNN_output_h(
            in_node_nf=node_nf, out_node_nf=node_nf*2, device=device, agg=agg
        )

    def sample_qu_xh(self, node_mask, edge_mask, x, h):
        net_out = self.net_fn(x, h, node_mask, edge_mask)
        mu, log_sigma = torch.chunk(net_out, chunks=2, dim=-1)

        eps = sample_gaussian_with_mask(mu.size(), mu.device, node_mask)
        log_q_eps = standard_gaussian_log_likelihood_with_mask(eps, node_mask)

        assert (mu * (1 - node_mask)).sum() < 1e-5 and \
               (log_sigma * (1 - node_mask)).sum() < 1e-5, \
               'These parameters should be masked.'
        u, ldj = affine(eps, mu, log_sigma)
        log_qu = log_q_eps - ldj

        return u, log_qu

    def transform_to_partition_v(self, h_category, h_integer, u_category, u_integer, node_mask):
        u_category, ldj_category = sigmoid(u_category, node_mask)
        u_integer, ldj_integer = sigmoid(u_integer, node_mask)
        ldj = ldj_category + ldj_integer

        v_category = transform_to_hypercube_partition(h_category, u_category)
        v_integer = transform_to_hypercube_partition(h_integer, u_integer)
        return v_category, v_integer, ldj

    def forward(self, tensor, node_mask, edge_mask, x):
        categorical, integer = tensor['categorical'], tensor['integer']

        h = torch.cat([categorical, integer], dim=2)

        n_categorical, n_integer = categorical.size(2), integer.size(2)

        u, log_qu_xh = self.sample_qu_xh(node_mask, edge_mask, x, h)

        u_categorical = u[:, :, :n_categorical]
        u_integer = u[:, :, n_categorical:]

        v_categorical, v_integer, ldj = self.transform_to_partition_v(
            categorical, integer, u_categorical, u_integer, node_mask)
        log_qv_xh = log_qu_xh - ldj

        if node_mask is not None:
            v_categorical = v_categorical * node_mask
            v_integer = v_integer * node_mask

        v = {'categorical': v_categorical, 'integer': v_integer}
        return v, log_qv_xh

    def reverse(self, tensor):
        categorical, integer = tensor['categorical'], tensor['integer']
        categorical, integer = torch.floor(categorical), torch.floor(integer)
        return {'categorical': categorical, 'integer': integer}


class ArgmaxAndVariationalDequantizer(VariationalDequantizer):
    def __init__(self, node_nf, device, agg='sum'):
        super().__init__(node_nf, device, agg)

    def transform_to_partition_v(self, h_category, h_integer, u_category, u_integer, node_mask):
        u_integer, ldj_integer = sigmoid(u_integer, node_mask)
        v_category, ldj_category = transform_to_argmax_partition(h_category,
                                                                 u_category,
                                                                 node_mask)
        ldj = ldj_category + ldj_integer
        v_integer = h_integer + u_integer
        return v_category, v_integer, ldj

    def reverse(self, tensor):
        categorical, integer = tensor['categorical'], tensor['integer']
        K = categorical.size(2)
        integer = torch.floor(integer)

        categorical = F.one_hot(torch.argmax(categorical, dim=-1), K)
        return {'categorical': categorical, 'integer': integer}
