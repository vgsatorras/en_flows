import torch


class Flow(torch.nn.Module):
    def __init__(self, transformations):
        super(Flow, self).__init__()
        self.transformations = torch.nn.ModuleList(transformations)

    def set_trace(self, trace):
        did_set_trace = False
        for module in self.modules():
            if hasattr(module, 'set_trace') and module is not self:
                module.set_trace(trace)
                did_set_trace = True
        assert did_set_trace

    def forward(self, x, node_mask=None, edge_mask=None, context=None):
        ldj = x.new_zeros(x.shape[0])
        reg_term = x.new_zeros(x.shape[0])

        for transform in self.transformations:
            x, delta_logp, delta_reg_term = transform(x, node_mask, edge_mask, context)

            ldj = ldj + delta_logp
            reg_term = reg_term + delta_reg_term

        return x, ldj, reg_term

    def reverse(self, z, node_mask=None, edge_mask=None, context=None):
        for transform in reversed(self.transformations):
            z = transform.reverse(z, node_mask, edge_mask, context)

        return z

    def reverse_chain(self, z, node_mask=None, edge_mask=None, context=None):
        for transform in reversed(self.transformations):
            if hasattr(transform, 'reverse_chain'):
                z_chain = transform.reverse_chain(z, node_mask, edge_mask, context)
                n_timesteps = z_chain.size(0)
                node_mask = node_mask.view(1, -1, 1)
                node_mask = node_mask.repeat(n_timesteps, 1, 1)

                node_mask = node_mask.view(-1, 1)
                z = z_chain.view(n_timesteps * z.size(0), *z.size()[1:])

            else:
                z = transform.reverse(z, node_mask, edge_mask, context)

        return z




class ConditionalFlow(torch.nn.Module):
    def __init__(self, transformations, context_net=None):
        super(ConditionalFlow, self).__init__()
        self.transformations = torch.nn.ModuleList(transformations)
        self.context_net = context_net

    def set_trace(self, trace):
        assert trace == 'exact' or trace == 'hutch'
        for module in self.modules():
            if hasattr(module, 'odefunc'):
                module.odefunc.method = trace

    def forward(self, x, node_mask, edge_mask, context):
        ldj = x.new_zeros(x.shape[0])

        if self.context_net is not None:
            context = self.context_net(context)

        for transform in self.transformations:
            x, delta_logp = transform(x, node_mask, edge_mask, context)

            ldj = ldj + delta_logp

        return x

    def reverse(self, z, node_mask, edge_mask, context):
        if self.context_net is not None:
            context = self.context_net(context)

        for transform in self.transformations:
            z = transform.reverse(z, node_mask, edge_mask, context)

        return z
