import torch
from .distances import distance_vectors, distances_from_vectors, distance_vectors_v2, diagonal_filter
from .particle_utils import remove_mean


class ScalingLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling = torch.param()

    def forward(self, q, p, delta_t, inverse=False):

        if inverse:
            q = q - delta_t * p
            p_update = self._transform(q)
            p = p + p_update

        else:
            p_update = self._transform(q)
            p = p - p_update
            q = q + delta_t * p

        return q, p


class NICEHamiltonianLayer(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self._transform = transform

    def forward(self, q, p, delta_t, inverse=False):

        if inverse:
            q = q - delta_t * p
            p_update = self._transform(q)
            p = p + p_update

        else:
            p_update = self._transform(q)
            p = p - p_update
            q = q + delta_t * p

        return q, p

class NICERLayer(torch.nn.Module):
    def __init__(self, transform1, transform2):
        super().__init__()
        self._transform1 = transform1
        self._transform2 = transform2

    def forward(self, q, p, delta_t, inverse=False):

        if inverse:
            q_update = self._transform2(p)
            q = q - q_update
            p_update = self._transform1(q)
            p = p - p_update

        else:
            p_update = self._transform1(q)
            p = p + p_update
            q_update = self._transform2(p)
            q = q + q_update

        return q, p

class RealNVPLayer(torch.nn.Module):
    def __init__(self, t1, t2, s1, s2):
        super().__init__()
        self._t1 = t1
        self._t2 = t2
        self._s1 = s1
        self._s2 = s2

    def forward(self, q, p, delta_t, inverse=False):

        if inverse:
            q_update = self._t2(p)
            q = q - q_update
            p_update = self._t1(q)
            p = p - p_update

        else:
            log_p_scale = self._s1()
            p_update = self._t1(q)
            p = torch.exp(log_p_scale) * p + p_update
            q_update = self._t2(p)
            q = q + q_update
            dlogp = -log_p_scale.sum(dim=-1, keepdim=True)
        return q, p

class EquivariantRealNVPLayer(torch.nn.Module):
    def __init__(self, t1, t2, s1, s2):
        super().__init__()
        self._t1 = t1
        self._t2 = t2
        self._s1 = s1
        self._s2 = s2

    def forward(self, q, p, inverse=False):

        if inverse:
            log_q_scale = self._s1(p)
            q_update = self._t2(p)
            q = (q - q_update) / torch.exp(log_q_scale)
            p_update = self._t1(q)
            log_p_scale = self._s1(q)
            p = (p - p_update) / torch.exp(log_p_scale)
            dlogp = -log_p_scale * p.shape[-1] - log_q_scale * q.shape[-1]

        else:
            log_p_scale = self._s1(q)
            p_update = self._t1(q)
            p = torch.exp(log_p_scale) * p + p_update
            log_q_scale = self._s1(p)
            q_update = self._t2(p)
            q = torch.exp(log_q_scale) * q + q_update
            dlogp = log_p_scale * p.shape[-1] + log_q_scale * q.shape[-1]

        return q, p, dlogp

class HamiltonianRealNVPFlow(torch.nn.Module):

    def __init__(self, layers, n_masked=None):
        super().__init__()
        self._layers = torch.nn.ModuleList(layers)
        self._n_masked = n_masked

    def forward(self, x, inverse=False):
        if self._n_masked is None:
            self._n_masked = x.shape[-1] // 2

        if x.shape[-1] % 2 != 0:
            raise ValueError(f"P and Q need to have the same dimension. The joint dimension is currently {x.shape[-1]}")

        q, p = x[..., :self._n_masked], x[..., self._n_masked:]

        if inverse:
            layers = reversed(self._layers)
        else:
            layers = self._layers

        logp = torch.zeros(*x.shape[:-1], 1).to(x)
        for layer in layers:
            q, p, dlogp = layer(q, p, inverse=inverse)
            logp += dlogp

        return torch.cat([q, p], dim=-1), -logp

class HamiltonianFlow(torch.nn.Module):

    def __init__(self, layers, delta_t=0.1, n_masked=None):
        super().__init__()
        self._layers = torch.nn.ModuleList(layers)
        self._n_masked = n_masked
        self._delta_t = delta_t

    def forward(self, x, inverse=False):
        if self._n_masked is None:
            self._n_masked = x.shape[-1] // 2

        if x.shape[-1] % 2 != 0:
            raise ValueError(f"P and Q need to have the same dimension. The joint dimension is currently {x.shape[-1]}")

        q, p = x[..., :self._n_masked], x[..., self._n_masked:]

        if inverse:
            layers = reversed(self._layers) 
        else:
            layers = self._layers

        for layer in layers:
            q, p = layer(q, p, self._delta_t, inverse=inverse)

        return torch.cat([q, p], dim=-1)

class EquivariantFunc(torch.nn.Module):
    def __init__(
            self,
            n_particles,
            n_dim,
            distance_transform,
            types=None,
            scalar_encoder=None,
            output_transform=None,
    ):
        super().__init__()
        self._n_particles = n_particles
        self._n_dim = n_dim
        self._distance_transform = distance_transform
        self._scalar_encoder = scalar_encoder
        self._output_transform = output_transform
        self.types = types

    def _scalar(self, d, n_batch):
        x = d.unsqueeze(-1)
        if self._scalar_encoder is not None:
            x = self._scalar_encoder(x)
        if self.types is not None:
            types_batch = self.types.repeat((n_batch, 1))
            xtypes = torch.cat([x, types_batch.view(x.shape[0], x.shape[1], x.shape[2], -1)], -1)
            scalar = self._distance_transform(xtypes)
        else:
            scalar = self._distance_transform(x)

        # scalar = (scalar * t).sum(dim=-1, keepdim=True)

        if self._output_transform is not None:
            scalar = self._output_transform(scalar, d)
        return scalar

    def forward(self, x):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dim)
        r = distance_vectors(x)
        
        d = distances_from_vectors(r)
        scalar = self._scalar(d, n_batch)
        v = (r * scalar).mean(dim=2).view(n_batch, -1)
        v = remove_mean(v, self._n_particles, self._n_dim)
        return v

class InvaraintScalarFunc(torch.nn.Module):
    def __init__(
            self,
            n_particles,
            n_dim,
            distance_transform,
            types=None,
            scalar_encoder=None,
            output_transform=None,
    ):
        super().__init__()
        self._n_particles = n_particles
        self._n_dim = n_dim
        self._distance_transform = distance_transform
        self._scalar_encoder = scalar_encoder
        self._output_transform = output_transform
        self.types = types

    def _scalar(self, d, n_batch):
        x = d.unsqueeze(-1)
        if self._scalar_encoder is not None:
            x = self._scalar_encoder(x)
        if self.types is not None:
            types_batch = self.types.repeat((n_batch, 1))
            xtypes = torch.cat([x, types_batch.view(x.shape[0], x.shape[1], x.shape[2], -1)], -1)
            scalar = self._distance_transform(xtypes)
        else:
            scalar = self._distance_transform(x)

        if self._output_transform is not None:
            scalar = self._output_transform(scalar, d)
        return scalar

    def forward(self, x):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dim)
        r = distance_vectors(x)

        d = distances_from_vectors(r)
        scalar = self._scalar(d, n_batch)
        return scalar.sum(dim=[1,2,3]).view(-1,1)
