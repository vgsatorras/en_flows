import torch
import numpy as np

from .autograd_utils import brute_force_jacobian_trace, brute_force_jacobian
from .distances import distance_vectors, distances_from_vectors, distance_vectors_v2, diagonal_filter
from .shape_utils import tile
from .particle_utils import remove_mean


class BruteForceDynamicsFunction(torch.nn.Module):
    """
    Computes the change of the system `dx/dt` over at state `x` and
    time `t`. Furthermore, computes the change of density, happening
    due to moving `x` infinitesimally in the direction `dx/dt`
    according to the "instantaneous change of variables rule" [1]

        `dlog(p(x(t))/dt = -div(dx(t)/dt)`

    [1] Neural Ordinary Differential Equations, Chen et. al,
        https://arxiv.org/abs/1806.07366

    Parameters
    ----------
    t: PyTorch tensor
        The current time
    x: PyTorch tensor
        The current state of the system

    Returns
    -------
    dx: PyTorch tensor
        The time derivative `dx/dt`.
    trace: PyTorch tensor
        The divergence of `dx/dt`.
    """

    def __init__(self, dynamics):
        """
        """
        super().__init__()
        self._dynamics = dynamics

    def forward(self, t, x):
        with torch.enable_grad(True):
            x = x.requires_grad_(True)
            dx = self._dynamics(t, x)
            trace = brute_force_jacobian_trace(dx, x).view(-1, 1)
        return dx, trace


class LossDynamics(torch.nn.Module):

    def __init__(self, dynamics, energy):
        super().__init__()
        self._dynamics = dynamics
        self._energy = energy

    def forward(self, t, state):
        x = state[:, :-2]
        logp = state[:, -2]
        x = x.view(x.shape[0], -1)
        dl = (self._energy(x).view(-1, 1) * t - logp.view(-1, 1)).mean()
        dl = dl * torch.ones(x.shape[0], 1).to(dl)
        dx, dlogp = self._dynamics(t, x)
        return torch.cat([dx, dlogp, dl], dim=-1)

class DensityDynamics(torch.nn.Module):
    """
    Computes the change of the system `dx/dt` over at state `x` and
    time `t`. Furthermore, computes the change of density, happening
    due to moving `x` infinitesimally in the direction `dx/dt`
    according to the "instantaneous change of variables rule" [1]

        `dlog(p(x(t))/dt = -div(dx(t)/dt)`

    [1] Neural Ordinary Differential Equations, Chen et. al,
        https://arxiv.org/abs/1806.07366

    Parameters
    ----------
    t: PyTorch tensor
        The current time
    x: PyTorch tensor
        The current state of the system

    Returns
    -------
    dstate: PyTorch tensor
        The combined state update of shape `[n_batch, n_dimensions]`
        containing the state update of the system state `dx/dt`
        (`dstate[:, :-1]`) and the update of the log density (`dstate[:, -1]`).
    """

    def __init__(self, dynamics):
        super().__init__()
        self._dynamics = dynamics
        self._n_evals = 0
        
    def _reset_num_evals(self):
        self._n_evals = 0.
    
    @property
    def num_evals(self):
        return self._n_evals
    
    def has_explicit_second_order(self):
        return self._dynamics.has_explicit_second_order()
    
    def second_order_derivatives(self, t, state):
        return self._dynamics.second_order_derivatives(t, state)

    def forward(self, t, state):
        self._n_evals += 1
        x = state[:, :-1]
        dx, jacobian_trace = self._dynamics(t, x)
        return torch.cat([dx, -jacobian_trace], dim=-1)


class InversedDynamics(torch.nn.Module):
    def __init__(self, dynamics, t_max=1.0):
        super().__init__()
        self._dynamics = dynamics
        self._t_max = t_max

    def forward(self, t, state):
        dx, trace = self._dynamics(self._t_max - t, state)
        return -dx, -trace


class Reciprocal(torch.nn.Module):
    def __init__(self, eps=1e-1):
        super().__init__()
        self._log_alpha = torch.nn.Parameter(torch.zeros([1]))
        self._log_beta = torch.nn.Parameter(torch.zeros([1]))
        self._log_gamma = torch.nn.Parameter(torch.zeros([1]))
        self._uu = torch.nn.Parameter(torch.zeros([1]))
        self._eps = eps

    def forward(self, scalar, d):
        return torch.exp(self._log_gamma) * scalar + torch.exp(self._log_beta)/ (torch.exp(self._log_alpha) * d.view(*scalar.shape) ** 4 - self._uu + self._eps)
    

class Exponential(torch.nn.Module):
    def __init__(self, eps=1e-1):
        super().__init__()
        self._log_alpha = torch.nn.Parameter(torch.zeros([1]))
        self._eps = eps

    def forward(self, scalar, d):
        return scalar * torch.exp(self._log_alpha * (d.view(*scalar.shape)))


class SimpleGradientField(torch.nn.Module):
    def __init__(
            self,
            n_particles,
            n_dim,
            distance_transform,
            time_transform,
            types=None,
            scalar_encoder=None,
            time_encoder=None,
            output_transform=None,
    ):
        super().__init__()
        self._n_particles = n_particles
        self._n_dim = n_dim
        self._distance_transform = distance_transform
        self._time_transform = time_transform
        self._scalar_encoder = scalar_encoder
        self._time_encoder = time_encoder
        self._output_transform = output_transform
        self.types = types

    def _scalar(self, t, d, n_batch):
        x = d.unsqueeze(-1)
        if self._scalar_encoder is not None:
            x = self._scalar_encoder(x)
        if self._time_encoder is not None:
            ts = self._time_encoder(t)
        else:
            ts = 1.
        if self.types is not None:
            types_batch = self.types.repeat((n_batch, 1))
            xtypes = torch.cat([x, types_batch.view(x.shape[0], x.shape[1], x.shape[2], -1)], -1)
            scalar = self._distance_transform(xtypes)
        else:
            scalar = self._distance_transform(x)
        # TODO implement t correctly!!!!!!!!!!!
        scalar = (scalar * ts).sum(dim=-1, keepdim=True)
        
        if self._output_transform is not None:
            scalar = self._output_transform(scalar, d)
        return scalar

    def forward(self, t, x, compute_trace=True):
        return self._forward_and_trace(t, x, compute_trace=compute_trace)

    def _forward_and_trace(self, t, x, compute_trace=True):
        n_batch = x.shape[0]

        x = x.view(n_batch, self._n_particles, self._n_dim)
        r = distance_vectors(x)
        trace = None
        if compute_trace:
            with torch.enable_grad():
                d = distances_from_vectors(r).requires_grad_(True)
                scalar = self._scalar(t, d, n_batch)
                r2d = d.unsqueeze(-1)
#                 r2d = (r.pow(2).sum(dim=-1) / d).unsqueeze(-1)
                first_sum = (
                    torch.autograd.grad(
                        scalar, d, r2d, create_graph=True, retain_graph=True
                    )[0]
                        .sum(dim=-1)
                        .sum(dim=-1)
                )
                second_sum = (scalar * self._n_dim).sum(-1).sum(-1).sum(-1)
                trace = (first_sum + second_sum).unsqueeze(-1) / (
                        self._n_particles - 1
                )
        else:
            d = distances_from_vectors(r)
            scalar = self._scalar(d, n_batch)
        v = (r * scalar).mean(dim=2).view(n_batch, -1)
        v = remove_mean(v, self._n_particles, self._n_dim)
        return v, trace
    
    def _forward_and_trace_2(self, t, x, k, compute_trace=True):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dim)
        r = distance_vectors(x)
        trace = None
        if compute_trace:
            with torch.enable_grad():
                d = distances_from_vectors(r).requires_grad_(True)
                scalar = self._scalar(d)
                
                rbfs = self._scalar_encoder(d.unsqueeze(-1))
                r2d = (r.pow(2).sum(dim=-1) / d).unsqueeze(-1)
                
                out = r2d
                for i in range(k):
                    out = torch.autograd.grad(
                            scalar, d, out, create_graph=True, retain_graph=True
                        )[0].unsqueeze(-1)
                    out = out + (scalar * self._n_dim)
                    out = out / (self._n_particles - 1)
                s = out.view(n_batch, -1).sum() 
        else:
            d = distances_from_vectors(r)
            scalar = self._scalar(d)
        return s
    
    def _forward(self, t, r):
        n_batch = r.shape[0]
        d = distances_from_vectors(r)
        scalar = self._scalar(d)
        v = (r * scalar).mean(dim=2).view(n_batch, -1)
        return v
    
    def _forward_and_trace_3(self, t, x, compute_trace=True):
        n_batch = x.shape[0]
        with torch.enable_grad():
            x = x.requires_grad_(True).contiguous()
            x_in = x.view(-1, self._n_dim)
            x_ = x_in.view(-1, self._n_particles, self._n_dim)
            r = distance_vectors_v2(x_, x_.detach())    
            v = self._forward(t, r).view(-1, self._n_dim)     
            if compute_trace:
                jac = brute_force_jacobian(v, x_in).view(n_batch, -1, self._n_dim, self._n_dim)
            else:
                trace = None
            v = v.view(x.shape)
            return v, jac


class DistanceField(torch.nn.Module):
    def __init__(self, n_particles, n_dimensions,
                 transformer):
        super().__init__()
        self._transformer = transformer
        self._n_particles = n_particles
        self._n_dimensions = n_dimensions

    def _forward(self, t, r):
        d = distances_from_vectors(r).unsqueeze(-1)
        f = self._transformer(t, d)
        y = (r * f).mean(dim=2)
        return y

    def forward(self, t, x, compute_trace=True):
        y, trace = self._forward_and_trace(t, x, compute_trace=compute_trace)
        y = remove_mean(y, self._n_particles, self._n_dimensions)
        return y, trace

    def _forward_and_trace(self, t, x, compute_trace=True):
        with torch.enable_grad():
            x = x.requires_grad_(True).contiguous()
            x_in = x.view(-1, self._n_dimensions)
            x_ = x_in.view(-1, self._n_particles, self._n_dimensions)
            r = distance_vectors_v2(x_, x_.detach())
            v = self._forward(t, r).view(-1, self._n_dimensions)
            if compute_trace:
                trace = brute_force_jacobian_trace(v, x_in).view(-1, self._n_particles).sum(dim=-1, keepdim=True)
                trace = trace * (1 - 1 / (self._n_particles))
            else:
                trace = None
            v = v.view(x.shape)
            return v, trace
        
        
class DeepSets(torch.nn.Module):
    
    def __init__(self, f, g, h, distance_encoder, time_encoder, output_transform=None):
        super().__init__()
        self._g = g
        self._h = h
        self._f = f
        self._distance_encoder = distance_encoder
        self._time_encoder = time_encoder
        self._output_transform = output_transform
        
    
    def forward(self, t, d):
        d_ = self._distance_encoder(d)
        g = self._g(d_)
        
        s = g.sum(dim=-2)
        h = self._h(s).unsqueeze(-2)
        h = h.expand(*d.shape[:-1], h.shape[-1])
        z = torch.cat([d_, h], dim=-1)
        f = self._f(z)
        if self._output_transform is not None:
            f = self._output_transform(f, d)
        return f

    
levi_civita = torch.Tensor(np.array([
    [[0, 0, 0],
     [0, 0, 1],
     [0, -1, 0]],
    [[0, 0, -1],
     [0, 0, 0],
     [1, 0, 0]],
    [[0, 1, 0],
     [-1, 0, 0],
     [0, 0, 0]],
]))

class MultiBodyConvolution(torch.nn.Module):
    
    def __init__(self, scalar_net, c_in=1, c_out=1, activation=None):
        super().__init__()
        self._scalar_net = scalar_net
        self._activation = activation
        self._filter_conv = torch.nn.Parameter(torch.ones(c_in, c_out) / (c_in))
        self._c_in = c_in
        self._c_out = c_out
        self._levi_civita = None
        
    def _convolve(self, lc, r, f, d, k):
        lcr = torch.tensordot(r, lc, dims=[[3], [1]])
        fk_ = torch.tensordot(f, k, dims=[[-1], [0]])
        fkd_ = fk_.unsqueeze(2) * d.unsqueeze(3)
        x = (lcr.unsqueeze(-1) * fkd_.unsqueeze(3)).sum(dim=2).sum(-2)
        return x
    
    def forward(self, t, r, rbfs, f_old=None):
        if self._levi_civita is None:
            self._levi_civita = levi_civita.to(r)
        s = self._scalar_net(rbfs)
        if f_old is not None:
            r = torch.einsum("ofi, nabf, nai -> nabo", self._levi_civita, r, f_old)
        f = (s * r).sum(dim=2)
        
        return f


class ConvolutionalDynamics(torch.nn.Module):
    
    def __init__(self, n_particles, n_dims, conv_layers, dist_encoder, is_resnet=False):
        super().__init__()
        self._n_particles = n_particles
        self._n_dims = n_dims
        self._is_resnet = is_resnet
        
        self._conv_layers = torch.nn.ModuleList(conv_layers)
        self._dist_encoder = dist_encoder
    
    def forward(self, t, x, compute_trace=True):
        y, trace = self._forward_and_trace(t, x, compute_trace=compute_trace)
        return y, trace
        
    def _forward(self, t, r):
        d = distances_from_vectors(r).unsqueeze(-1)
        rbfs = self._dist_encoder(d)
        
        f = None
        for conv_layer in self._conv_layers:
            f_ = conv_layer(t, r, rbfs, f)
            if f is None or not self._is_resnet:
                f = f_
            else:
                f = f + f_
        return f
    
    def _forward_and_trace(self, t, x, compute_trace=True):
        with torch.enable_grad():
            x = x.requires_grad_(True).contiguous()
            x_in = x.view(-1, self._n_dims)
            x_ = x_in.view(-1, self._n_particles, self._n_dims)
            r = distance_vectors_v2(x_, x_.detach())    
            v = self._forward(t, r).view(-1, self._n_dims)     
            if compute_trace:
                trace = brute_force_jacobian_trace(v, x_in).view(-1, self._n_particles).sum(dim=-1, keepdim=True)
                trace = trace * (1 - 1 / (self._n_particles))
            else:
                trace = None
            v = remove_mean(v, self._n_particles, self._n_dims)
            v = v.view(x.shape)
            return v, trace

        
class HelmholtzDynamics(torch.nn.Module):
    
    def __init__(self, n_particles, n_dims, scalar_radial_net, scalar_rotational_net, 
                 dist_encoder, output_transform_radial=None, output_transform_rotational=None, types=None):
        super().__init__()
        self._n_particles = n_particles
        self._n_dim = n_dims
        
        self._scalar_radial_net = scalar_radial_net
        self._scalar_rotational_net = scalar_rotational_net
        
        self._output_transform_radial = output_transform_radial
        self._output_transform_rotational = output_transform_rotational
        
        self._dist_encoder = dist_encoder
        
        self._levi_civita = None
        self.types = types
    
    def forward(self, t, x, compute_trace=True):
        y, trace = self._forward_and_trace(t, x, compute_trace=compute_trace)
        return y, trace

    def _scalar_radial(self, rbfs, d, n_batch):
        if self.types is not None:
            types_batch = self.types.repeat((n_batch, 1))
            # print(x.shape, types_batch.view(x.shape[0], x.shape[1], x.shape[2], -1).shape, self.types.device)
            xtypes = torch.cat([rbfs, types_batch.view(rbfs.shape[0], rbfs.shape[1], rbfs.shape[2], -1)], -1)
            s = self._scalar_radial_net(xtypes)
        else:
            s = self._scalar_radial_net(rbfs)
        if self._output_transform_radial is not None:
            s = self._output_transform_radial(s, d)
        return s
    
    def _scalar_rotational(self, rbfs, d, n_batch):
        if self.types is not None:
            types_batch = self.types.repeat((n_batch, 1))
            # print(x.shape, types_batch.view(x.shape[0], x.shape[1], x.shape[2], -1).shape, self.types.device)
            xtypes = torch.cat([rbfs, types_batch.view(rbfs.shape[0], rbfs.shape[1], rbfs.shape[2], -1)], -1)
            s = self._scalar_rotational_net(xtypes)
        else:
            s = self._scalar_rotational_net(rbfs)
        if self._output_transform_rotational is not None:
            s = self._output_transform_rotational(s, d)
        return s
    
    def _forward_and_trace(self, t, x, compute_trace=True):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dim)
        r = distance_vectors(x)
        trace = None
        if compute_trace:
            with torch.enable_grad():
                d = distances_from_vectors(r).requires_grad_(True).unsqueeze(-1)
                
                if self._dist_encoder is not None:
                    rbfs = self._dist_encoder(d)
                else:
                    rbfs = d

                
                s1 = self._scalar_radial(rbfs, d, n_batch)
                s2 = self._scalar_rotational(rbfs, d, n_batch)
                
                
                r2d = (r.pow(2).sum(dim=-1, keepdim=True) / d)

                first_sum = (
                    torch.autograd.grad(
                        s1, d, r2d, create_graph=True, retain_graph=True
                    )[0]
                    .sum(dim=1)
                    .sum(dim=1)
                )
                second_sum = (s1 * self._n_dim).sum(1).sum(1)
                trace = (first_sum + second_sum) / (
                    self._n_particles - 1
                )
        else:
            d = distances_from_vectors(r)
            scalar = self._scalar(d)

        v = (r * s1).mean(dim=2)

        if self._levi_civita is None:
            self._levi_civita = levi_civita.to(r)
        r = torch.einsum("ofi, nabf, nai -> nabo", self._levi_civita, r, v)
        v = v + (r * s2).mean(dim=2)
        
        v = v.view(n_batch, -1)
        
        v = remove_mean(v, self._n_particles, self._n_dim)
        
        return v, trace

class FeatureDynamics(torch.nn.Module):
    
    def __init__(self, n_particles, n_dims, n_feature_dims, s1, s2, f,
                 dist_encoder, output_transform, is_resnet=False):
        super().__init__()
        self._n_particles = n_particles
        self._n_dim = n_dims
        self._n_feature_dims = n_feature_dims
        self._is_resnet = is_resnet
        
        self._s1 = s1
        self._s2 = s2
        
        self._feat_transf = f
        
        self._output_transform = output_transform 
        
        self._dist_encoder = dist_encoder
        
        self._levi_civita = None
    
    def forward(self, t, x, compute_trace=True):
        y, trace = self._forward_and_trace(t, x, compute_trace=compute_trace)
        return y, trace
        
    def _forward(self, t, r):
        d = distances_from_vectors(r).unsqueeze(-1)
        rbfs = self._dist_encoder(d)
        
        f = None
        for conv_layer in self._conv_layers:
            f_ = conv_layer(t, r, rbfs, f)
            if f is None or not self._is_resnet:
                f = f_
            else:
                f = f + f_
        return f
    
    def _feature_dynamics(self, t, rbfs, feats):
        
        n_batch = rbfs.shape[0]
        
        
        inp = torch.cat([rbfs, torch.softmax(feats, dim=-1)], dim=-1)
        
        v_feats = self._feat_transf(inp)
        
        v_feats = v_feats.mean(dim=2)
        
        return v_feats
    
    def _forward_and_trace(self, t, state, compute_trace=True):
        n_batch = state.shape[0]
        x = state[:, :self._n_particles * self._n_dim].view(n_batch, self._n_particles, self._n_dim)
        feats = state[:, self._n_particles * self._n_dim:].view(n_batch, self._n_particles, -1)
        
        r = distance_vectors(x)
        trace = None
        if compute_trace:
            with torch.enable_grad():
                d = distances_from_vectors(r).requires_grad_(True).unsqueeze(-1)
                
                rbfs = self._dist_encoder(d)
                filt = diagonal_filter(self._n_particles, self._n_particles, rbfs.device)
                
                other_feats = tile(feats.unsqueeze(1), dim=1, n_tile=self._n_particles)
                other_feats = torch.transpose(other_feats, 1, 2)
                other_feats = other_feats[:, filt].view(n_batch, self._n_particles, self._n_particles-1, -1)

                inp = torch.cat([rbfs, feats.unsqueeze(2).expand_as(other_feats), other_feats], axis=-1)

                s1 = self._s1(inp)
                if self._output_transform is not None:
                    s1 = self._output_transform(s1, d)
                
                s2 = self._s2(inp)
                if self._output_transform is not None:
                    s2 = self._output_transform(s2, d)
                
                r2d = (r.pow(2).sum(dim=-1, keepdims=True) / d)

                first_sum = (
                    torch.autograd.grad(
                        s1, d, r2d, create_graph=True, retain_graph=True
                    )[0]
                    .sum(dim=1)
                    .sum(dim=1)
                )
                second_sum = (s1 * self._n_dim).sum(1).sum(1)
                trace = (first_sum + second_sum) / (
                    self._n_particles - 1
                )
        else:
            d = distances_from_vectors(r)
            scalar = self._scalar(d)
            
        v_feats = self._feature_dynamics(None, rbfs, other_feats).view(n_batch, -1)

        v_x = (r * s1).mean(dim=2)

        if self._levi_civita is None:
            self._levi_civita = levi_civita.to(r)
        r = torch.einsum("ofi, nabf, nai -> nabo", self._levi_civita, r, v_x)
        v_x = v_x + (r * s2).mean(dim=2)
        
        v_x = remove_mean(v_x, self._n_particles, self._n_dim).view(n_batch, -1)
        
        v = torch.cat([v_x, v_feats], axis=-1)
        
        v = v.reshape(n_batch, -1)
        
        
        
        return v, trace


def rbf_kernels(d, mu, neg_log_gamma, derivative=False):
    inv_gamma = torch.exp(neg_log_gamma)
    rbfs = torch.exp(-(d - mu).pow(2) * inv_gamma.pow(2))
    srbfs = rbfs.sum(dim=-1, keepdim=True)
    kernels = rbfs / (1e-6 + srbfs)
    if derivative:
        drbfs = -2 * (d - mu) * inv_gamma.pow(2) * rbfs
        sdrbfs = drbfs.sum(dim=-1, keepdim=True)
        dkernels = drbfs / (1e-6 + srbfs) - rbfs * sdrbfs / (1e-6 + srbfs ** 2)
    else:
        dkernels = None
    return kernels, dkernels


class KernelDynamics(torch.nn.Module):
    
    def __init__(self, n_particles, n_dimensions, 
                 mus, gammas, 
                 mus_time=None, gammas_time=None,
                 optimize_d_gammas=False,
                 optimize_t_gammas=False):
        super().__init__()
        self._n_particles = n_particles
        self._n_dimensions = n_dimensions

        self.register_buffer('_mus', mus)
        self.register_buffer('_neg_log_gammas', -torch.log(gammas))
        self._n_kernels = self._mus.shape[0]

        self.register_buffer('_mus_time', mus_time)
        self.register_buffer('_neg_log_gammas_time', -torch.log(gammas_time))

        if self._mus_time is None:
            self._n_out = 1
        else:
            assert self._neg_log_gammas_time is not None and self._neg_log_gammas_time.shape[0] == self._mus_time.shape[0]
            self._n_out = self._mus_time.shape[0]
        
        if optimize_d_gammas:
            self._neg_log_gammas = torch.nn.Parameter(self._neg_log_gammas)
            
        if optimize_t_gammas:
            self._neg_log_gammas_time = torch.nn.Parameter(self._neg_log_gammas_time)
        
        
        
        self._weights = torch.nn.Parameter(
            torch.Tensor(self._n_kernels, self._n_out).normal_() * np.sqrt(1. / self._n_kernels)
        )
        self._bias = torch.nn.Parameter(
            torch.Tensor(1, self._n_out).zero_()
        )
        
        self._importance = torch.nn.Parameter(
            torch.Tensor(self._n_kernels).zero_()
        )

    def before_ode(self):
        pass
        
    def _force_mag(self, t, d, derivative=False):
        
        importance = self._importance
        
        rbfs, d_rbfs = rbf_kernels(d, self._mus, self._neg_log_gammas, derivative=derivative)    
        
        force_mag = (rbfs + importance.pow(2).view(1, 1, 1, -1)) @ self._weights + self._bias
        if derivative:
            d_force_mag = (d_rbfs) @ self._weights
        else:
            d_force_mag = None
        if self._mus_time is not None:
            trbfs, _ = rbf_kernels(t, self._mus_time, self._neg_log_gammas_time)
            force_mag = (force_mag * trbfs).sum(dim=-1, keepdim=True)
            if derivative:
                d_force_mag = (d_force_mag * trbfs).sum(dim=-1, keepdim=True)
        return force_mag, d_force_mag



    def forward(self, t, x, compute_divergence=True):
        n_batch = x.shape[0] # size = (n_batch, n_particles * n_dimensions)

        x = x.view(n_batch, self._n_particles, self._n_dimensions)
        r = distance_vectors(x)
        
        d = distances_from_vectors(r).unsqueeze(-1)
        
        force_mag, d_force_mag = self._force_mag(t, d, derivative=compute_divergence)
        forces = (r * force_mag).sum(dim=-2)
        forces = forces.view(n_batch, -1)

        if compute_divergence:
            divergence = (d * d_force_mag + self._n_dimensions * force_mag).view(n_batch, -1).sum(dim=-1)
            divergence = divergence.unsqueeze(-1)
        else:
            divergence = None

        # forces.size() --> (n_batch, n_particles * n_dimensions)
        # divergence.size() --> (n_batch, 1)
        return forces, divergence
    
class KernelDynamics_V2(torch.nn.Module):
    
    def __init__(self, n_particles, n_dimensions, 
                 mus, gammas, 
                 mus_time=None, gammas_time=None,
                 optimize_d_gammas=False,
                 optimize_t_gammas=False):
        super().__init__()
        self._n_particles = n_particles
        self._n_dimensions = n_dimensions
        
        self._mus = mus
        self._neg_log_gammas = -torch.log(gammas)
        
        self._n_kernels = self._mus.shape[0]
        
        self._mus_time = mus_time
        self._neg_log_gammas_time = -torch.log(gammas_time)
        
        if self._mus_time is None:
            self._n_out = 1
        else:
            assert self._neg_log_gammas_time is not None and self._neg_log_gammas_time.shape[0] == self._mus_time.shape[0]
            self._n_out = self._mus_time.shape[0]
        
        if optimize_d_gammas:
            self._neg_log_gammas = torch.nn.Parameter(self._neg_log_gammas)
            
        if optimize_t_gammas:
            self._neg_log_gammas_time = torch.nn.Parameter(self._neg_log_gammas_time)
            
        self._importance = torch.nn.Parameter(
            torch.Tensor(self._n_kernels).zero_()
        )
        
        
        
        self._weights = torch.nn.Parameter(
            torch.Tensor(self._n_kernels, self._n_out).normal_() * np.sqrt(1. / self._n_kernels)
        )
        self._bias = torch.nn.Parameter(
            torch.Tensor(1, self._n_out).zero_() 
        )
        
        self._bias_out = torch.nn.Parameter(
            torch.Tensor(self._n_out, 1).zero_()
        )
        
        self._softplus = torch.nn.Softplus()
        
    def _force_mag(self, t, d, derivative=False):
        
        importance = self._importance
        
        rbfs, d_rbfs = rbf_kernels(d, self._mus, self._neg_log_gammas, derivative=derivative)    
        
        force_mag = (rbfs + importance.pow(2).view(1, 1, 1, -1)) @ self._weights + self._bias
        if derivative:
            d_force_mag = (d_rbfs) @ self._weights
        else:
            d_force_mag = None
        if self._mus_time is not None:
            trbfs, _ = rbf_kernels(t, self._mus_time, self._neg_log_gammas_time)
            force_mag = (force_mag * trbfs).sum(dim=-1, keepdim=True)
            if derivative:
                d_force_mag = (d_force_mag * trbfs).sum(dim=-1, keepdim=True)
        return force_mag, d_force_mag, trbfs
    
    
    def _activation(self, x):
        return self._softplus(x) - np.log(np.exp(0) + 1)
    
    def _d_activation(self, x):
        return torch.sigmoid(x)

    def forward(self, t, x, compute_divergence=True):
        n_batch = x.shape[0]

        x = x.view(n_batch, self._n_particles, self._n_dimensions)
        r = distance_vectors(x)
        
        d = distances_from_vectors(r).unsqueeze(-1)
        
        force_mag, d_force_mag, trbfs = self._force_mag(t, d, derivative=compute_divergence)
        
        forces = (r * force_mag).sum(dim=-2, keepdim=True)
        
        _force_norms_pre_sqr = forces.pow(2).sum(dim=-1, keepdim=True)
        _force_norms_pre = (_force_norms_pre_sqr + 1e-3).sqrt() 
        
        
        bias = (self._bias_out.pow(2) * trbfs).sum()
        pre_activation = _force_norms_pre + bias
        _force_norms_post = self._activation(pre_activation)
        _d_force_norms_post = self._d_activation(pre_activation)
        
        _forces_normed = forces / _force_norms_pre
        
        divergence = (d * d_force_mag + self._n_dimensions * force_mag)
        
        xJx = (
            d_force_mag / (d + 1e-6) * (r * _forces_normed).sum(dim=-1, keepdim=True).pow(2)
          + force_mag
        )
        
        _divergence1 = _d_force_norms_post * xJx
        _divergence2 = _force_norms_post / _force_norms_pre * divergence
        _divergence3 = _force_norms_post / _force_norms_pre * xJx

        _divergence = _divergence1 + _divergence2 - _divergence3
        
        _forces = (_forces_normed * _force_norms_post).sum(dim=-2)
        
        return _forces.view(n_batch, -1), _divergence.view(n_batch, -1).sum(dim=-1, keepdim=True)


class SchNet(torch.nn.Module):
    def __init__(self, transformation, rbf_encoder, n_particles, n_dimesnion, features, feature_encoding, cfconv1, cfconv2, cfconv3):
        super().__init__()
        self._transformation = transformation
        self._feature_encoding = feature_encoding
        self._cfconv1 = cfconv1
        self._cfconv2 = cfconv2
        self._cfconv3 = cfconv3
        self._rbf_encoder = rbf_encoder
        self._n_particles = n_particles
        self._n_dimension = n_dimesnion
        self._dim = self._n_particles * self._n_dimension
        self._feature_shape = features

    def forward(self, t, xs):
        n_batch = xs.shape[0]
        xs = xs.view(n_batch, self._n_particles, self._n_dimension)

        r = distance_vectors(xs)
        d = distances_from_vectors(r).unsqueeze(-1)
        
        rbfs = self._rbf_encoder(d)
        features = torch.ones(n_batch * self._n_particles).to(xs)
        features = self._feature_encoding(features.view(-1,1))
        # (b,n,f)
        features = features.view(n_batch, self._n_particles, self._feature_shape)
        
        # (b, n, n, rbf) -->(bnn, rbf)
        rbfs = rbfs.view(n_batch * self._n_particles * (self._n_particles - 1), -1)
        
        # cfconv1
        # (bnn, rbf) --> (bnn, f)
        W = self._cfconv1(rbfs)
        # calc update by elementwise mult (b,n,n,f)
        update = features.unsqueeze(2) * W.view(n_batch, self._n_particles, self._n_particles - 1, self._feature_shape)
        # features are update with sum of all filtered features
        features = features + update.sum(2)
        features = features.view(n_batch, self._n_particles, self._feature_shape)
        
        # cfconv2
        # (bnn, rbf) --> (bnn, f)
        W = self._cfconv2(rbfs)
        # calc update by elementwise mult (b,n,n,f)
        update = features.unsqueeze(2) * W.view(n_batch, self._n_particles, self._n_particles - 1, self._feature_shape)
        # features are update with sum of all filtered features
        features = features + update.sum(2)
        features = features.view(n_batch, self._n_particles, self._feature_shape)
        
        # cfconv3
        # (bnn, rbf) --> (bnn, f)
        W = self._cfconv3(rbfs)
        # calc update by elementwise mult (b,n,n,f)
        update = features.unsqueeze(2) * W.view(n_batch, self._n_particles, self._n_particles - 1, self._feature_shape)
        # features are update with sum of all filtered features
        features = features + update.sum(2)
        
        # add features for potential
        features = self._transformation(features.view(-1, self._feature_shape))
        potential = features.view(n_batch, -1).sum(-1)
        dxs = -1. * torch.autograd.grad(potential, xs, torch.ones_like(potential), create_graph=True, only_inputs=False)[0]
        return self._remove_mean(dxs)

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dimension)
        x = x - torch.mean(x, dim=1, keepdim=True)
        return x.view(-1, self._dim)


class SimpleEqDynamics(torch.nn.Module):
    def __init__(self, transformation, rbf_encoder, n_particles, n_dimesnion, n_rbfs):
        super().__init__()
        self._transformation = transformation
        self._rbf_encoder = rbf_encoder
        self._n_particles = n_particles
        self._n_dimension = n_dimesnion
        self._dim = self._n_particles * self._n_dimension
        self._n_rbfs = n_rbfs

    def forward(self, t, xs):
        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)
            t.requires_grad_(True)

            n_batch = xs.shape[0]
            xs = xs.view(n_batch, self._n_particles, self._n_dimension)

            r = distance_vectors(xs)
            d = distances_from_vectors(r).unsqueeze(-1)
            rbfs = self._rbf_encoder(d)
            features = self._transformation(rbfs.view(-1, self._n_rbfs))
            potential = features.view(n_batch, -1).sum(-1)

            dxs = -1. * torch.autograd.grad(potential, xs, torch.ones_like(potential), create_graph=True, only_inputs=False)[0]
            dxs = dxs.view(-1, self._n_particles, self._n_dimension)
            output = self._remove_mean(dxs)
        return output


    def _remove_mean(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        return x
