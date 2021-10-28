import time

import torch

from .flows2 import Flow

# class Flow(torch.nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#     def _forward(self, *xs, **kwargs):
#         raise NotImplementedError()
#
#     def _inverse(self, *xs, **kwargs):
#         raise NotImplementedError()
#
#     def forward(self, *xs, inverse=False, **kwargs):
#         if inverse:
#             return self._inverse(*xs, **kwargs)
#         else:
#             return self._forward(*xs, **kwargs)


class RegularizedDensityDynamics(torch.nn.Module):
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

    def before_ode(self, **kwargs):
        self._dynamics.before_ode(**kwargs)

    def forward(self, t, state):
        *xs, _ = state
        *dxs, div, reg_term = self._dynamics(t, *xs)
        return (*dxs, -div, reg_term)


class RegularizedInversedDynamics(torch.nn.Module):
    def __init__(self, dynamics, t_max=1.0):
        super().__init__()
        self._dynamics = dynamics
        self._t_max = t_max

    def before_ode(self, **kwargs):
        self._dynamics.before_ode(**kwargs)

    def forward(self, t, state):
        *dxs, trace, reg_term = self._dynamics(self._t_max - t, state)
        return [-dx for dx in dxs] + [-trace], reg_term


class RegularizedDiffEqFlow(Flow):
    def __init__(
            self,
            dynamics,
            integrator="dopri5",
            atol=1e-10,
            rtol=1e-5,
            n_time_steps=2,
            t_max = 1.,
            use_checkpoints=False,
            regularization_weight=0.001,
            **kwargs
    ):
        super().__init__()
        self._dynamics = RegularizedDensityDynamics(dynamics)
        self._inverse_dynamics = RegularizedDensityDynamics(
            RegularizedInversedDynamics(dynamics, t_max))
        self._integrator_method = integrator
        self._integrator_atol = atol
        self._integrator_rtol = rtol
        self._n_time_steps = n_time_steps
        self._t_max = t_max
        self._use_checkpoints = use_checkpoints
        self._kwargs = kwargs
        self.regularization_weight = regularization_weight

    def _forward(self, *xs, **kwargs):
        return self._run_ode(*xs, dynamics=self._dynamics, **kwargs)

    def _inverse(self, *xs, **kwargs):
        return self._run_ode(*xs, dynamics=self._inverse_dynamics, **kwargs)

    def _run_ode(self, *xs, dynamics, **kwargs):
        assert(all(x.shape[0] == xs[0].shape[0] for x in xs[1:]))
        n_batch = xs[0].shape[0]
        logp_init = torch.zeros(n_batch, 1).to(xs[0])
        reg_term_init = torch.zeros(n_batch, 1).to(xs[0])
        state = [*xs, logp_init, reg_term_init]
        ts = torch.linspace(0.0, self._t_max, self._n_time_steps).to(xs[0])
        kwargs = {**self._kwargs, **kwargs}
        if "brute_force" in kwargs:
            dynamics.before_ode(**kwargs)
        else:
            dynamics.before_ode()
        if not self._use_checkpoints:
            from torchdiffeq import odeint_adjoint
            *ys, dlogp = odeint_adjoint(
                dynamics,
                state,
                t=ts,
                method=self._integrator_method,
                rtol=self._integrator_rtol,
                atol=self._integrator_atol,
                options=kwargs
            )
        else:
            from deprecated.anode.adjoint import odesolver_adjoint
            state = odesolver_adjoint(dynamics, state, options=kwargs)
        ys = [y[-1] for y in ys]
        dlogp = dlogp[-1]
        return (*ys, dlogp)


class HutchinsonEstimator(torch.nn.Module):
    def __init__(self, dynamics_function, noise=None, brute_force=False):
        super().__init__()
        self._dynamics_function = dynamics_function
        self._noise = noise
        self._brute_force = brute_force
        self._reset_noise = False
        self.counter = 0
    def before_ode(self, **kwargs):
        pass

    def reset_noise(self, reset_noise=True):
        self._reset_noise = reset_noise
    
    def forward(self, t, xs):
        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)

            t1 = time.time()
            dxs = self._dynamics_function(t, xs)
            t2 = time.time()
            print("Counter: %d \t Time dynamics_function: %.3f"  % (self.counter, t2-t1))
            self.counter += 1

            assert len(dxs.shape) == 2, "`dxs` must have shape [n_btach, system_dim]"
            system_dim = dxs.shape[-1]
            if self._brute_force is True:
                divergence = 0.0
                for i in range(system_dim):
                    ddxsi_dxs = torch.autograd.grad(
                        dxs[:, [i]],
                        xs,
                        torch.ones_like(dxs[:, [i]]),
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    divergence += ddxsi_dxs[:, [i]]
                    
            elif self._brute_force is False:
                if self._reset_noise is True:
                    self._reset_noise = False
                    self._noise = torch.randint(low=0, high=2, size=xs.shape).to(xs) * 2 - 1
                noise_ddxs = torch.autograd.grad(dxs, xs, self._noise, create_graph=True)[0]
                divergence = torch.sum((noise_ddxs * self._noise).view(-1, system_dim), 1, keepdim=True)

                frobenius_term = noise_ddxs.view(-1, system_dim).pow(2).sum(1, keepdim=True)
                #TODO implementation not finished.
            else:
                raise "Only Hutchinson and bruteforce are implemented."
        return dxs, divergence 


class BlackBoxDynamics(torch.nn.Module):
    def __init__(self, transformation, n_particles, n_dimesnion):
        super().__init__()
        self._transformation = transformation
        self._n_particles = n_particles
        self._n_dimension = n_dimesnion
        self._dim = self._n_particles * self._n_dimension

    def forward(self, t, xs):
        dxs = self._transformation(xs)
        return self._remove_mean(dxs)


    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._n_dimension)
        x = x - torch.mean(x, dim=1, keepdim=True)
        return x.view(-1, self._dim)
