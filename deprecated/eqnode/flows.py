import torch
from torchdiffeq import odeint_adjoint

from deprecated.anode.adjoint import odesolver_adjoint
from .dynamics import InversedDynamics, LossDynamics, DensityDynamics


class Flow(torch.nn.Module):
    def forward(self, x, inverse=False):
        pass


class StackedFlow(torch.nn.Module):
    def __init__(self, flows):
        super().__init__()
        self._flows = flows

    def forward(self, x, inverse=False):
        n_batch = x.shape[0]
        flows = self._flows
        if inverse:
            flows = reversed(flows)
        logp = torch.zeros(n_batch, 1)
        for flow in flows:
            x, dlogp = flow(x, inverse=inverse)
            logp = logp + dlogp
        return x, logp


class ContinuousNormalizingFlow(torch.nn.Module):
    def __init__(
        self,
        dynamics,
        integrator="dopri5",
        atol=1e-10,
        rtol=1e-5,
        n_time_steps=2,
        **kwargs
    ):
        super().__init__()
        self._dynamics = DensityDynamics(dynamics)
        self._inverse_dynamics = DensityDynamics(InversedDynamics(dynamics))
        self._integrator_method = integrator
        self._integrator_atol = atol
        self._integrator_rtol = rtol
        self._n_time_steps = n_time_steps
        self._kwargs = kwargs

    def forward(self, x, t=1.0, inverse=False, keep_all=False, checkpoint=False, **kwargs):
        n_batch = x.shape[0]
        logp_init = torch.zeros(n_batch, 1).to(x)
        state = torch.cat([x, logp_init], dim=-1).contiguous()
        if inverse:
            dynamics = self._inverse_dynamics
            dynamics._t_max = t
        else:
            dynamics = self._dynamics
        ts = torch.linspace(0.0, t, self._n_time_steps).to(x)
        
        kwargs = {**self._kwargs, **kwargs}
        
        if not checkpoint:
            state = odeint_adjoint(
                dynamics,
                state,
                t=ts,
                method=self._integrator_method,
                rtol=1e-5,
                atol=1e-10,
                options=kwargs
            )
        else:
            state = odesolver_adjoint(dynamics, state, options=kwargs)
        
        if len(state.shape) < 3 or not keep_all:
            if len(state.shape) > 2:
                state = state[-1]
            x = state[:, :-1]
            logp = state[:, -1:]
        else:
            x = [s[:, :-1] for s in state]
            logp = [s[:, -1:] for s in state]

        return x, logp

    def set_trace(self, trace):
        pass

class ContinuousNormalizingFlowOld(torch.nn.Module):
    def __init__(
            self,
            dynamics,
            integrator="dopri5",
            atol=1e-10,
            rtol=1e-5,
            n_time_steps=2,
    ):
        super().__init__()
        self._dynamics = dynamics
        self._inverse_dynamics = InversedDynamics(dynamics)
        self._integrator_method = integrator
        self._integrator_atol = atol
        self._integrator_rtol = rtol
        self._n_time_steps = n_time_steps

    def forward(self, x, t=1.0, inverse=False, keep_all=False):
        n_batch = x.shape[0]
        logp_init = torch.zeros(n_batch, 1).to(x)
        state = torch.cat([x, logp_init], dim=-1).contiguous()
        if inverse:
            dynamics = self._inverse_dynamics
            dynamics._t_max = t
        else:
            dynamics = self._dynamics
        ts = torch.linspace(0.0, t, self._n_time_steps).to(x)

        state = odeint_adjoint(
            dynamics,
            state,
            t=ts,
            method=self._integrator_method,
            rtol=1e-5,
            atol=1e-10,
        )

        if not keep_all:
            state = state[-1]
            x = state[:, :-1]
            logp = state[:, -1:]
        else:
            x = [s[:, :-1] for s in state]
            logp = [s[:, -1:] for s in state]

        return x, logp
    

class ContinuousNormalizingFlowMultiTemperature(torch.nn.Module):
    def __init__(
        self,
        dynamics,
        energy,
        integrator="dopri5",
        atol=1e-10,
        rtol=1e-5,
        n_time_steps=2,
    ):
        super().__init__()
        self._dynamics = LossDynamics(dynamics, energy)
        self._energy = energy
        self._inverse_dynamics = InversedDynamics(dynamics)
        self._integrator_method = integrator
        self._integrator_atol = atol
        self._integrator_rtol = rtol
        self._n_time_steps = n_time_steps

    def forward(self, x, t=1.0, inverse=False):
        n_batch = x.shape[0]
        logp_init = torch.zeros(n_batch, 1).to(x)
        loss_init = torch.zeros(n_batch, 1).to(x)
        state = torch.cat([x, logp_init, loss_init], dim=-1).contiguous()
        if inverse:
            dynamics = self._inverse_dynamics
            dynamics._t_max = t
        else:
            dynamics = self._dynamics
        ts = torch.linspace(0.0, t, self._n_time_steps).to(x)
        
        state = odeint_adjoint(
            dynamics,
            state,
            t=ts,
            method=self._integrator_method,
            rtol=1e-5,
            atol=1e-10,
        )[-1]
        
        x = state[:, :-2]
        logp = state[:, -2]
        loss = state[:, -1]

        return x, logp, loss

class DiffEqFlow(Flow):
    def __init__(
            self,
            dynamics,
            integrator="dopri5",
            atol=1e-10,
            rtol=1e-5,
            n_time_steps=2,
            t_max = 1.,
            use_checkpoints=False,
            **kwargs
    ):
        super().__init__()
        self._dynamics = DensityDynamics(dynamics)
        self._inverse_dynamics = DensityDynamics(InversedDynamics(dynamics, t_max))
        self._integrator_method = integrator
        self._integrator_atol = atol
        self._integrator_rtol = rtol
        self._n_time_steps = n_time_steps
        self._t_max = t_max
        self._use_checkpoints = use_checkpoints
        self._kwargs = kwargs

    def _forward(self, *xs, **kwargs):
        return self._run_ode(*xs, dynamics=self._dynamics, **kwargs)

    def _inverse(self, *xs, **kwargs):
        return self._run_ode(*xs, dynamics=self._inverse_dynamics, **kwargs)

    def _run_ode(self, *xs, dynamics, **kwargs):
        # TODO: kwargs should be parsed to avoid conflicts!
        assert(all(x.shape[0] == xs[0].shape[0] for x in xs[1:]))
        n_batch = xs[0].shape[0]
        logp_init = torch.zeros(n_batch, 1).to(xs[0])
        state = [*xs, logp_init]
        ts = torch.linspace(0.0, self._t_max, self._n_time_steps).to(xs[0])
        kwargs = {**self._kwargs, **kwargs}
        # remove this and give every dynmaics kwargs
        if "brute_force" in kwargs:
            print("yes")
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