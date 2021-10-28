import torch
import warnings

class Flow(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def _forward(self, *xs, **kwargs):
        raise NotImplementedError()
    
    def _inverse(self, *xs, **kwargs):
        raise NotImplementedError()
    
    def forward(self, *xs, inverse=False, **kwargs):
        if inverse:
            return self._inverse(*xs, **kwargs)
        else:
            return self._forward(*xs, **kwargs)

    def reverse(self, *xs, **kwargs):
        output, _ = self.forward(*xs, inverse=True, **kwargs)
        return output



class DensityDynamics(torch.nn.Module):
    """
    Computes the change of the system `dx/dt` over at state `x` and
    time `t`. Furthermore, computes the change of density, happening
    due to moving `x` infinitesimally in the direction `dx/dt`
    according to the "instantaneous change of variables rule" [1]
        `dlogp(p(x(t))/dt = -div(dx(t)/dt)`
    [1] Neural Ordinary Differential Equations, Chen et. al,
        https://arxiv.org/abs/1806.07366
    Parameters
    ----------
    t: PyTorch tensor
        The current time
    state: PyTorch tensor
        The current state of the system.
        Consisting of
    Returns
    -------
    (*dxs, -dlogp): Tuple of PyTorch tensors
        The combined state update of shape `[n_batch, n_dimensions]`
        containing the state update of the system state `dx/dt`
        (`dxs`) and the update of the log density (`dlogp`).
    """

    def __init__(self, dynamics):
        super().__init__()
        self._dynamics = dynamics
        self._n_evals = 0
        
    def forward(self, t, state):
        *xs, _ = state
        *dxs, dlogp = self._dynamics(t, *xs)
        return (*dxs, -dlogp)


class RegularizedDensityDynamics(torch.nn.Module):
        """
        Computes the change of the system `dx/dt` over at state `x` and
        time `t`. Furthermore, computes the change of density, happening
        due to moving `x` infinitesimally in the direction `dx/dt`
        according to the "instantaneous change of variables rule" [1]
            `dlogp(p(x(t))/dt = -div(dx(t)/dt)`
        [1] Neural Ordinary Differential Equations, Chen et. al,
            https://arxiv.org/abs/1806.07366
        Parameters
        ----------
        t: PyTorch tensor
            The current time
        state: PyTorch tensor
            The current state of the system.
            Consisting of
        Returns
        -------
        (*dxs, -dlogp): Tuple of PyTorch tensors
            The combined state update of shape `[n_batch, n_dimensions]`
            containing the state update of the system state `dx/dt`
            (`dxs`) and the update of the log density (`dlogp`).
        """

        def __init__(self, dynamics):
            super().__init__()
            self._dynamics = dynamics
            self._n_evals = 0

        def forward(self, t, state):
            *xs, _, _, _ = state
            *dxs, dlogp, t3, t4 = self._dynamics(t, *xs)
            return (*dxs, -dlogp, t3, t4)


# TODO: write docstrings


class InversedDynamics(torch.nn.Module):
    def __init__(self, dynamics, t_max=1.0):
        super().__init__()
        self._dynamics = dynamics
        self._t_max = t_max

    def forward(self, t, state):
        *dxs, trace = self._dynamics(self._t_max - t, state)
        return [-dx for dx in dxs] + [-trace]


class RegularizedInversedDynamics(torch.nn.Module):
    def __init__(self, dynamics, t_max=1.0):
        super().__init__()
        self._dynamics = dynamics
        self._t_max = t_max

    def forward(self, t, state):
        *dxs, trace, t3, t4 = self._dynamics(self._t_max - t, state)
        return [-dx for dx in dxs] + [-trace] + [t3] + [t4]


class AnodeDynamics(torch.nn.Module):
    """
    Converts the the concatenated state, which is required for the ANODE ode solver,
    to the tuple (`xs`, `dlogp`) for the following dynamics functions.
    Parameters
    ----------
    t: PyTorch tensor
        The current time
    state: PyTorch tensor
        The current state of the system
    Returns
    -------
    state: PyTorch tensor
        The combined state update of shape `[n_batch, n_dimensions]`
        containing the state update of the system state `dx/dt`
        (`state[:, :-1]`) and the update of the log density (`state[:, -1]`).
    """

    def __init__(self, dynamics):
        super().__init__()
        self._dynamics = dynamics

    def forward(self, t, state):
        xs = state[:, :-1]
        dlogp = state[:, -1:]
        state = (xs, dlogp)
        *dxs, div = self._dynamics(t, state)
        state = torch.cat([*dxs, div], dim=-1)
        return state


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
        state = (*xs, logp_init)
        ts = torch.linspace(0.0, self._t_max, self._n_time_steps).to(xs[0])
        kwargs = {**self._kwargs, **kwargs}
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
            ys = [y[-1] for y in ys]
        else:
            from deprecated.anode.adjoint import odesolver_adjoint
            state = torch.cat(state, dim=-1)
            anode_dynamics = AnodeDynamics(dynamics)
            state = odesolver_adjoint(anode_dynamics, state, options=kwargs)
            ys = [state[:, :-1]]
            dlogp = [state[:, -1:]]
        dlogp = dlogp[-1]
        return (*ys, dlogp)

    def set_trace(self, trace):
        warnings.warn("Set trace as called but this method is not using any approximation")

        pass


class RegularizedDiffEqFlow(Flow):
    def __init__(
            self,
            dynamics,
            integrator="dopri5",
            atol=1e-10,
            rtol=1e-5,
            n_time_steps=2,
            t_max=1.,
            use_checkpoints=False,
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

    def _forward(self, *xs, **kwargs):
        return self._run_ode(*xs, dynamics=self._dynamics, **kwargs)

    def _inverse(self, *xs, **kwargs):
        return self._run_ode(*xs, dynamics=self._inverse_dynamics, **kwargs)

    def _run_ode(self, *xs, dynamics, **kwargs):
        # TODO: kwargs should be parsed to avoid conflicts!
        assert (all(x.shape[0] == xs[0].shape[0] for x in xs[1:]))
        n_batch = xs[0].shape[0]
        logp_init = torch.zeros(n_batch, 1).to(xs[0])
        frobenius_init = torch.zeros_like(logp_init)
        dx2_init = torch.zeros_like(logp_init)
        state = (*xs, logp_init, frobenius_init, dx2_init)
        ts = torch.linspace(0.0, self._t_max, self._n_time_steps).to(xs[0])
        kwargs = {**self._kwargs, **kwargs}
        if not self._use_checkpoints:
            from torchdiffeq import odeint_adjoint
            *ys, dlogp, dfrobenius, d_dx2 = odeint_adjoint(
                dynamics,
                state,
                t=ts,
                method=self._integrator_method,
                rtol=self._integrator_rtol,
                atol=self._integrator_atol,
                options=kwargs
            )
            ys = [y[-1] for y in ys]
        else:
            raise NotImplementedError
            # from anode.adjoint import odesolver_adjoint
            # state = torch.cat(state, dim=-1)
            # anode_dynamics = AnodeDynamics(dynamics)
            # state = odesolver_adjoint(anode_dynamics, state, options=kwargs)
            # ys = [state[:, :-1]]
            # dlogp = [state[:, -1:]]
        dlogp = dlogp[-1]
        dfrobenius = dfrobenius[-1]
        d_dx2 = d_dx2[-1]
        return (*ys, dlogp, dfrobenius, d_dx2)
    

