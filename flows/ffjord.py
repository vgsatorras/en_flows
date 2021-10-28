import torch
import itertools
from torchdiffeq import odeint_adjoint as odeint
import numpy as np

# TODO Emiel


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


class FFJORD(torch.nn.Module):
    """
    Continuous-time flow FFJORD [1].

    Args:
        dynamics (nn.Module): The ODE dynamics function f(t,x).
        trace_method (str): The trace estimation method. One of {'exact', 'hutch'}.

    References:
        [1] FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models,
            Grathwohl et al., 2019, https://arxiv.org/abs/1810.01367
    """
    def __init__(self, dynamics, trace_method='hutch', ode_regularization=0, hutch_noise='gaussian'):
        super(FFJORD, self).__init__()

        self.odefunc = ODEfunc(
            dynamics, method=trace_method, ode_regularization=ode_regularization, hutch_noise=hutch_noise)

        self.set_integration_time()
        self.set_odeint()

    def set_integration_time(self, times=[0.0, 1.0]):
        device = next(iter(self.odefunc.parameters())).device
        self.register_buffer('int_time', torch.tensor(times, dtype=torch.float, device=device))
        self.register_buffer('inv_int_time', torch.tensor(list(reversed(times)), dtype=torch.float, device=device))

    def set_odeint(self, method='dopri5', rtol=1e-4, atol=1e-4):
        self.method = method
        self._atol = atol
        self._rtol = rtol
        self._atol_test = 1e-7
        self._rtol_test = 1e-7

    def set_trace(self, trace):
        assert trace == 'exact' or trace == 'hutch'
        self.odefunc.method = trace

    @property
    def atol(self):
        return self._atol if self.training else self._atol_test

    @property
    def rtol(self):
        return self._rtol if self.training else self._rtol_test

    def forward(self, x, node_mask=None, edge_mask=None, context=None):
        ldj = x.new_zeros(x.shape[0])
        reg_term = x.new_zeros(x.shape[0])

        state = (x, ldj, reg_term)

        self.odefunc.before_odeint(x)
        # print(state, self.odefunc, self.int_time, self.method)

        # self.odefunc.forward = self.odefunc.wrap_forward(
        #     node_mask, edge_mask, context)

        # Wrap forward, do not unwrap until backward call!!!
        if node_mask is not None or edge_mask is not None or context is not None:
            self.odefunc.dynamics.forward = self.odefunc.dynamics.wrap_forward(
                node_mask, edge_mask, context)

        statet = odeint(self.odefunc, state, self.int_time,
                        method=self.method,
                        rtol=self.rtol,
                        atol=self.atol)

        zt, ldjt, reg_termt = statet
        z, ldj, reg_term = zt[-1], ldjt[-1], reg_termt[-1]
        return z, ldj, reg_term

    def reverse_fn(self, z, node_mask=None, edge_mask=None, context=None):
        self.odefunc.before_odeint(z)
        if node_mask is not None or edge_mask is not None or context is not None:
            self.odefunc.dynamics.forward = self.odefunc.dynamics.wrap_forward(
                node_mask, edge_mask, context)

        with torch.no_grad():
            xt = odeint(self.odefunc.dynamics, z, self.inv_int_time,
                        method=self.method,
                        rtol=self.rtol,
                        atol=self.atol)

        if node_mask is not None or edge_mask is not None or context is not None:
            self.odefunc.dynamics.forward = self.odefunc.dynamics.unwrap_forward()
        return xt

    def reverse(self, z, node_mask=None, edge_mask=None, context=None):
        xt = self.reverse_fn(z, node_mask, edge_mask, context)
        x = xt[-1]
        return x

    def reverse_chain(self, z, node_mask, edge_mask, context=None):
        self.set_integration_time(times=list(np.linspace(0, 1, 50)))
        xt = self.reverse_fn(z, node_mask, edge_mask, context)
        self.set_integration_time(times=[0.0, 1.0])
        return xt


class ODEfunc(torch.nn.Module):
    def __init__(self, dynamics, method='hutch', ode_regularization=0, hutch_noise='gaussian'):
        assert method in {'exact', 'hutch'}
        super(ODEfunc, self).__init__()
        self.dynamics = dynamics
        self.hutch_noise = hutch_noise
        self.method = method
        self.ode_regularization = ode_regularization

    def set_trace_exact(self):
        self.method = 'exact'

    def set_trace_hutch(self):
        self.method = 'hutch'

    @staticmethod
    def hutch_trace(f, y, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = sum_except_batch(e_dzdx_e)
        return approx_tr_dzdx

    @staticmethod
    def only_frobenius(f, y, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        frobenius = sum_except_batch(e_dzdx.pow(2))
        return frobenius

    @staticmethod
    def hutch_trace_and_frobenius(f, y, e=None):
        """Hutchinson's estimator for the Jacobian trace"""
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        frobenius = sum_except_batch(e_dzdx.pow(2))
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = sum_except_batch(e_dzdx_e)
        return approx_tr_dzdx, frobenius

    @staticmethod
    def exact_trace(f, y):
        """Exact Jacobian trace"""
        dims = y.size()[1:]
        tr_dzdx = 0.0
        dim_ranges = [range(d) for d in dims]
        for idcs in itertools.product(*dim_ranges):
            batch_idcs = (slice(None),) + idcs
            tr_dzdx += torch.autograd.grad(f[batch_idcs].sum(), y, create_graph=True)[0][batch_idcs]
        return tr_dzdx

    def before_odeint(self, tensor):
        self.num_evals = 0
        if self.method == 'hutch':

            if self.hutch_noise == 'gaussian':
                # With _eps ~ Normal(0, 1).
                self._eps = torch.randn_like(tensor)
            elif self.hutch_noise == 'bernoulli':
                # With _eps ~ Rademacher (== Bernoulli on -1 +1 with 50/50 chance).
                self._eps = torch.randint(low=0, high=2, size=tensor.size()).to(tensor) * 2 - 1
            else:
                raise Exception("Wrong hutchinson noise type")
        #try:
        #    self.dynamics.forward = self.dynamics.unwrap_forward()
        #except:
        #    warnings.warn("Warning: dynamics.unwrap_forward() was called but there is nothing to unwrap")

    def forward(self, t, state):
        x, ldj, reg_term = state

        self.num_evals += 1
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            t.requires_grad_(True)

            # We always need the dynamics :).
            dx = self.dynamics(t, x)

            if self.ode_regularization > 0:
                # L2-squared norm of (dx)
                dx2 = sum_except_batch(dx.pow(2))

                # If trace is computed exact, frobenius norm is still estimated.
                if self.method == 'exact':
                    ldj = self.exact_trace(dx, x)
                    frobenius = self.only_frobenius(dx, x, e=self._eps)

                # Combined computation for trace and frobenius estimators.
                elif self.method == 'hutch':
                    ldj, frobenius = self.hutch_trace_and_frobenius(dx, x, e=self._eps)

                reg_term = frobenius + dx2

            else:
                if self.method == 'exact':
                    ldj = self.exact_trace(dx, x)

                elif self.method == 'hutch':
                    ldj = self.hutch_trace(dx, x, e=self._eps)

                # No regularization terms, set to zero.
                reg_term = torch.zeros_like(ldj)

        return dx, ldj, reg_term
