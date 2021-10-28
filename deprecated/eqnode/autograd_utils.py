import torch


def brute_force_jacobian_trace(y, x):
    """
    Computes the trace of the jacobian matrix `dy/dx` by
    brute-forcing over the components of `x`. This requires
    `O(D)` backward passes, where D is the dimension of `x`.

    Parameters
    ----------
    y: PyTorch tensor
        Result of a computation depending on `x` of shape
        `[n_batch, n_dimensions_in]`
    x: PyTorch tensor
        Argument to the computation yielding `y` of shape
        `[n_batch, n_dimensions_out]`

    Returns
    -------
    trace: PyTorch tensor
        Trace of Jacobian matrix `dy/dx` of shape `[n_batch, 1]`.

    Examples
    --------
    TODO
    """
    system_dim = x.shape[-1]
    trace = 0.0
    for i in range(system_dim):
        dyi_dx = torch.autograd.grad(
            y[:, i],
            x,
            torch.ones_like(y[:, i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        trace = trace + dyi_dx[:, i]
    return trace.contiguous()

def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

def sample_gaussian_like(y):
    return torch.randn_like(y)

def approx_jacobian_trace(y, x, rademacher=False, _e = None):
    """
    Computes the trace of the jacobian matrix `dy/dx` by
    using the Hutchingson trace estimator.

    Parameters
    ----------
    y: PyTorch tensor
        Result of a computation depending on `x` of shape
        `[n_batch, n_dimensions_in]`
    x: PyTorch tensor
        Argument to the computation yielding `y` of shape
        `[n_batch, n_dimensions_out]`

    Returns
    -------
    trace: PyTorch tensor
        Trace of Jacobian matrix `dy/dx` of shape `[n_batch, 1]`.

    Examples
    --------
    TODO
    """

    batchsize = x.shape[0]

    if _e is None:
        if rademacher:
            _e = sample_rademacher_like(x)
        else:
            _e = sample_gaussian_like(x)
            # e_vjp_dhdy = torch.autograd.grad(h, y, self._e, create_graph=True)[0]
    # e_vjp_dfdy = torch.autograd.grad(dy, h, e_vjp_dhdy, create_graph=True)[0]


    e_vjp_dfdy = torch.autograd.grad(y, x, _e, create_graph=True)[0]


    #print("divergence_fn_autograd: ", start_a.elapsed_time(end_a))
    #print(h.shape, y.shape, self._e.shape)
    #print(dy.shape)
    #print(e_vjp_dhdy.shape, e_vjp_dfdy.shape, (e_vjp_dfdy * self._e).view(batchsize, -1).shape)
    divergence = torch.sum((e_vjp_dfdy * _e).view(batchsize, -1), 1, keepdim=True)
    return divergence


def brute_force_jacobian(y, x):
    """
    TODO docstring
    """
    dim_i = y.shape[-1]
    dim_j = x.shape[-1]
    trace = 0.0
    
    outs = []
    for j in range(dim_j):
        for i in range(dim_i):
            dy_dx = torch.autograd.grad(
                y[..., i].sum(), x, retain_graph=True
            )[0]
            print(dy_dx)
            outs.append(dy_dx[..., j])
    
    jac = torch.cat(outs).view(-1, dim_i, dim_j)
    
    return jac


class _ClipGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, max_norm):
        ctx._max_norm = max_norm
        return input

    @staticmethod
    def backward(ctx, grad_output):
        max_norm = ctx._max_norm
        grad_norm = torch.norm(grad_output, p=2, dim=1)
        coeff = max_norm / torch.max(grad_norm, max_norm * torch.ones_like(grad_norm))
        return grad_output * coeff.view(-1, 1), None, None

clip_grad = _ClipGradient.apply