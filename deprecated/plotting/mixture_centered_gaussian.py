import torch
import numpy as np

from deprecated.plotting.rational_quadratic import unconstrained_rational_quadratic_spline


def sum_except_batch(x):
    return x.view(x.size(0), -1).sum(-1)


def elementwise_gaussian_log_likelihood(x, mean, log_stdev):
    elementwise = -0.5 * ((x - mean) / log_stdev.exp()) ** 2 - 0.5 * np.log(
        2 * np.pi) - log_stdev
    return elementwise


def mixture_gaussians_log_likelihood(x, log_pi, mean, log_stdev):
    B, K = log_pi.size()
    assert mean.size(1) == K
    log_probs = elementwise_gaussian_log_likelihood(
        x.unsqueeze(1), mean, log_stdev)

    log_probs = log_probs.view(B, K, -1).sum(dim=2)

    log_prob = torch.logsumexp(log_pi + log_probs, dim=1)
    return log_prob


def inv_function(x):
    unconstrained_rational_quadratic_spline()

    return


def main():
    pass


main()
