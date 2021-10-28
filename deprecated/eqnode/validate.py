import torch
import numpy as np

def diversity_score(logp):
    n = len(logp)
    
    H0 = np.log(n)
    D0 = np.exp(H0)
    
    H = -np.sum(np.exp(logp) * logp)
    D = np.exp(H)
    
    # normalization has to take into account that we work in log space
    score = ((D - 1) / n) / ((D0 - 1) / n)
    
    return score


def diversity_score_torch(logp):
    n = len(logp)
    
    H0 = np.log(n)
    D0 = np.exp(H0)
    
    H = -torch.sum(torch.exp(logp) * logp)
    D = torch.exp(H)
    
    # normalization has to take into account that we work in log space
    score = ((D - 1) / n) / ((D0 - 1) / n)
    
    return score

def compute_log_weights(potential, x, z, prior, dlogp, inverse=False):
    log_weights = -potential._energy_torch(x).view(-1) + (prior.likelihood(z).view(-1) - dlogp.view(-1))
    log_weights = log_weights - torch.logsumexp(log_weights, dim=-1)
    return log_weights


