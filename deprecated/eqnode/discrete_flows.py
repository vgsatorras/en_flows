import torch

class RealNVPLayer(torch.nn.Module):
    
    def __init__(self, transform):
        super().__init__()
        self._transform = transform
    
    def forward(self, x_left, x_right, inverse=False):
        if inverse:
            x_left, x_right = x_right, x_left

        log_scale_and_shift = self._transform(x_left)
        log_scale = log_scale_and_shift[..., :x_right.shape[-1]]
        shift = log_scale_and_shift[..., x_right.shape[-1]:]

        if inverse:
            x_right = torch.exp(-log_scale) * (x_right - shift)
            dlogp = -log_scale.sum(dim=-1, keepdim=True)
        else:
            x_right = torch.exp(log_scale) * x_right + shift
            dlogp = log_scale.sum(dim=-1, keepdim=True)
            
        if not inverse:
            x_left, x_right = x_right, x_left
        
        return x_left, x_right, dlogp


class DiscreteFlow(torch.nn.Module):
    
    def __init__(self, layers, n_masked=None):
        super().__init__()
        self._layers = torch.nn.ModuleList(layers)
        self._n_masked = None
    
    def forward(self, x, inverse=False):
        if self._n_masked is None:
            self._n_masked = x.shape[-1] // 2

        x_left, x_right = x[..., :self._n_masked], x[..., self._n_masked:]
            
        if inverse:
            layers = reversed(self._layers)
        else:
            layers = self._layers
        
        logp = torch.zeros(*x.shape[:-1], 1).to(x)
        for layer in layers:
            x_left, x_right, dlogp = layer(x_left, x_right, inverse=inverse)
            logp += dlogp
        
        return torch.cat([x_left, x_right], dim=-1), -logp