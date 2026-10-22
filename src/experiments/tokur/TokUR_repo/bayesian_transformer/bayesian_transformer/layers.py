import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankBayesianLinear(nn.Module):
    def __init__(
        self, 
        base_layer=None, 
        in_features=None, 
        out_features=None, 
        bias=True, 
        device=None, 
        dtype=None,
        bayes_sigma=1e-3,
        bayes_noise='right',
        basis_idx=(0,),
        **kwargs,
    ):
        super().__init__()
        if base_layer is None:
            self.linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        elif isinstance(base_layer, nn.Linear):
            self.linear = base_layer
        else:
            raise ValueError(f'base_layer {base_layer} must be None or nn.Linear')
        
        assert bayes_noise in ('right', 'left'), f'bayes_noise {bayes_noise} must be `right` or `left`'
        self.bayes_noise = bayes_noise
        self.bayes_sigma = bayes_sigma
        self.fetch_basis_vectors(
            base_layer=base_layer,
            bayes_noise=bayes_noise,
            basis_idx=basis_idx,
        )

        # by default, use sampling when forward.
        self.set_sample(sample=True)
    
    def forward(self, x):
        # make self.sample as an internal status of a layer.
        sample = self.sample
        # return the mean prediction
        if not sample: 
            return self.linear(x)
        
        original_shape = x.shape
        x = x.view(-1, self.linear.in_features)
        if self.bayes_noise == 'right':
            # \sum_{i=1}_{r} \eps_i b_i, where \eps_i ~ N(0, \sigma_q^2 * ||x||_2^2)
            # noise.shape: (batch_size, r)
            noise = torch.normal(0, 1, size=(x.shape[0], self.basis_vectors.shape[1]), device=x.device, dtype=x.dtype)
            # noise is proportional to the norm of x.
            noise *= self.bayes_sigma*x.square().sum(-1, keepdims=True).sqrt() # shape: (1, batch_size)
            # x.shape: (batch_size, in_features)
            # self.bayes_vector.shape: (out_features, r)
            return (self.linear(x) + noise @ self.basis_vectors.T).view(*original_shape[:-1], -1)
        elif self.bayes_noise == 'left':
            # noise_scale.shape: (batch_size, 1)
            noise_scale = self.bayes_sigma * (x @ self.basis_vectors).square().sum(-1, keepdims=True).sqrt()
            noise = noise_scale * torch.normal(0, 1, size=(x.shape[0], self.linear.out_features), device=x.device, dtype=x.dtype)
            return (self.linear(x) + noise).view(*original_shape[:-1], -1)
    
    @torch.no_grad()
    def fetch_basis_vectors(
        self, 
        base_layer, 
        bayes_noise, 
        basis_idx=(0,),
    ):
        # need to use SVD to calculate the basis vectors, which could be a bit slow.
        old_dtype = base_layer.weight.dtype
        U, _, V = torch.svd(base_layer.weight.float())
        if bayes_noise == 'right':
            # right vector is noise, so self.bayes_vector = left vector
            self.register_buffer('basis_vectors', U[:, basis_idx].to(old_dtype))
        else:
            # left vector is noise so self.bayes_vector = right vector 
            self.register_buffer('basis_vectors', V[:, basis_idx].to(old_dtype))
    
    def set_sample(self, sample=True):
        self.sample = sample
        
    def set_sigma(self, sigma=1e-3):
        self.bayes_sigma = sigma

class FullRankBayesianLinear(nn.Module):
    def __init__(
        self, 
        base_layer=None, 
        in_features=None, 
        out_features=None, 
        bias=True, 
        device=None, 
        dtype=None,
        bayes_sigma=1e-3,
        bayes_noise='self-scaling',
        **kwargs,
    ):
        super().__init__()
        if base_layer is None:
            self.linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        elif isinstance(base_layer, nn.Linear):
            self.linear = base_layer
        else:
            raise ValueError(f'base_layer {base_layer} must be None or nn.Linear')
        
        assert bayes_noise in ('self-scaling', 'std'), f'bayes_noise {bayes_noise} must be `self-scaling` or `std`'
        self.bayes_noise = bayes_noise
        self.bayes_sigma = bayes_sigma

        # by default, use sampling when forward.
        self.set_sample(sample=True)
    
    def forward(self, x):
        # make self.sample as an internal status of a layer.
        sample = self.sample
        # return the mean prediction
        if not sample: 
            return self.linear(x)

        noise = torch.normal(0, 1, size=self.linear.weight.shape, device=self.linear.weight.device, dtype=self.linear.weight.dtype)
        noise *= self.bayes_sigma
        if self.bayes_noise == 'self-scaling':
            noise *= self.linear.weight
            # we don't need abs()
            
        return (x @ (self.linear.weight+noise).T + self.linear.bias) if self.linear.bias is not None else (x @ (self.linear.weight+noise).T)

    def set_sample(self, sample=True):
        self.sample = sample
        
    def set_sigma(self, sigma=1e-3):
        self.bayes_sigma = sigma