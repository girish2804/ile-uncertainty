import os
import sys
import torch
import torch.nn as nn
from contextlib import contextmanager

# Dynamic path resolution to find directory of layers.py directly,
# bypassing the package __init__.py which has incompatible vLLM and flash attention imports.
tokur_dir = os.path.dirname(os.path.abspath(__file__))
layers_dir = os.path.join(tokur_dir, 'TokUR_repo', 'bayesian_transformer', 'bayesian_transformer')
if layers_dir not in sys.path:
    sys.path.insert(0, layers_dir)

from layers import LowRankBayesianLinear


# Global cache for SVD basis vectors to prevent repeating SVD computations.
BASIS_VECTORS_CACHE = {}


def clear_tokur_cache():
    """
    Explicitly clears the global SVD basis vectors cache.
    Crucial when sequentially benchmarking multiple models to avoid memory address (id)
    collisions from CPython allocator reusing pointer IDs across deleted/new model weights.
    """
    global BASIS_VECTORS_CACHE
    BASIS_VECTORS_CACHE.clear()


@torch.no_grad()
def patched_fetch_basis_vectors(self, base_layer, bayes_noise, basis_idx=(0,)):
    """
    Patched version of LowRankBayesianLinear.fetch_basis_vectors that caches computed basis vectors
    and performs SVD on CPU to prevent GPU hangs and CUDA context conflicts with TensorFlow.
    Incorporates weight shape and dtype into cache key to prevent collision bugs.
    """
    w_shape = tuple(base_layer.weight.shape) if hasattr(base_layer, 'weight') and base_layer.weight is not None else ()
    w_dtype = base_layer.weight.dtype if hasattr(base_layer, 'weight') and base_layer.weight is not None else None
    cache_key = (id(base_layer.weight), w_shape, w_dtype, bayes_noise, basis_idx)
    if cache_key in BASIS_VECTORS_CACHE:
        self.register_buffer('basis_vectors', BASIS_VECTORS_CACHE[cache_key])
        return

    old_dtype = base_layer.weight.dtype
    # Perform SVD on CPU to avoid CUDA conflicts (especially when TensorFlow/BLEURT is initialized on GPU)
    # and to ensure maximum numerical stability.
    weight_param = base_layer.weight
    if weight_param.device.type == 'meta':
        raise RuntimeError(
            f"Layer weight for '{type(base_layer).__name__}' is a meta tensor — "
            "accelerate offloaded this layer to CPU as a placeholder with no data. "
            "Reduce the model size, use a smaller model, or set `cpu: '0GiB'` in max_memory "
            "to force all layers onto the GPU. Check run_tokur_baseline.py device_map_arg."
        )
    weight_cpu = weight_param.detach().cpu().float()
    U, _, V = torch.svd(weight_cpu)
    
    if bayes_noise == 'right':
        basis_vectors = U[:, basis_idx].to(device=base_layer.weight.device, dtype=old_dtype)
    else:
        basis_vectors = V[:, basis_idx].to(device=base_layer.weight.device, dtype=old_dtype)
        
    BASIS_VECTORS_CACHE[cache_key] = basis_vectors
    self.register_buffer('basis_vectors', basis_vectors)


def patched_forward(self, x):
    """
    Robust forward pass replacing restrictive .view(...) with .reshape(...) to avoid
    'RuntimeError: view size is not compatible with input tensor size and stride'
    under modern FlashAttention, SDPA, and RoPE non-contiguous activation tensors,
    and ensuring basis_vectors always matches the activation device and dtype.
    """
    sample = getattr(self, 'sample', True)
    if not sample:
        return self.linear(x)

    original_shape = x.shape
    x_flat = x.reshape(-1, self.linear.in_features)
    bv = self.basis_vectors.to(device=x.device, dtype=x.dtype)

    if self.bayes_noise == 'right':
        noise = torch.normal(0, 1, size=(x_flat.shape[0], bv.shape[1]), device=x.device, dtype=x.dtype)
        noise = noise * (self.bayes_sigma * x_flat.square().sum(-1, keepdim=True).sqrt())
        out_flat = self.linear(x_flat) + (noise @ bv.T)
        return out_flat.reshape(*original_shape[:-1], -1)
    elif self.bayes_noise == 'left':
        noise_scale = self.bayes_sigma * (x_flat @ bv).square().sum(-1, keepdim=True).sqrt()
        noise = noise_scale * torch.normal(0, 1, size=(x_flat.shape[0], self.linear.out_features), device=x.device, dtype=x.dtype)
        out_flat = self.linear(x_flat) + noise
        return out_flat.reshape(*original_shape[:-1], -1)


# Monkey-patch the methods on the official LowRankBayesianLinear class
LowRankBayesianLinear.fetch_basis_vectors = patched_fetch_basis_vectors
LowRankBayesianLinear.forward = patched_forward


def wrap_model_with_tokur(model: nn.Module, rank: int = 4, sigma: float = 0.05, target_modules: list = None):
    """
    In-place wraps the target nn.Linear layers of the model with official LowRankBayesianLinear layers.
    This should be run once at model initialization to avoid any overhead during inference loops.
    """
    clear_tokur_cache()  # Guaranteed clean slate for every model
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]

    bayesian_layers = []

    def _replace_layers(parent_module):
        for name, child in parent_module.named_children():
            if isinstance(child, nn.Linear) and any(tgt in name for tgt in target_modules):
                basis_idx = tuple(range(rank)) if rank > 0 else (0,)
                wrapped = LowRankBayesianLinear(
                    base_layer=child,
                    bayes_sigma=sigma,
                    bayes_noise='right',
                    basis_idx=basis_idx
                )
                setattr(parent_module, name, wrapped)
                bayesian_layers.append(wrapped)
            else:
                _replace_layers(child)

    _replace_layers(model)
    return bayesian_layers


def set_model_sampling(bayesian_layers: list, sample: bool = True):
    """
    Enables (True) or disables (False) low-rank stochastic weight perturbations.
    """
    for layer in bayesian_layers:
        layer.set_sample(sample)


@contextmanager
def apply_tokur_perturbations(model: nn.Module, rank: int = 4, sigma: float = 0.05,
                              target_modules: list = None, seed: int = None):
    """
    Context manager version for temporary applications (fallback/backward compatibility).
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if seed is not None:
        rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    replaced_modules = {}

    def _replace_layers(parent_module):
        for name, child in parent_module.named_children():
            if isinstance(child, nn.Linear) and any(tgt in name for tgt in target_modules):
                basis_idx = tuple(range(rank)) if rank > 0 else (0,)
                wrapped = LowRankBayesianLinear(
                    base_layer=child,
                    bayes_sigma=sigma,
                    bayes_noise='right',
                    basis_idx=basis_idx
                )
                setattr(parent_module, name, wrapped)
                replaced_modules[id(wrapped)] = (parent_module, name, child)
            else:
                _replace_layers(child)

    try:
        _replace_layers(model)
        yield model
    finally:
        for wrapped_id, (parent_module, name, original_child) in replaced_modules.items():
            setattr(parent_module, name, original_child)
            
        if seed is not None:
            torch.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
