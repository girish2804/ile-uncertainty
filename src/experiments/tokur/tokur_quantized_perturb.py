import os
import sys
import torch
import torch.nn as nn
from contextlib import contextmanager

# Check for bitsandbytes linear layer types
try:
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnb_F
    BNB_AVAILABLE = True
except ImportError:
    bnb = None
    bnb_F = None
    BNB_AVAILABLE = False


def is_linear_layer(module: nn.Module) -> bool:
    """Checks if a module is a linear layer (standard PyTorch, BitsAndBytes 4-bit, or 8-bit)."""
    if isinstance(module, nn.Linear):
        return True
    if BNB_AVAILABLE:
        try:
            from bitsandbytes.nn import Linear4bit, Linear8bitLt
            if isinstance(module, (Linear4bit, Linear8bitLt)):
                return True
        except ImportError:
            pass
    mod_name = module.__class__.__name__.lower()
    return "linear" in mod_name and hasattr(module, "weight")


class QuantizedLowRankBayesianLinear(nn.Module):
    r"""
    Universal LowRankBayesianLinear layer supporting standard nn.Linear as well as
    BitsAndBytes 4-bit (Linear4bit) and 8-bit (Linear8bitLt) quantized layers.
    
    Applies low-rank stochastic weight perturbations directly in activation space:
        y = linear(x) + noise @ basis_vectors^T
    where noise ~ N(0, \sigma^2 * ||x||_2^2).
    
    Matches TokUR (Wang-ML-Lab / ICLR 2026) while completely preserving the integrity
    of 4-bit / 8-bit quantized weights without unquantizing during inference.
    """
    def __init__(
        self,
        base_layer: nn.Module,
        bayes_sigma: float = 0.05,
        bayes_noise: str = 'right',
        rank: int = 4,
        basis_idx: tuple = None
    ):
        super().__init__()
        self.linear = base_layer
        self.bayes_noise = bayes_noise
        self.bayes_sigma = bayes_sigma
        self.rank = rank
        self.in_features = getattr(base_layer, 'in_features', None)
        self.out_features = getattr(base_layer, 'out_features', None)

        if basis_idx is None:
            basis_idx = tuple(range(rank)) if rank > 0 else (0,)
        self.basis_idx = basis_idx

        # Compute basis vectors once via SVD and store in persistent buffer
        self.fetch_basis_vectors()

        # By default, sampling is disabled (deterministic greedy generation)
        self.sample = False

    @torch.no_grad()
    def fetch_basis_vectors(self):
        """
        Extracts weight representation and computes low-rank basis vectors via SVD.
        Safely handles BitsAndBytes 4-bit (Params4bit), 8-bit (Int8Params), and FP16/BF16/FP32 weights.
        """
        weight_param = self.linear.weight
        target_device = weight_param.device
        target_dtype = getattr(self.linear, 'compute_dtype', torch.bfloat16)

        # 1. Dequantize if layer is 4-bit or 8-bit
        is_4bit = BNB_AVAILABLE and hasattr(weight_param, 'quant_state') and weight_param.quant_state is not None
        is_8bit = BNB_AVAILABLE and hasattr(weight_param, 'CB')

        if is_4bit:
            # Safely dequantize 4-bit weights into CPU float32 for SVD
            weight_dequant = bnb_F.dequantize_4bit(weight_param.data, weight_param.quant_state)
            weight_cpu = weight_dequant.detach().cpu().float()
            del weight_dequant
        elif is_8bit:
            weight_dequant = bnb_F.dequantize_blockwise(weight_param.data, weight_param.quant_state)
            weight_cpu = weight_dequant.detach().cpu().float()
            del weight_dequant
        else:
            weight_cpu = weight_param.detach().cpu().float()

        # Update in/out features if not already set
        if self.out_features is None:
            self.out_features = weight_cpu.shape[0]
        if self.in_features is None:
            self.in_features = weight_cpu.shape[1]

        # 2. Fast Low-Rank SVD (torch.svd_lowrank runs in milliseconds on CPU)
        rank_q = max(self.rank, 6)
        try:
            U, S, V = torch.svd_lowrank(weight_cpu, q=rank_q)
        except Exception:
            try:
                U, S, V = torch.linalg.svd(weight_cpu, full_matrices=False)
            except Exception:
                U, S, V = torch.svd(weight_cpu)

        del weight_cpu

        idx_list = list(self.basis_idx)
        if self.bayes_noise == 'right':
            basis = U[:, idx_list]  # shape: [out_features, rank]
        else:
            basis = V[:, idx_list]  # shape: [in_features, rank]

        # Register buffer matching target device & compute dtype
        if target_device.type != 'meta':
            basis = basis.to(device=target_device, dtype=target_dtype)
        else:
            basis = basis.to(dtype=target_dtype)

        self.register_buffer('basis_vectors', basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. If sample is False, calls original linear layer directly.
        If sample is True, injects TokUR low-rank stochastic weight perturbation.
        """
        out = self.linear(x)
        if not self.sample:
            return out

        orig_shape = out.shape
        in_dim = self.in_features if self.in_features is not None else x.shape[-1]
        x_flat = x.view(-1, in_dim)

        basis = self.basis_vectors.to(device=out.device, dtype=out.dtype)
        r = basis.shape[1]

        # Sample standard normal noise: shape [batch_tokens, r]
        noise = torch.normal(0.0, 1.0, size=(x_flat.shape[0], r), device=out.device, dtype=out.dtype)
        # Scale noise proportional to the norm of x: \sigma * ||x||_2
        x_norm = x_flat.square().sum(dim=-1, keepdim=True).sqrt().to(dtype=out.dtype)
        noise = noise * (self.bayes_sigma * x_norm)

        # Low-rank perturbation in output space: noise @ basis^T -> shape [batch_tokens, out_features]
        perturbation = noise @ basis.T
        out_flat = out.view(-1, out.shape[-1]) + perturbation
        return out_flat.view(*orig_shape)

    def set_sample(self, sample: bool = True):
        self.sample = sample

    def set_sigma(self, sigma: float = 0.05):
        self.bayes_sigma = sigma


def wrap_model_with_tokur_quantized(
    model: nn.Module,
    rank: int = 4,
    sigma: float = 0.05,
    target_modules: list = None
) -> list:
    """
    In-place wraps target linear projections of the quantized model with QuantizedLowRankBayesianLinear.
    Runs once after model is loaded with device_map="auto" on GPU.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    bayesian_layers = []

    def _replace_layers(parent_module):
        for name, child in list(parent_module.named_children()):
            if is_linear_layer(child) and any(tgt in name for tgt in target_modules):
                basis_idx = tuple(range(rank)) if rank > 0 else (0,)
                wrapped = QuantizedLowRankBayesianLinear(
                    base_layer=child,
                    bayes_sigma=sigma,
                    bayes_noise='right',
                    rank=rank,
                    basis_idx=basis_idx
                )
                setattr(parent_module, name, wrapped)
                bayesian_layers.append(wrapped)
            else:
                _replace_layers(child)

    _replace_layers(model)
    return bayesian_layers


def set_model_sampling(bayesian_layers: list, sample: bool = True):
    """Enables (True) or disables (False) low-rank stochastic weight perturbations."""
    for layer in bayesian_layers:
        layer.set_sample(sample)
