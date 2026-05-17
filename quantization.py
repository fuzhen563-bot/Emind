"""
Emind 量化推理模块 — INT4 weight-only + FP8 支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


# =============================================================================
# INT4 Quantized Linear (weight-only, per-group)
# =============================================================================

class Int4QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, group_size: int = 128, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups = (in_features + group_size - 1) // group_size

        self.register_buffer("qweight", torch.empty(out_features, self.n_groups, group_size // 2, dtype=torch.uint8))
        self.register_buffer("scales", torch.empty(out_features, self.n_groups, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    @staticmethod
    def quantize_weight(weight: torch.Tensor, group_size: int = 128) -> tuple:
        out_dim, in_dim = weight.shape
        n_groups = (in_dim + group_size - 1) // group_size

        padded_in = in_dim
        if in_dim % group_size != 0:
            padded_in = n_groups * group_size
            pad = torch.zeros(out_dim, padded_in - in_dim, dtype=weight.dtype, device=weight.device)
            weight = torch.cat([weight, pad], dim=1)

        w_groups = weight.view(out_dim, n_groups, group_size)
        scales = w_groups.abs().max(dim=-1, keepdim=True)[0] / 7.0
        scales = scales.clamp(min=1e-8)
        q = (w_groups / scales).round().clamp(-7, 7).to(torch.int8)

        # Pack two INT4 into one byte
        q = q.view(out_dim, n_groups, group_size // 2, 2)
        q_packed = (q[..., 0] & 0x0F) | ((q[..., 1] & 0x0F) << 4)
        return q_packed.to(torch.uint8), scales.squeeze(-1).to(torch.float16)

    @staticmethod
    def dequantize_weight(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        out_dim, n_groups, half_group = qweight.shape
        group_size = half_group * 2
        low = (qweight & 0x0F).to(torch.int8)
        low = torch.where(low > 7, low - 16, low)
        high = ((qweight >> 4) & 0x0F).to(torch.int8)
        high = torch.where(high > 7, high - 16, high)
        q = torch.stack([low, high], dim=-1).view(out_dim, n_groups, group_size)
        return (q.float() * scales.unsqueeze(-1).float()).to(scales.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight(self.qweight, self.scales)
        w = w[:, :self.in_features]  # remove padding
        out = F.linear(x, w, self.bias)
        return out


# =============================================================================
# FP8 Linear (requires H100 / CUDA compute 8.9+)
# =============================================================================

def has_fp8_support() -> bool:
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_capability()
    return cc[0] >= 9


class FP8Linear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        weight = linear.weight.data
        self.register_buffer("scale", weight.abs().max().unsqueeze(0))
        q = (weight.float() / self.scale).clamp(-448.0, 448.0)
        self.register_buffer("qweight", q.to(torch.float8_e4m3fn))
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.qweight.float() * self.scale
        return F.linear(x, w.to(x.dtype), self.bias)


# =============================================================================
# 量化入口
# =============================================================================

def quantize_model(model: nn.Module, mode: str = "int4", group_size: int = 128) -> nn.Module:
    if mode == "int4":
        _apply_int4(model, group_size)
    elif mode == "fp8":
        if not has_fp8_support():
            raise RuntimeError("FP8 requires H100 GPU (compute capability >= 9.0)")
        _apply_fp8(model)
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")
    return model


def _apply_int4(module: nn.Module, group_size: int = 128):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name != "lm_head":
            qweight, scales = Int4QuantizedLinear.quantize_weight(child.weight.data, group_size)
            qlin = Int4QuantizedLinear(child.in_features, child.out_features, group_size, child.bias is not None)
            qlin.qweight = qweight
            qlin.scales = scales
            if child.bias is not None:
                qlin.bias = child.bias.data.to(torch.float16)
            setattr(module, name, qlin)
        else:
            _apply_int4(child, group_size)


def _apply_fp8(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name != "lm_head":
            setattr(module, name, FP8Linear(child))
        else:
            _apply_fp8(child)


def estimate_model_size(model: nn.Module, mode: Optional[str] = None) -> str:
    total = 0
    for p in model.parameters():
        total += p.numel() * (p.element_size() if mode is None else (0.5 if "int4" in str(mode) else 1))
    if mode and "int4" in str(mode):
        total *= 0.5
    if total < 1e9:
        return f"{total / 1e6:.1f}MB"
    return f"{total / 1e9:.1f}GB"


# =============================================================================
# Bench
# =============================================================================

if __name__ == "__main__":
    from model import EmindConfig, EmindLM

    cfg = EmindConfig(d_model=512, n_heads=8, n_kv_heads=4, n_layers=2, d_ff=2048, max_seq_len=128)
    model = EmindLM(cfg)
    print(f"Original: {estimate_model_size(model)}")

    quantize_model(model, mode="int4")
    print(f"INT4: {estimate_model_size(model, 'int4')}")

    ids = torch.randint(0, cfg.vocab_size, (1, 64))
    loss, logits, _ = model(ids, labels=ids)
    print(f"Loss after INT4: {loss.item():.4f}")
