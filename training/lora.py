"""
Emind LoRA — Low-Rank Adaptation for efficient fine-tuning.
"""
import math
import torch
import torch.nn as nn
from typing import Optional, List

from model import EmindLM, EmindConfig


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x @ self.A @ self.B) * self.scaling


def apply_lora(model: EmindLM, target_modules: Optional[List[str]] = None, rank: int = 8, alpha: float = 16.0):
    """
    Wrap linear layers in model with LoRA.
    Only affects named modules in target_modules (default: q_proj, k_proj, v_proj, o_proj).
    """
    if target_modules is None:
        target_modules = ["W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"]

    lora_params = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        base_name = name.split(".")[-1]
        if not any(t in base_name for t in target_modules):
            continue

        lora = LoRALayer(module.in_features, module.out_features, rank=rank, alpha=alpha)
        setattr(module, "lora", lora)

        # 冻结原始线性权重，只训练 LoRA 参数
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False

        # 保存原始 forward，供 merge_lora 恢复
        module._original_forward = module.forward

        def new_forward(x, _m=module):
            return _m._original_forward(x) + _m.lora(x)

        module.forward = new_forward
        lora_params += sum(p.numel() for p in lora.parameters())

    print(f"LoRA applied to {len(target_modules)} module types, {lora_params:,} trainable params")
    return model


def merge_lora(model: EmindLM):
    """Merge LoRA weights back into original linear layers and restore original forward."""
    for module in model.modules():
        if hasattr(module, "lora") and isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.data += (module.lora.A @ module.lora.B).T * module.lora.scaling
            module.forward = module._original_forward
            del module.lora
            del module._original_forward
    return model


def lora_state_dict(model: EmindLM) -> dict:
    return {k: v for k, v in model.state_dict().items() if "lora" in k}
