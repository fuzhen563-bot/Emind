import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass, field


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================

def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0,
                          device: Optional[torch.device] = None,
                          scaling_factor: float = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    if scaling_factor != 1.0:
        freqs = freqs / scaling_factor
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def precompute_rope_freqs_yarn(dim: int, max_len: int, original_max_len: int = 4096,
                               theta: float = 10000.0, device: Optional[torch.device] = None,
                               scaling_factor: float = 32.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    wavelengths = 2 * math.pi / (freqs + 1e-8)

    ratio = scaling_factor
    low_wavelen = original_max_len
    high_wavelen = original_max_len / ratio

    scales = torch.ones_like(freqs)
    scales = torch.where(wavelengths > low_wavelen,
                         torch.full_like(scales, ratio), scales)
    mid_mask = (wavelengths <= low_wavelen) & (wavelengths >= high_wavelen)
    if mid_mask.any():
        t = (wavelengths[mid_mask] / low_wavelen - 1.0 / ratio) / (1.0 - 1.0 / ratio)
        scales[mid_mask] = 1.0 + t * (ratio - 1.0)

    freqs = freqs / scales
    t = torch.arange(int(max_len), device=device).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dim = x.shape[-1]
    x_half = x.float().reshape(*x.shape[:-1], -1, 2)
    x_rot = torch.stack([-x_half[..., 1], x_half[..., 0]], dim=-1).reshape_as(x)
    seq_len = x.shape[1]
    cos = cos[:seq_len, :dim].view(1, seq_len, 1, dim)
    sin = sin[:seq_len, :dim].view(1, seq_len, 1, dim)
    return (x.float() * cos + x_rot * sin).to(x.dtype)


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# =============================================================================
# Grouped Query Attention (GQA) with KV Cache
# =============================================================================

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1, max_seq_len: int = 4096, qk_norm: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(d_model)
            self.k_norm = RMSNorm(n_kv_heads * self.d_k)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k)

        if self.q_norm is not None:
            q = self.q_norm(q.view(batch_size, seq_len, -1)).view_as(q)
            k = self.k_norm(k.view(batch_size, seq_len, -1)).view_as(k)

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        new_kv_cache = (k.detach(), v.detach())

        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.W_o(out)
        return out, new_kv_cache


# =============================================================================
# SwiGLU Feed-Forward
# =============================================================================

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))


# =============================================================================
# MoE FFN (Mixture of Experts with SwiGLU)
# =============================================================================

class MoEFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8, n_active_experts: int = 2,
                 dropout: float = 0.1, aux_loss_coef: float = 0.01):
        super().__init__()
        self.n_experts = n_experts
        self.n_active_experts = n_active_experts
        self.aux_loss_coef = aux_loss_coef
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff, dropout) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, s, d = x.shape
        x_flat = x.view(-1, d)
        n_tokens = x_flat.shape[0]

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.n_active_experts, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        out = torch.zeros_like(x_flat)
        for i in range(self.n_experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = self.experts[i](x_flat[mask])
                weights = top_k_probs[top_k_indices == i].to(expert_out.dtype).unsqueeze(-1)
                out[mask] += expert_out * weights

        # Load balancing auxiliary loss
        tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
        tokens_per_expert.scatter_add_(0, top_k_indices.view(-1),
                                       torch.ones(n_tokens * self.n_active_experts, device=x.device))
        frac = tokens_per_expert / (n_tokens * self.n_active_experts + 1e-8)
        mean_prob = router_probs.mean(dim=0)
        aux_loss = self.n_experts * (frac * mean_prob).sum()
        aux_loss = aux_loss * self.aux_loss_coef

        return out.view(b, s, d), aux_loss


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int,
                 dropout: float = 0.1, max_seq_len: int = 4096,
                 qk_norm: bool = False, parallel_attn_ffn: bool = False,
                 use_moe: bool = False, n_experts: int = 8,
                 n_active_experts: int = 2, moe_aux_loss_coef: float = 0.01):
        super().__init__()
        self.parallel = parallel_attn_ffn
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout, max_seq_len, qk_norm=qk_norm)
        if use_moe:
            self.feed_forward = MoEFFN(d_model, d_ff, n_experts, n_active_experts, dropout, moe_aux_loss_coef)
        else:
            self.feed_forward = SwiGLU(d_model, d_ff, dropout)
        self.use_moe = use_moe
        self.ln1 = RMSNorm(d_model)
        if not parallel_attn_ffn:
            self.ln2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        if self.parallel:
            h = self.ln1(x)
            attn_out, new_cache = self.attention(h, cos, sin, mask, kv_cache)
            if self.use_moe:
                ff_out, aux_loss = self.feed_forward(h)
            else:
                ff_out = self.feed_forward(h)
                aux_loss = torch.tensor(0.0, device=x.device)
            x = x + self.dropout1(attn_out) + self.dropout2(ff_out)
        else:
            attn_out, new_cache = self.attention(self.ln1(x), cos, sin, mask, kv_cache)
            x = x + self.dropout1(attn_out)
            h = self.ln2(x)
            if self.use_moe:
                ff_out, aux_loss = self.feed_forward(h)
            else:
                ff_out = self.feed_forward(h)
                aux_loss = torch.tensor(0.0, device=x.device)
            x = x + self.dropout2(ff_out)
        return x, new_cache, aux_loss


# =============================================================================
# Emind LM Model
# =============================================================================

@dataclass
class EmindConfig:
    vocab_size: int = 32000
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    n_layers: int = 32
    d_ff: int = 11008
    max_seq_len: int = 4096
    dropout: float = 0.0
    rope_theta: float = 10000.0
    rope_scaling_type: Optional[str] = None  # "ntk", "linear", "yarn", or None
    rope_scaling_factor: float = 1.0        # e.g. 32 for 4K→128K
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1
    activation_checkpointing: bool = False
    qk_norm: bool = False
    parallel_attn_ffn: bool = False
    # MoE
    use_moe: bool = False
    n_experts: int = 8
    n_active_experts: int = 2
    moe_aux_loss_coef: float = 0.01
    # YaRN
    original_max_len: int = 4096

    @classmethod
    def from_dict(cls, d: dict) -> "EmindConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


class EmindLM(nn.Module):
    def __init__(self, config: EmindConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.n_kv_heads, config.d_ff,
                config.dropout, config.max_seq_len, qk_norm=config.qk_norm,
                parallel_attn_ffn=config.parallel_attn_ffn,
                use_moe=config.use_moe, n_experts=config.n_experts,
                n_active_experts=config.n_active_experts,
                moe_aux_loss_coef=config.moe_aux_loss_coef,
            )
            for _ in range(config.n_layers)
        ])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        scale = config.rope_scaling_factor if config.rope_scaling_type else 1.0
        if config.rope_scaling_type == "yarn":
            cos, sin = precompute_rope_freqs_yarn(
                config.d_model // config.n_heads,
                config.max_seq_len,
                original_max_len=config.original_max_len,
                theta=config.rope_theta,
                scaling_factor=scale,
            )
        else:
            cos, sin = precompute_rope_freqs(
                config.d_model // config.n_heads,
                config.max_seq_len,
                config.rope_theta,
                scaling_factor=scale,
            )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids)
        cos, sin = self.rope_cos.to(device), self.rope_sin.to(device)
        mask = self.create_causal_mask(seq_len, device)

        new_caches = []
        use_ckpt = self.training and self.config.activation_checkpointing
        aux_loss_total = torch.tensor(0.0, device=device)
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            if use_ckpt and cache is None:
                x, new_cache, aux_loss = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, mask, None, use_reentrant=False
                )
            else:
                x, new_cache, aux_loss = layer(x, cos, sin, mask, cache)
            new_caches.append(new_cache)
            aux_loss_total = aux_loss_total + aux_loss

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=self.config.pad_token_id)
            loss = loss + aux_loss_total

        return loss, logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        device = input_ids.device
        batch_size = input_ids.shape[0]

        cos, sin = self.rope_cos.to(device), self.rope_sin.to(device)
        gen_ids = input_ids.clone()

        kv_caches = [None] * len(self.layers)

        for _ in range(max_new_tokens):
            cur_len = gen_ids.shape[1]
            if cur_len == input_ids.shape[1]:
                x = self.token_embedding(gen_ids)
                mask = self.create_causal_mask(cur_len, device)
                new_caches = []
                for i, layer in enumerate(self.layers):
                    x, new_cache, _ = layer(x, cos, sin, mask, kv_caches[i])
                    new_caches.append(new_cache)
                kv_caches = new_caches
            else:
                x = self.token_embedding(gen_ids[:, -1:])
                mask = torch.ones(1, 1, 1, 1, device=device)
                cos_step = cos[cur_len - 1:cur_len]
                sin_step = sin[cur_len - 1:cur_len]
                new_caches = []
                for i, layer in enumerate(self.layers):
                    x, new_cache, _ = layer(x, cos_step, sin_step, mask, kv_caches[i])
                    new_caches.append(new_cache)
                kv_caches = new_caches

            x = self.ln_f(x)
            logits = self.lm_head(x[:, -1:, :]).squeeze(1)

            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    for tid in torch.unique(gen_ids[b]):
                        if tid < logits.shape[-1]:
                            if logits[b, tid] < 0:
                                logits[b, tid] *= repetition_penalty
                            else:
                                logits[b, tid] /= repetition_penalty

            logits = logits / temperature

            if top_k is not None and top_k > 0:
                v = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1:]
                logits[logits < v] = float('-inf')

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            gen_ids = torch.cat([gen_ids, next_token], dim=1)

            if (next_token == eos_token_id).any():
                break

        return gen_ids


def create_model(config: Union[dict, EmindConfig]) -> EmindLM:
    if isinstance(config, dict):
        config = EmindConfig.from_dict(config)
    return EmindLM(config)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("Testing Emind model with modern architecture...")
    cfg = EmindConfig(vocab_size=32000, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6, d_ff=2048, max_seq_len=512)
    model = EmindLM(cfg)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    ids = torch.randint(0, cfg.vocab_size, (2, 64))
    loss, logits, _ = model(ids, labels=ids)
    print(f"Loss: {loss.item():.4f}, Logits: {logits.shape}")

    out = model.generate(torch.tensor([[1, 100, 200, 300]]), max_new_tokens=20)
    print(f"Generated: {out.shape}")

    print("All tests passed!")
