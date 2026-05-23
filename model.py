import random
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
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1, max_seq_len: int = 4096, qk_norm: bool = False, use_kv_quant: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.use_kv_quant = use_kv_quant

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(d_model)
            self.k_norm = RMSNorm(n_kv_heads * self.d_k)
        else:
            self.q_norm = self.k_norm = None

    @staticmethod
    def quantize_kv(k: torch.Tensor, v: torch.Tensor):
        if k.numel() < 256:
            return k, v, None, None
        k_scale = k.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8) / 127.0
        v_scale = v.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8) / 127.0
        k_q = (k.float() / k_scale).round().clamp(-127, 127).to(torch.int8)
        v_q = (v.float() / v_scale).round().clamp(-127, 127).to(torch.int8)
        return k_q, v_q, k_scale, v_scale

    @staticmethod
    def dequantize_kv(k_q, v_q, k_scale, v_scale):
        return k_q.float() * k_scale, v_q.float() * v_scale

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
            if self.use_kv_quant and k_cache.dtype == torch.int8:
                k_cache, v_cache = self.dequantize_kv(k_cache, v_cache, *kv_cache[2:4])
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)

        if self.use_kv_quant and not self.training:
            k_q, v_q, k_s, v_s = self.quantize_kv(k, v)
            new_kv_cache = (k_q, v_q, k_s, v_s)
        else:
            new_kv_cache = (k.detach(), v.detach())

        k = k.repeat_interleave(self.n_rep, dim=2).contiguous()
        v = v.repeat_interleave(self.n_rep, dim=2).contiguous()

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None),
                                              dropout_p=self.dropout if self.training else 0.0)
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
                 n_active_experts: int = 2, moe_aux_loss_coef: float = 0.01,
                 use_kv_quant: bool = False):
        super().__init__()
        self.parallel = parallel_attn_ffn
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout, max_seq_len, qk_norm=qk_norm, use_kv_quant=use_kv_quant)
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
    # KV Cache
    use_kv_quant: bool = False
    # YaRN
    original_max_len: int = 4096

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, f"d_model({self.d_model}) must be divisible by n_heads({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads({self.n_heads}) must be divisible by n_kv_heads({self.n_kv_heads})"

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
                use_kv_quant=config.use_kv_quant,
            )
            for _ in range(config.n_layers)
        ])
        self.ln_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

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
        self.lm_head.weight = self.token_embedding.weight

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
        output_hidden_states: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids)
        cos, sin = self.rope_cos.to(device), self.rope_sin.to(device)
        mask = self.create_causal_mask(seq_len, device)

        hidden_states = [] if output_hidden_states else None
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
            if output_hidden_states:
                hidden_states.append(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=self.config.pad_token_id)

        return loss, logits, new_caches, hidden_states if output_hidden_states else None, aux_loss_total, x

    def apply_neftune_noise(self, noise_alpha: float = 5.0):
        if not self.training:
            return
        eps = noise_alpha / (self.config.d_model ** 0.5)
        with torch.no_grad():
            embed = self.token_embedding
            noise = torch.randn_like(embed.weight) * eps
            embed.weight.add_(noise)

    @torch.no_grad()
    def generate_beam(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        device = input_ids.device
        batch_size = input_ids.shape[0]
        vocab_size = self.config.vocab_size

        cos, sin = self.rope_cos.to(device), self.rope_sin.to(device)
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        done = torch.zeros(batch_size * num_beams, dtype=torch.bool, device=device)
        kv_caches = [None] * len(self.layers)
        gen_ids = input_ids.repeat_interleave(num_beams, dim=0)

        for step in range(max_new_tokens):
            if done.all():
                break

            _, logits, kv_caches, _, _, _ = self(gen_ids, kv_caches=None if step == 0 else kv_caches)
            logits = logits[:, -1, :] / 1.0
            log_probs = F.log_softmax(logits, dim=-1)
            vocab_log_probs = log_probs + beam_scores.unsqueeze(-1)
            vocab_log_probs[done.unsqueeze(-1).expand_as(vocab_log_probs)] = float('-inf')

            if step == 0:
                topk_log_probs, topk_indices = vocab_log_probs[0::num_beams].topk(num_beams * 2, dim=-1)
            else:
                topk_log_probs, topk_indices = vocab_log_probs.view(batch_size, -1).topk(num_beams * 2, dim=-1)

            topk_beam_indices = topk_indices // vocab_size
            topk_token_indices = topk_indices % vocab_size

            new_beam_scores = topk_log_probs.flatten()[:batch_size * num_beams]
            new_token_ids = topk_token_indices.flatten()[:batch_size * num_beams]
            new_beam_indices = topk_beam_indices.flatten()[:batch_size * num_beams]

            gen_ids = gen_ids.view(batch_size, num_beams, -1)
            gen_ids = gen_ids.gather(1, new_beam_indices.view(batch_size, num_beams, 1, 1).expand(-1, -1, -1, gen_ids.shape[-1]))
            gen_ids = gen_ids.view(batch_size * num_beams, -1)
            gen_ids = torch.cat([gen_ids, new_token_ids.view(-1, 1)], dim=-1)

            # KV cache 重排 — 同步 beam 重组
            idx = new_beam_indices  # (batch * num_beams,)
            for li in range(len(kv_caches)):
                if kv_caches[li] is not None:
                    c = kv_caches[li]
                    k, v = c[0][idx], c[1][idx]
                    if len(c) >= 4 and c[2] is not None:
                        kv_caches[li] = (k, v, c[2][idx], c[3][idx])
                    else:
                        kv_caches[li] = (k, v)

            beam_scores = new_beam_scores
            done |= (new_token_ids.view(-1) == eos_token_id)
            if early_stopping and done.any():
                break

        gen_ids = gen_ids.view(batch_size, num_beams, -1)
        beam_scores = beam_scores.view(batch_size, num_beams)
        if length_penalty != 1.0:
            lengths = (gen_ids != self.config.pad_token_id).sum(dim=-1).float()
            beam_scores = beam_scores / (lengths ** length_penalty)
        best_beams = beam_scores.argmax(dim=-1)
        return gen_ids[torch.arange(batch_size, device=device), best_beams]

    @torch.no_grad()
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        draft_layers: int = 2,
        gamma: int = 4,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        device = input_ids.device
        batch_size = input_ids.shape[0]

        def _clone_cache(cache):
            if cache is None:
                return None
            if len(cache) == 2:
                return (cache[0].clone(), cache[1].clone())
            return (cache[0].clone(), cache[1].clone(), cache[2], cache[3])

        def _trim_cache(cache, keep_len):
            if cache is None:
                return None
            k = cache[0][:, :keep_len]
            v = cache[1][:, :keep_len]
            if len(cache) == 2:
                return (k, v)
            return (k, v, cache[2][:, :keep_len] if cache[2] is not None else None,
                    cache[3][:, :keep_len] if cache[3] is not None else None)

        cos, sin = self.rope_cos.to(device), self.rope_sin.to(device)
        gen_ids = input_ids.clone()
        kv_caches = [None] * len(self.layers)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        is_prefill = True

        while gen_ids.shape[1] - input_ids.shape[1] < max_new_tokens:
            if done.all():
                break

            cur_len = gen_ids.shape[1]
            new_token = gen_ids[:, -1:] if cur_len > input_ids.shape[1] else gen_ids

            if is_prefill:
                mask = self.create_causal_mask(cur_len, device)
            else:
                mask = torch.zeros(1, 1, 1, cur_len, device=device)

            cos_step = cos[max(0, cur_len - new_token.shape[1]):cur_len]
            sin_step = sin[max(0, cur_len - new_token.shape[1]):cur_len]

            # 保存原始 cache
            saved_caches = [_clone_cache(c) for c in kv_caches]

            # Draft: 用独立 cache (clone 自原始)
            draft_caches = [_clone_cache(saved_caches[i]) for i in range(draft_layers)]
            x_draft = self.token_embedding(new_token)
            for i in range(draft_layers):
                x_draft, dc, _ = self.layers[i](x_draft, cos_step, sin_step, mask, draft_caches[i])
                draft_caches[i] = dc

            draft_logits = self.lm_head(self.ln_f(x_draft[:, -1:, :])).squeeze(1)
            draft_probs = F.softmax(draft_logits / temperature, dim=-1)
            draft_tokens = [torch.multinomial(draft_probs, num_samples=1)]

            for _ in range(gamma - 1):
                x_d = self.token_embedding(draft_tokens[-1])
                for i in range(draft_layers):
                    x_d, dc, _ = self.layers[i](x_d, cos_step, sin_step, mask, draft_caches[i])
                    draft_caches[i] = dc
                d_logits = self.lm_head(self.ln_f(x_d[:, -1:, :])).squeeze(1)
                d_probs = F.softmax(d_logits / temperature, dim=-1)
                draft_tokens.append(torch.multinomial(d_probs, num_samples=1))

            draft_seq = torch.cat(draft_tokens, dim=-1)

            # 验证: 用原始 cache 的独立副本
            verify_caches = [_clone_cache(saved_caches[i]) for i in range(len(self.layers))]
            x_verify = self.token_embedding(torch.cat([new_token, draft_seq], dim=1))
            for i in range(len(self.layers)):
                x_verify, vc, _ = self.layers[i](x_verify, cos, sin, mask, verify_caches[i])
                verify_caches[i] = vc

            full_logits = self.lm_head(self.ln_f(x_verify))
            full_probs = F.softmax(full_logits[:, :new_token.shape[1]:, :] / temperature, dim=-1)
            draft_probs_base = F.softmax(draft_logits, dim=-1)

            # 逐 token 接受/拒绝
            accepted = 0
            for n in range(gamma):
                if accepted >= gamma:
                    break
                # draft_seq[0, n] 是第 n 个 draft token
                # full_probs[0, n] 是验证模型对该位置的概率分布
                p_m = full_probs[0, n, draft_seq[0, n]].item()
                p_d = draft_probs_base[0, draft_seq[0, n]].item() if n == 0 else d_probs[0, draft_seq[0, n]].item()
                if torch.rand(1, device=device).item() < min(1.0, p_m / max(p_d, 1e-8)):
                    accepted += 1
                else:
                    # 拒绝: 从残差分布采样修正 token
                    p_res = full_probs[0, n] - (draft_probs_base[0] if n == 0 else d_probs[0])
                    p_res = p_res.clamp(min=0)
                    p_res = p_res / (p_res.sum() + 1e-8)
                    correction = torch.multinomial(p_res, num_samples=1).unsqueeze(0)
                    gen_ids = torch.cat([gen_ids, correction], dim=1)
                    break

            # 更新主 cache
            if accepted > 0:
                keep_len = cur_len + accepted
                keep_len = min(keep_len, verify_caches[0][0].shape[1]) if verify_caches[0] is not None else keep_len
                for i in range(len(self.layers)):
                    kv_caches[i] = _trim_cache(verify_caches[i], keep_len)
            elif accepted == 0 and saved_caches[0] is not None:
                for i in range(len(self.layers)):
                    kv_caches[i] = _clone_cache(saved_caches[i])

            gen_ids = torch.cat([gen_ids, draft_seq[:, :accepted]], dim=1)

            is_prefill = False

            for t in draft_seq[0, :accepted]:
                if t == eos_token_id:
                    done[:] = True
                    break
            if done.any():
                break

        return gen_ids

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
        use_dola: bool = False,
        dola_gamma: float = 0.5,
        dola_premature_layer_ratio: float = 0.5,
        min_p: float = 0.0,
    ) -> torch.Tensor:
        self.eval()
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        device = input_ids.device
        batch_size = input_ids.shape[0]

        cos, sin = self.rope_cos.to(device), self.rope_sin.to(device)
        gen_ids = input_ids.clone()

        kv_caches = [None] * len(self.layers)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        n_layers = len(self.layers)
        prem_layer = int(n_layers * dola_premature_layer_ratio) if use_dola else -1
        prem_layer = max(0, min(n_layers - 2, prem_layer))

        for _ in range(max_new_tokens):
            active = ~done
            if not active.any():
                break

            cur_len = gen_ids.shape[1]
            is_prefill = cur_len == input_ids.shape[1]

            if is_prefill:
                if use_dola:
                    _, _, kv_caches, hs, _, _ = self(gen_ids, output_hidden_states=True)
                else:
                    x = self.token_embedding(gen_ids)
                    mask = self.create_causal_mask(cur_len, device)
                    new_caches = []
                    for i, layer in enumerate(self.layers):
                        x, new_cache, _ = layer(x, cos, sin, mask, kv_caches[i])
                        new_caches.append(new_cache)
                    kv_caches = new_caches
            else:
                if use_dola:
                    _, _, kv_caches, hs, _, _ = self(gen_ids[:, -1:], kv_caches=kv_caches, output_hidden_states=True)
                else:
                    x = self.token_embedding(gen_ids[:, -1:])
                    mask = torch.zeros(1, 1, 1, cur_len, device=device)
                    cos_step = cos[cur_len - 1:cur_len]
                    sin_step = sin[cur_len - 1:cur_len]
                    new_caches = []
                    for i, layer in enumerate(self.layers):
                        x, new_cache, _ = layer(x, cos_step, sin_step, mask, kv_caches[i])
                        new_caches.append(new_cache)
                    kv_caches = new_caches

            if use_dola:
                mature_hidden = hs[-1]
                prem_hidden = hs[prem_layer]
                mature_logits = self.lm_head(self.ln_f(mature_hidden[:, -1:, :])).squeeze(1)
                prem_logits = self.lm_head(self.ln_f(prem_hidden[:, -1:, :])).squeeze(1)
                logits = mature_logits - dola_gamma * prem_logits
            else:
                x = self.ln_f(x)
                logits = self.lm_head(x[:, -1:, :]).squeeze(1)

            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    gen_tokens = gen_ids[b, input_ids.shape[1]:]
                    for tid in torch.unique(gen_tokens):
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
                sorted_mask = cum_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                sorted_logits[sorted_mask] = float('-inf')
                logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

            if min_p > 0:
                probs = F.softmax(logits, dim=-1)
                threshold = probs.max(dim=-1, keepdim=True)[0] * min_p
                logits[probs < threshold] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            gen_ids = torch.cat([gen_ids, next_token], dim=1)

            done |= (next_token.squeeze(-1) == eos_token_id)

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
