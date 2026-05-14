"""IP-Adapter for Anima DiT with SigLIP2 image encoder.

Architecture (ZImage/SD3-style for DiT):
1. SigLIP2 extracts patch features [B, N_patches, 768] from ref image
2. Resampler (4-layer Perceiver, 64 queries): 768 → 2048-dim tokens
3. Per DiT block: to_k/to_v projections, reuse text cross-attn query
4. Injection: gate * (text_attn + λ * ip_attn) + residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Resampler ──────────────────────────────────────────────────────

class ResamplerLayer(nn.Module):
    """One layer: self-attn on queries + cross-attn to image features + FF."""

    def __init__(self, dim=2048, num_heads=16, ff_mult=4):
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult, bias=False),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim, bias=False),
        )

    def forward(self, q, kv):
        # q: [B, 64, dim], kv: [B, N_patches, dim]
        q = q + self.self_attn(self.norm_self(q), self.norm_self(q), self.norm_self(q))[0]
        q = q + self.cross_attn(self.norm_cross(q), self.norm_kv(kv), self.norm_kv(kv))[0]
        q = q + self.ff(self.norm_ff(q))
        return q


class Resampler(nn.Module):
    """Projects image patch features → fixed number of tokens for DiT.

    Lightweight: internal_dim=1024, 2 layers, 32 queries (~40M instead of ~190M).
    """

    def __init__(self, input_dim=768, internal_dim=1024, output_dim=2048,
                 num_queries=32, depth=2, num_heads=16, ff_mult=4):
        super().__init__()
        self.num_queries = num_queries
        self.input_proj = nn.Linear(input_dim, internal_dim, bias=False)
        self.output_proj = nn.Linear(internal_dim, output_dim, bias=False)
        self.latents = nn.Parameter(torch.randn(num_queries, internal_dim) * 0.02)
        self.layers = nn.ModuleList([
            ResamplerLayer(internal_dim, num_heads, ff_mult) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        """x: [B, N_patches, input_dim] → [B, num_queries, output_dim]"""
        B = x.shape[0]
        x = self.input_proj(x)  # [B, N, internal_dim]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, Nq, internal_dim]
        for layer in self.layers:
            q = layer(q, x)
        return self.norm(self.output_proj(q))  # [B, Nq, output_dim]


# ─── Per-Block IP Cross-Attention ───────────────────────────────────

class IPCrossAttn(nn.Module):
    """Per-block: to_k/to_v with bottleneck (2048→256→2048, ~1M per block)."""

    def __init__(self, dim=2048, context_dim=2048, num_heads=16, bottleneck=256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 128
        self.to_k_down = nn.Linear(context_dim, bottleneck, bias=False)
        self.to_k_up = nn.Linear(bottleneck, dim, bias=False)
        self.to_v_down = nn.Linear(context_dim, bottleneck, bias=False)
        self.to_v_up = nn.Linear(bottleneck, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)
        # Zero-init output_proj for stable start, small init for others
        nn.init.zeros_(self.output_proj.weight)
        for m in [self.to_k_down, self.to_k_up, self.to_v_down, self.to_v_up]:
            nn.init.normal_(m.weight, std=0.01)

    def forward(self, q_text, image_tokens):
        """q_text: [B, S, dim], image_tokens: [B, Nq, dim] → [B, S, dim]"""
        B, S, D = q_text.shape
        L = image_tokens.shape[1]

        q = q_text.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k_up(self.to_k_down(image_tokens)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v_up(self.to_v_down(image_tokens)).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.output_proj(out)


# ─── IP-Adapter Module ──────────────────────────────────────────────

class IPAdapterSigLIP(nn.Module):
    """IP-Adapter with SigLIP2 image encoder + resampler + per-block IP cross-attn.

    Trainable: resampler + ip_cross_attns + ip_scales (~74M).
    Frozen: SigLIP2 + Anima DiT.
    """

    def __init__(self, input_dim=768, dit_dim=2048, num_blocks=28,
                 num_queries=32, resampler_depth=2, resampler_heads=16,
                 resampler_internal=1024, ip_heads=16, ip_bottleneck=256):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_queries = num_queries

        # Resampler: SigLIP features → tokens
        self.resampler = Resampler(
            input_dim=input_dim, internal_dim=resampler_internal,
            output_dim=dit_dim, num_queries=num_queries,
            depth=resampler_depth, num_heads=resampler_heads,
        )

        # Per-block IP cross-attention (bottlenecked to_k/to_v)
        self.ip_cross_attns = nn.ModuleList([
            IPCrossAttn(dim=dit_dim, context_dim=dit_dim,
                        num_heads=ip_heads, bottleneck=ip_bottleneck)
            for _ in range(num_blocks)
        ])

        # Per-block scale (lambda), init small
        self.ip_scales = nn.ParameterList([
            nn.Parameter(torch.full((1,), 0.0))
            for _ in range(num_blocks)
        ])

    def encode_ref(self, siglip_features):
        """SigLIP patch features → image tokens for DiT."""
        return self.resampler(siglip_features)  # [B, 64, 2048]

    def forward_block(self, block_idx, query, image_tokens):
        """IP cross-attention for one DiT block."""
        ip_out = self.ip_cross_attns[block_idx](query, image_tokens)
        return self.ip_scales[block_idx] * ip_out
