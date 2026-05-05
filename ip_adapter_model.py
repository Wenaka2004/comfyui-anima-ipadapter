"""IP-Adapter model for Anima — v2 with improved resampler.

v2 fixes:
1. Input projection: 1x1024 token → 8x1024 tokens before perceiver
2. Cross-attn gate in PerceiverLayer (learnable scalar, init=2.0)
3. Fewer latents: 8 instead of 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class InputProjection(nn.Module):
    """Project single 1024-dim token into multiple tokens for perceiver."""

    def __init__(self, emb_dim=1024, num_input_tokens=8):
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.proj = nn.Linear(emb_dim, emb_dim * num_input_tokens, bias=False)

    def forward(self, x):
        B = x.shape[0]
        out = self.proj(x)
        out = out.reshape(B, self.num_input_tokens, -1)
        return out


class PerceiverLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, ff_mult):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(emb_dim)
        self.self_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(emb_dim)
        self.cross_attn_norm_kv = nn.LayerNorm(emb_dim)
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.cross_attn_gate = nn.Parameter(torch.tensor(2.0))
        self.ff_norm = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * ff_mult),
            nn.GELU(),
            nn.Linear(emb_dim * ff_mult, emb_dim),
        )

    def forward(self, x, context):
        residual = x
        x_norm = self.self_attn_norm(x)
        x_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = residual + x_attn
        residual = x
        x_norm = self.cross_attn_norm(x)
        ctx_norm = self.cross_attn_norm_kv(context)
        x_attn, _ = self.cross_attn(x_norm, ctx_norm, ctx_norm)
        x = residual + self.cross_attn_gate * x_attn
        residual = x
        x = residual + self.ff(self.ff_norm(x))
        return x


class PerceiverResampler(nn.Module):
    def __init__(self, emb_dim=1024, num_tokens=8, num_input_tokens=8, num_layers=2, num_heads=4, ff_mult=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.input_proj = InputProjection(emb_dim, num_input_tokens)
        self.latents = nn.Parameter(torch.randn(num_tokens, emb_dim) * 0.02)
        self.layers = nn.ModuleList([
            PerceiverLayer(emb_dim, num_heads, ff_mult) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, image_emb):
        B = image_emb.shape[0]
        context = self.input_proj(image_emb)
        x = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            x = layer(x, context)
        return self.norm(x)


class IPAdapterCrossAttention(nn.Module):
    def __init__(self, x_dim=2048, context_dim=1024, ip_dim=512, num_heads=8, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(x_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, x_dim, bias=False)
        nn.init.zeros_(self.output_proj.weight)

    def forward(self, x, context):
        B, S, _ = x.shape
        L = context.shape[1]
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(context).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(context).view(B, L, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.output_proj(out)


class IPAdapter(nn.Module):
    def __init__(
        self,
        emb_dim=1024,
        x_dim=2048,
        num_blocks=28,
        ip_dim=512,
        num_ip_heads=8,
        ip_head_dim=64,
        num_tokens=8,
        num_perceiver_layers=2,
        num_perceiver_heads=4,
        num_input_tokens=8,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.resampler = PerceiverResampler(
            emb_dim=emb_dim,
            num_tokens=num_tokens,
            num_input_tokens=num_input_tokens,
            num_layers=num_perceiver_layers,
            num_heads=num_perceiver_heads,
        )
        self.ip_cross_attns = nn.ModuleList([
            IPAdapterCrossAttention(
                x_dim=x_dim,
                context_dim=emb_dim,
                ip_dim=ip_dim,
                num_heads=num_ip_heads,
                head_dim=ip_head_dim,
            )
            for _ in range(num_blocks)
        ])
        self.ip_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1))
            for _ in range(num_blocks)
        ])

    def resample(self, image_emb):
        return self.resampler(image_emb)

    def forward_block(self, block_idx, normalized_x, image_tokens):
        ip_out = self.ip_cross_attns[block_idx](normalized_x, image_tokens)
        return self.ip_scales[block_idx] * ip_out
