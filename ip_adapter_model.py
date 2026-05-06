"""IP-Adapter model for Anima — v3 with MLP Token Projector.

v3: Replaces Perceiver Resampler with pure feedforward MLP projector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLPTokenProjector(nn.Module):
    """Pure feedforward: 1024 → hidden → num_tokens * 1024. No collapse risk."""

    def __init__(self, input_dim=1024, output_dim=1024, num_tokens=8, hidden_mult=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        hidden_dim = num_tokens * output_dim * hidden_mult
        self.hidden_per_token = hidden_dim // num_tokens

        self.proj_up = nn.Linear(input_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(self.hidden_per_token)
        self.proj_down = nn.Linear(self.hidden_per_token, output_dim, bias=False)
        self.norm_out = nn.LayerNorm(output_dim)

        nn.init.normal_(self.proj_up.weight, std=0.02)
        nn.init.normal_(self.proj_down.weight, std=0.02)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        h = self.proj_up(x)
        h = h.view(x.shape[0], self.num_tokens, self.hidden_per_token)
        h = self.norm(h)
        h = F.gelu(h)
        tokens = self.proj_down(h)
        tokens = self.norm_out(tokens)
        return tokens


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
        nn.init.normal_(self.output_proj.weight, std=0.01)
        nn.init.normal_(self.q_proj.weight, std=0.01)
        nn.init.normal_(self.k_proj.weight, std=0.01)
        nn.init.normal_(self.v_proj.weight, std=0.01)

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
        mlp_hidden_mult=4,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.token_projector = MLPTokenProjector(
            input_dim=emb_dim,
            output_dim=emb_dim,
            num_tokens=num_tokens,
            hidden_mult=mlp_hidden_mult,
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
            nn.Parameter(torch.full((1,), 0.01))
            for _ in range(num_blocks)
        ])

    def resample(self, image_emb):
        return self.token_projector(image_emb)

    def forward_block(self, block_idx, normalized_x, image_tokens):
        ip_out = self.ip_cross_attns[block_idx](normalized_x, image_tokens)
        return self.ip_scales[block_idx] * ip_out
