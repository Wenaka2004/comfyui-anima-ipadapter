"""IP-Adapter for Anima DiT with SigLIP2 image encoder.

Architecture inspired by InstantCharacter:
1. SigLIP2 extracts patch features [B, N_patches, 768]
2. TimeResampler (Perceiver + timestep AdaLN): 768 → 2048-dim tokens
3. Per DiT block: to_k_ip/to_v_ip with RMSNorm, reuse text cross-attn query
4. Injection: text_attn_output + scale * ip_attn_output (no magnitude hack)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x):
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * self.dim ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed


# ─── TimeResampler ──────────────────────────────────────────────────

class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, shift=None, scale=None):
        """x: image features [B, N, dim], latents: [B, Nq, dim]"""
        x = self.norm_kv(x)
        latents = self.norm_q(latents)

        if shift is not None and scale is not None:
            latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        B, L, _ = latents.shape
        q = self.to_q(latents)
        kv_input = torch.cat([x, latents], dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = q.view(B, L, self.heads, -1).transpose(1, 2)
        k = k.view(B, k.shape[1], self.heads, -1).transpose(1, 2)
        v = v.view(B, v.shape[1], self.heads, -1).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.to_out(out)


class TimeResampler(nn.Module):
    """Perceiver resampler with timestep conditioning via AdaLN.

    Key difference from vanilla Perceiver: timestep embedding modulates
    the latent tokens via shift/scale in each layer, so the output
    image tokens adapt to the current denoising step.
    """
    def __init__(self, input_dim=768, dim=1024, output_dim=2048,
                 num_queries=32, depth=4, dim_head=64, heads=16,
                 ff_mult=4, time_embed_dim=320):
        super().__init__()
        self.num_queries = num_queries
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
        self.proj_in = nn.Linear(input_dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, output_dim, bias=False)
        self.norm_out = nn.LayerNorm(output_dim)

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        # Sinusoidal timestep projection
        self.time_proj = nn.Linear(1, time_embed_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * ff_mult, bias=False),
                    nn.GELU(),
                    nn.Linear(dim * ff_mult, dim, bias=False),
                ),
                # AdaLN modulation: shift_msa, scale_msa, shift_ff, scale_ff
                nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True)),
            ]))

    def _embed_timestep(self, t):
        """t: [B] timestep values in [0, 1]"""
        # Simple sinusoidal + MLP
        half_dim = self.time_proj.out_features // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if emb.shape[-1] < self.time_proj.out_features:
            emb = F.pad(emb, (0, self.time_proj.out_features - emb.shape[-1]))
        return self.time_mlp(emb)

    def forward(self, x, timestep):
        """x: [B, N, input_dim], timestep: [B] in [0,1] → [B, Nq, output_dim]"""
        B = x.shape[0]
        x = self.proj_in(x)
        latents = self.latents.expand(B, -1, -1)

        temb = self._embed_timestep(timestep)

        for attn, ff, adaLN_mod in self.layers:
            shift_msa, scale_msa, shift_ff, scale_ff = adaLN_mod(temb).chunk(4, dim=1)
            # Perceiver attention with AdaLN on latents
            latents = attn(x, latents, shift=shift_msa, scale=scale_msa) + latents
            # FFN with AdaLN (matching InstantCharacter)
            res = latents
            for i, layer in enumerate(ff):
                latents = layer(latents)
                if i == 0 and isinstance(layer, nn.LayerNorm):
                    latents = latents * (1 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
            latents = latents + res

        return self.norm_out(self.proj_out(latents))


# ─── Per-Block IP Cross-Attention ───────────────────────────────────

class IPCrossAttn(nn.Module):
    """Per-block IP attention with RMSNorm on Q and K (InstantCharacter-style).

    Key differences from our old design:
    - RMSNorm on ip_q and ip_k for stable attention computation
    - Full-dim to_k_ip / to_v_ip (no bottleneck)
    - No output_proj (just add directly)
    - No magnitude normalization needed — RMSNorm handles scale matching
    """
    def __init__(self, hidden_size=2048, ip_hidden_dim=2048, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 128

        self.norm_ip_q = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_ip_k = RMSNorm(self.head_dim, eps=1e-6)

        self.to_k_ip = nn.Linear(ip_hidden_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_dim, hidden_size, bias=False)

    def forward(self, q_text, ip_hidden_states):
        """q_text: [B, S, hidden_size], ip_hidden_states: [B, Nq, ip_hidden_dim]"""
        B, S, D = q_text.shape
        L = ip_hidden_states.shape[1]

        # Q from text cross-attn query (reuse, like InstantCharacter)
        q = q_text.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.norm_ip_q(q)

        # K, V from IP hidden states
        ip_k = self.to_k_ip(ip_hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        ip_k = self.norm_ip_k(ip_k)
        ip_v = self.to_v_ip(ip_hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, ip_k, ip_v)
        out = out.transpose(1, 2).reshape(B, S, D)
        return out


# ─── IP-Adapter Module ──────────────────────────────────────────────

class IPAdapterSigLIP(nn.Module):
    """IP-Adapter with SigLIP2 + TimeResampler + per-block IP cross-attn.

    Architecture (InstantCharacter-inspired):
    - TimeResampler: SigLIP features → timestep-aware image tokens
    - Per-block IPCrossAttn: RMSNorm-stabilized attention, no bottleneck
    - Injection: text_attn_out + scale * ip_attn_out
    """
    def __init__(self, input_dim=768, dit_dim=2048, num_blocks=28,
                 num_queries=32, resampler_depth=4, resampler_heads=16,
                 resampler_dim=1024, resampler_dim_head=64,
                 ip_heads=16, time_embed_dim=320):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_queries = num_queries

        # TimeResampler
        self.resampler = TimeResampler(
            input_dim=input_dim, dim=resampler_dim,
            output_dim=dit_dim, num_queries=num_queries,
            depth=resampler_depth, dim_head=resampler_dim_head,
            heads=resampler_heads, ff_mult=4,
            time_embed_dim=time_embed_dim,
        )

        # Per-block IP cross-attention (full-dim, with RMSNorm)
        self.ip_cross_attns = nn.ModuleList([
            IPCrossAttn(hidden_size=dit_dim, ip_hidden_dim=dit_dim,
                        num_heads=ip_heads)
            for _ in range(num_blocks)
        ])

        # Per-block scale
        self.ip_scales = nn.ParameterList([
            nn.Parameter(torch.full((1,), 0.01))
            for _ in range(num_blocks)
        ])

    def encode_ref(self, siglip_features, timestep=None):
        """SigLIP features → image tokens. timestep: [B] in [0,1]."""
        if timestep is None:
            # Default to t=0.5 if not provided (midpoint)
            B = siglip_features.shape[0]
            timestep = torch.full((B,), 0.5, device=siglip_features.device, dtype=siglip_features.dtype)
        return self.resampler(siglip_features, timestep)

    def forward_block(self, block_idx, query, image_tokens, scale_override=None):
        """IP cross-attention for one DiT block."""
        ip_out = self.ip_cross_attns[block_idx](query, image_tokens)
        scale = scale_override if scale_override is not None else self.ip_scales[block_idx]
        return scale * ip_out
