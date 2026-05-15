"""IP-Adapter for Anima DiT — v3 InstantCharacter-style.

Architecture per arXiv:2504.12395:
1. General vision encoder: SigLIP2 (deep + shallow features)
2. Intermediate encoder: cross-layer transformer (shallow→deep fusion)
3. Projection head: TimeResampler (timestep-aware Perceiver with AdaLN)
4. Per DiT block: IPCrossAttn with RMSNorm on Q/K
5. Injection: text_attn_out + scale * ip_attn_out
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


# ─── Intermediate Encoder (Cross-Layer Fusion) ──────────────────────

class CrossLayerEncoder(nn.Module):
    """Fuses shallow + deep features from vision encoder via cross-attention.

    Per InstantCharacter paper §3.1: "each feature pathway is independently
    processed by a separate transformer encoder to integrate with high-level
    semantic features."
    """
    def __init__(self, shallow_dim, deep_dim, hidden_dim, num_layers=4, num_heads=16):
        super().__init__()
        # Project both streams to hidden_dim
        self.shallow_proj = nn.Linear(shallow_dim, hidden_dim, bias=False)
        self.deep_proj = nn.Linear(deep_dim, hidden_dim, bias=False)
        self.norm_shallow = RMSNorm(hidden_dim)
        self.norm_deep = RMSNorm(hidden_dim)

        # Cross-layer transformer: shallow features attend to deep features
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.cross_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])

    def forward(self, shallow_features, deep_features):
        """shallow: [B, N1, C1], deep: [B, N2, C2] → [B, N1+N2, hidden_dim]"""
        s = self.norm_shallow(self.shallow_proj(shallow_features))
        d = self.norm_deep(self.deep_proj(deep_features))
        # Cross-attention: shallow queries attend to deep keys/values
        for layer in self.cross_layers:
            s = layer(s, memory=d)
        return torch.cat([s, d], dim=1)


# ─── TimeResampler (Projection Head) ────────────────────────────────

class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, shift=None, scale=None):
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
    """Timestep-aware Q-Former (Projection Head per §3.1).

    Processes intermediate encoder outputs as KV, learnable queries via
    Perceiver attention with AdaLN timestep conditioning.
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

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
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
                nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True)),
            ]))

    def _embed_timestep(self, t):
        half_dim = self.time_proj.out_features // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.cos(), args.sin()], dim=-1)
        if emb.shape[-1] < self.time_proj.out_features:
            emb = F.pad(emb, (0, self.time_proj.out_features - emb.shape[-1]))
        return self.time_mlp(emb)

    def forward(self, x, timestep):
        B = x.shape[0]
        x = self.proj_in(x)
        latents = self.latents.expand(B, -1, -1)
        temb = self._embed_timestep(timestep)

        for attn, ff, adaLN_mod in self.layers:
            shift_msa, scale_msa, shift_ff, scale_ff = adaLN_mod(temb).chunk(4, dim=1)
            latents = attn(x, latents, shift=shift_msa, scale=scale_msa) + latents
            res = latents
            for i, layer in enumerate(ff):
                latents = layer(latents)
                if i == 0 and isinstance(layer, nn.LayerNorm):
                    latents = latents * (1 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
            latents = latents + res

        return self.norm_out(self.proj_out(latents))


# ─── Per-Block IP Cross-Attention ───────────────────────────────────

class IPCrossAttn(nn.Module):
    """Per-block IP attention with RMSNorm on Q and K (InstantCharacter-style)."""
    def __init__(self, hidden_size=2048, ip_hidden_dim=2048, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm_ip_q = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_ip_k = RMSNorm(self.head_dim, eps=1e-6)
        self.to_k_ip = nn.Linear(ip_hidden_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_dim, hidden_size, bias=False)

    def forward(self, q_text, ip_hidden_states):
        B, S, D = q_text.shape
        L = ip_hidden_states.shape[1]
        q = q_text.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.norm_ip_q(q)
        ip_k = self.to_k_ip(ip_hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        ip_k = self.norm_ip_k(ip_k)
        ip_v = self.to_v_ip(ip_hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, ip_k, ip_v)
        out = out.transpose(1, 2).reshape(B, S, D)
        return out


# ─── IP-Adapter Module ──────────────────────────────────────────────

class IPAdapterSigLIP(nn.Module):
    """IP-Adapter with CrossLayerEncoder + TimeResampler + IPCrossAttn.

    Per InstantCharacter (arXiv:2504.12395):
    1. General vision encoder: SigLIP2 deep + shallow features
    2. Intermediate encoder: CrossLayerEncoder (shallow→deep cross-attention fusion)
    3. Projection head: TimeResampler (timestep-aware Perceiver with AdaLN)
    4. Per-block IPCrossAttn with RMSNorm
    """
    def __init__(self, siglip_dim=768, siglip_shallow_dim=768,
                 dit_dim=2048, num_blocks=28,
                 num_queries=32, resampler_depth=4, resampler_heads=16,
                 resampler_dim=1024, resampler_dim_head=64,
                 intermediate_dim=768, intermediate_layers=4, intermediate_heads=12,
                 ip_heads=16, time_embed_dim=320,
                 use_intermediate_encoder=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_queries = num_queries
        self.use_intermediate_encoder = use_intermediate_encoder

        if use_intermediate_encoder:
            # Intermediate encoder: fuse shallow + deep SigLIP features
            self.intermediate_encoder = CrossLayerEncoder(
                shallow_dim=siglip_shallow_dim, deep_dim=siglip_dim,
                hidden_dim=intermediate_dim, num_layers=intermediate_layers,
                num_heads=intermediate_heads,
            )
            resampler_input_dim = intermediate_dim * 2  # cat(shallow, deep) after projection
        else:
            self.intermediate_encoder = None
            resampler_input_dim = siglip_dim

        # Projection head: TimeResampler
        self.resampler = TimeResampler(
            input_dim=resampler_input_dim, dim=resampler_dim,
            output_dim=dit_dim, num_queries=num_queries,
            depth=resampler_depth, dim_head=resampler_dim_head,
            heads=resampler_heads, ff_mult=4,
            time_embed_dim=time_embed_dim,
        )

        # Per-block IP cross-attention
        self.ip_cross_attns = nn.ModuleList([
            IPCrossAttn(hidden_size=dit_dim, ip_hidden_dim=dit_dim, num_heads=ip_heads)
            for _ in range(num_blocks)
        ])

        # Per-block scale
        self.ip_scales = nn.ParameterList([
            nn.Parameter(torch.full((1,), 0.01))
            for _ in range(num_blocks)
        ])

    def encode_ref(self, siglip_features, timestep=None, shallow_features=None):
        """SigLIP features → image tokens.

        Args:
            siglip_features: deep features [B, N_deep, 768]
            timestep: [B] in [0,1]
            shallow_features: shallow features [B, N_shallow, 768] (optional)
        """
        if timestep is None:
            B = siglip_features.shape[0]
            timestep = torch.full((B,), 0.5, device=siglip_features.device, dtype=siglip_features.dtype)

        if self.use_intermediate_encoder and shallow_features is not None:
            x = self.intermediate_encoder(shallow_features, siglip_features)
        else:
            x = siglip_features

        return self.resampler(x, timestep)

    def forward_block(self, block_idx, query, image_tokens, scale_override=None):
        ip_out = self.ip_cross_attns[block_idx](query, image_tokens)
        scale = scale_override if scale_override is not None else self.ip_scales[block_idx]
        return scale * ip_out
