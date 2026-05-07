"""IP-Adapter model for Anima — v4 with AdaLN modulation.

Per-block: VL emb → MLP → per-channel scale/shift, applied after block MLP.
No pseudo-tokens, no cross-attention — global vector → global modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockModulation(nn.Module):
    """Maps VL embedding to per-channel scale/shift for one DiT block."""

    def __init__(self, vl_dim=1024, hidden=256, block_dim=2048):
        super().__init__()
        self.proj = nn.Linear(vl_dim, hidden, bias=False)
        self.mod = nn.Linear(hidden, 2 * block_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.normal_(self.mod.weight, std=0.001)

    def forward(self, vl_emb):
        h = F.silu(self.proj(vl_emb))
        out = self.mod(h)
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift


class IPAdapter(nn.Module):
    """IP-Adapter v4: AdaLN modulation from VL embedding.

    For each of 28 DiT blocks:
        x = x * (1 + ip_scale * scale) + ip_scale * shift
    """

    def __init__(
        self,
        vl_dim=768,
        block_dim=2048,
        num_blocks=28,
        modulation_hidden=256,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_mods = nn.ModuleList([
            BlockModulation(vl_dim=vl_dim, hidden=modulation_hidden, block_dim=block_dim)
            for _ in range(num_blocks)
        ])
        self.ip_scales = nn.ParameterList([
            nn.Parameter(torch.full((1,), 0.01))
            for _ in range(num_blocks)
        ])

    def get_modulation(self, block_idx, vl_emb):
        """Get (scale, shift) for one DiT block.

        Args:
            block_idx: DiT block index (0-27)
            vl_emb: [B, vl_dim] — raw Qwen3-VL embedding
        Returns:
            scale: [B, block_dim], shift: [B, block_dim]
        """
        if vl_emb.dim() == 3 and vl_emb.shape[1] == 1:
            vl_emb = vl_emb.squeeze(1)
        scale, shift = self.block_mods[block_idx](vl_emb)
        # Normalize: clip scale L2 norm to <= 1.0 so modulation doesn't explode
        scale_norm = scale.norm(dim=-1, keepdim=True).clamp(min=1.0)
        scale = scale / scale_norm
        shift = shift / scale_norm
        s = self.ip_scales[block_idx]
        return s * scale, s * shift
