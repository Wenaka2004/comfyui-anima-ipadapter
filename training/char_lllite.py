"""Character-conditioned LLLite for Anima DiT (~5M trainable params).

SigLIP(ref) → [768] → Linear(→32) → cond_emb
  For each DiT block q_proj: LLLiteModule injects cond into q_proj input.

No resampler, no cross-attn, no collapse risk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLLiteModule(nn.Module):
    """Injects cond_emb into one Linear's input. ~180K params per 2048-dim q_proj."""

    def __init__(self, in_dim, cond_dim=32, mlp_dim=64):
        super().__init__()
        self.cond_dim = cond_dim
        self.down = nn.Linear(in_dim, mlp_dim, bias=False)
        self.mid = nn.Linear(cond_dim + mlp_dim, mlp_dim, bias=False)
        self.up = nn.Linear(mlp_dim, in_dim, bias=False)
        self.cond_to_film = nn.Linear(cond_dim, mlp_dim * 2, bias=False)
        self.depth_emb = nn.Parameter(torch.zeros(cond_dim))
        self.multiplier = 1.0

        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.normal_(self.mid.weight, std=0.01)
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.normal_(self.cond_to_film.weight, std=0.01)

    def forward(self, x, cond):
        """x: [B, S, D] (before q_proj), cond: [B, 1, C]"""
        if self.multiplier == 0.0 or cond is None:
            return None
        cond = cond + self.depth_emb
        h = F.silu(self.down(x))
        film = self.cond_to_film(cond)
        gamma, beta = film.chunk(2, dim=-1)
        h = self.mid(torch.cat([cond.expand(-1, x.shape[1], -1), h], dim=-1))
        h = h * (1.0 + gamma) + beta
        h = F.silu(h)
        out = self.up(h)
        # Normalize: keep L2 norm per-token bounded to prevent explosion
        out_norm = out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        out = out / out_norm  # L2-normalize, bounded at unit norm
        return out * self.multiplier


class CharLLLite(nn.Module):
    """Character LLLite adapter for Anima DiT.

    Self-contained: hooks/unhooks DiT q_proj layers internally.
    """

    def __init__(self, dit, siglip_dim=768, cond_dim=32, mlp_dim=64):
        super().__init__()
        self.cond_dim = cond_dim

        # SigLIP → cond
        self.siglip_proj = nn.Sequential(
            nn.Linear(siglip_dim, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, cond_dim, bias=False),
        )

        # Discover q_proj in DiT blocks
        self.lllite_mods = nn.ModuleList()
        self._targets = []
        self._orig_forwards = {}
        for i, block in enumerate(dit.blocks):
            sa = getattr(block, 'self_attn', None)
            if sa and hasattr(sa, 'q_proj') and isinstance(sa.q_proj, nn.Linear):
                mod = LLLiteModule(in_dim=sa.q_proj.in_features,
                                   cond_dim=cond_dim, mlp_dim=mlp_dim)
                self.lllite_mods.append(mod)
                self._targets.append(sa.q_proj)

        self._cond = None  # current cond_emb

    def encode_ref(self, siglip_features):
        """siglip: [B, N, 768] → cond: [B, 1, cond_dim]"""
        pooled = siglip_features.mean(dim=1)
        return self.siglip_proj(pooled).unsqueeze(1)

    def set_cond(self, cond):
        self._cond = cond

    def set_multiplier(self, m):
        for mod in self.lllite_mods:
            mod.multiplier = m

    def apply_to(self):
        cond = self._cond
        for i, (target, mod) in enumerate(zip(self._targets, self.lllite_mods)):
            orig = target.forward
            self._orig_forwards[i] = orig

            def make_fwd(m=mod, c=cond, of=orig):
                def fwd(x):
                    out = m(x, c)
                    if out is None:
                        return of(x)
                    return of(x + out)
                return fwd
            target.forward = make_fwd()

    def restore(self):
        for i, target in enumerate(self._targets):
            if i in self._orig_forwards:
                target.forward = self._orig_forwards[i]
        self._orig_forwards.clear()
