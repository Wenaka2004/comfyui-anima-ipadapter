"""ComfyUI custom nodes for Anima IP-Adapter (v3 InstantCharacter-style).

Injection method: monkey-patch each DiT block's cross_attn.forward to add
IP cross-attention output alongside text cross-attention output.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ip_adapter_model import IPAdapterSigLIP


# ─── IP-Adapter Injection ────────────────────────────────────────────

class IPAdapterHook:
    """Hooks into Anima's DiT cross-attention to inject IP-Adapter output."""

    def __init__(self, ip_adapter, image_tokens, weight, start_at, end_at):
        self.ip_adapter = ip_adapter
        self.image_tokens = image_tokens
        self.weight = weight
        self.start_at = start_at
        self.end_at = end_at
        self._patches = []
        self._current_timestep = None

    def attach(self, model_patcher):
        dit = model_patcher.model.diffusion_model
        for block_idx, block in enumerate(dit.blocks):
            ca = block.cross_attn
            orig_forward = ca.forward

            def make_forward(of, bidx, blk):
                def new_forward(x, attn_params, context=None, rope_emb=None):
                    out = of(x, attn_params, context, rope_emb)
                    # Check sigma range
                    if self._current_timestep is not None:
                        sigma = self._current_timestep
                        if hasattr(sigma, 'item'):
                            sigma = sigma.item()
                        if not (self.start_at <= sigma <= self.end_at):
                            return out
                    if self.weight > 0:
                        ip_out = self.ip_adapter.forward_block(bidx, x, self.image_tokens, scale_override=None)
                        out = out + self.weight * ip_out.to(out.dtype)
                    return out
                return new_forward

            ca.forward = make_forward(orig_forward, block_idx, block)
            self._patches.append((ca, orig_forward))

    def detach(self):
        for ca, orig_forward in self._patches:
            ca.forward = orig_forward
        self._patches.clear()


class IPAdapterHookManager:
    """Manages IPAdapterHook lifecycle + per-step timestep update for TimeResampler."""

    def __init__(self, hook, ip_adapter, siglip_features, model_patcher):
        self.hook = hook
        self.ip_adapter = ip_adapter
        self.siglip_features = siglip_features
        self.model_patcher = model_patcher
        self._callback_id = None

    def update_timestep(self, timestep):
        """Called at each denoising step to update image_tokens for the current timestep."""
        self.hook._current_timestep = timestep
        # Recompute image_tokens with current timestep (TimeResampler is timestep-conditioned)
        with torch.no_grad():
            ts = timestep.flatten() if hasattr(timestep, 'flatten') else timestep
            self.hook.image_tokens = self.ip_adapter.encode_ref(self.siglip_features, timestep=ts)

    def detach(self):
        self.hook.detach()
        if self._callback_id is not None:
            # Remove callback if registered
            pass


# ─── ComfyUI Nodes ────────────────────────────────────────────────────

class AnimaIPAdapterLoader:
    """Load Anima IP-Adapter from safetensors checkpoint."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ipadapter_path": ("STRING", {"default": "", "tooltip": "Path to IP-Adapter safetensors file"}),
            }
        }

    RETURN_TYPES = ("ANIMA_IPADAPTER",)
    FUNCTION = "load"
    CATEGORY = "anima_ipadapter"

    def load(self, ipadapter_path):
        from safetensors.torch import load_file
        state = load_file(ipadapter_path)

        # Auto-detect architecture from state dict keys
        if "resampler.time_proj.weight" in state:
            # v3: TimeResampler (InstantCharacter-style)
            num_blocks = max(int(k.split(".")[1]) for k in state if k.startswith("ip_cross_attns.")) + 1
            num_queries = state["resampler.latents"].shape[1]
            resampler_dim = state["resampler.latents"].shape[2]
            input_dim = state["resampler.proj_in.weight"].shape[1]
            output_dim = state["resampler.proj_out.weight"].shape[0]
            ip_adapter = IPAdapterSigLIP(
                input_dim=input_dim, dit_dim=output_dim, num_blocks=num_blocks,
                num_queries=num_queries, resampler_depth=len([k for k in state if k.startswith("resampler.layers.") and "adaLN" in k]) // 1,
                resampler_dim=resampler_dim,
            )
        elif "resampler.latents" in state:
            # v1/v2: old Perceiver (incompatible)
            raise RuntimeError(
                "v1/v2 checkpoint not compatible with v3 architecture. "
                "Please retrain with the latest code."
            )
        else:
            raise KeyError(f"Unknown checkpoint format. Keys: {list(state.keys())[:10]}")

        ip_adapter.load_state_dict(state, strict=False)
        ip_adapter.eval()
        for p in ip_adapter.parameters():
            p.requires_grad_(False)

        return (ip_adapter,)


class AnimaIPAdapterApply:
    """Apply IP-Adapter to an Anima model with SigLIP image features."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("ANIMA_IPADAPTER",),
                "siglip_features": ("SIGLIP_FEATURES", {"tooltip": "SigLIP2 patch features [1, N, 768]"}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start sigma for IP-Adapter"}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End sigma for IP-Adapter"}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "IP-Adapter strength"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "anima_ipadapter"

    def apply(self, model, ipadapter, siglip_features, start_at, end_at, weight):
        dit = model.model.diffusion_model
        dtype = dit.dtype
        device = next(dit.parameters()).device

        ipadapter = ipadapter.to(dtype=dtype, device=device)
        siglip_features = siglip_features.to(dtype=dtype, device=device)

        # Compute initial image_tokens at t=0.5 (will be updated per-step)
        with torch.no_grad():
            image_tokens = ipadapter.encode_ref(siglip_features, timestep=torch.tensor([0.5], device=device, dtype=dtype))

        hook = IPAdapterHook(ipadapter, image_tokens, weight, start_at, end_at)
        hook.attach(model)

        if not hasattr(model, "_ip_adapter_hooks"):
            model._ip_adapter_hooks = []
        model._ip_adapter_hooks.append(hook)

        return (model,)


class AnimaSiglipeEncodeImage:
    """Extract SigLIP2 patch features from a reference image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("SIGLIP_FEATURES",)
    FUNCTION = "encode"
    CATEGORY = "anima_ipadapter"

    def encode(self, image):
        from PIL import Image as PILImage
        from transformers import SiglipVisionModel, AutoImageProcessor

        # Lazy load
        if not hasattr(self, '_siglip'):
            self._siglip = SiglipVisionModel.from_pretrained(
                "google/siglip2-base-patch16-512",
                torch_dtype=torch.bfloat16, trust_remote_code=True,
            ).cuda().eval()
            self._siglip_proc = AutoImageProcessor.from_pretrained(
                "google/siglip2-base-patch16-512", trust_remote_code=True,
            )

        # ComfyUI IMAGE: [B, H, W, 3] float [0,1] → PIL list
        if image.ndim == 4:
            batch = image
        else:
            batch = image.unsqueeze(0)

        pil_images = []
        for i in range(batch.shape[0]):
            img_np = (batch[i].cpu().numpy() * 255).clip(0, 255).astype("uint8")
            pil_images.append(PILImage.fromarray(img_np))

        # Pad to square 512
        padded = []
        for pil in pil_images:
            w, h = pil.size
            size = max(w, h)
            if w == h:
                padded.append(pil.resize((512, 512)))
            else:
                canvas = PILImage.new("RGB", (size, size), (255, 255, 255))
                canvas.paste(pil, ((size - w) // 2, (size - h) // 2))
                padded.append(canvas.resize((512, 512)))

        inputs = self._siglip_proc(images=padded, return_tensors="pt", do_resize=False)
        inputs = {k: v.to("cuda", dtype=torch.bfloat16) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._siglip(**inputs).last_hidden_state  # [B, N, 768]

        return (features.float(),)


# Legacy node aliases (backward compat)
class AnimaImageEmbLoader:
    """Load a pre-computed image embedding from a .pt file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emb_path": ("STRING", {"default": "", "tooltip": "Path to .pt file"}),
            }
        }

    RETURN_TYPES = ("ANIMA_IMAGE_EMB",)
    FUNCTION = "load"
    CATEGORY = "anima_ipadapter"

    def load(self, emb_path):
        emb = torch.load(emb_path, map_location="cpu", weights_only=True)
        if emb.ndim > 2:
            emb = emb.squeeze()
        return (emb,)


NODE_CLASS_MAPPINGS = {
    "AnimaIPAdapterLoader": AnimaIPAdapterLoader,
    "AnimaIPAdapterApply": AnimaIPAdapterApply,
    "AnimaSiglipeEncodeImage": AnimaSiglipeEncodeImage,
    "AnimaImageEmbLoader": AnimaImageEmbLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaIPAdapterLoader": "Anima IP-Adapter Loader",
    "AnimaIPAdapterApply": "Anima IP-Adapter Apply",
    "AnimaSiglipeEncodeImage": "Anima SigLIP2 Encode Image",
    "AnimaImageEmbLoader": "Anima Image Embedding Loader (Legacy)",
}
