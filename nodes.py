"""ComfyUI custom nodes for Anima IP-Adapter."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ip_adapter_model import IPAdapter


# ─── IP-Adapter Injection ────────────────────────────────────────────

def _move_to_match(tensor_or_module, target_device, target_dtype):
    """Move a tensor or nn.Module to target device/dtype if needed. Returns the moved object."""
    if isinstance(tensor_or_module, torch.Tensor):
        if tensor_or_module.device != target_device or tensor_or_module.dtype != target_dtype:
            return tensor_or_module.to(device=target_device, dtype=target_dtype)
    elif isinstance(tensor_or_module, nn.Module):
        cur_dev = next(tensor_or_module.parameters()).device
        cur_dt = next(tensor_or_module.parameters()).dtype
        if cur_dev != target_device or cur_dt != target_dtype:
            return tensor_or_module.to(device=target_device, dtype=target_dtype)
    return tensor_or_module


def _block_forward_with_ip(block, block_idx, ip_adapter, image_tokens,
                            weight, start_at, end_at, current_sigma,
                            x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs):
    """Block.forward with IP-Adapter cross-attention injection.

    Identical to Block.forward but adds ip_cross_attn after text cross_attn:
        original:  x = gate * text_attn_out + residual
        with IPA:  x = gate * (text_attn_out + λ * ip_attn_out) + residual
    """
    residual_dtype = x_B_T_H_W_D.dtype
    compute_dtype = emb_B_T_D.dtype

    rope_emb = kwargs.get("rope_emb_L_1_1_D", None)
    adaln_lora = kwargs.get("adaln_lora_B_T_3D", None)
    extra_pos_emb = kwargs.get("extra_per_block_pos_emb", None)
    transformer_options = kwargs.get("transformer_options", {})

    if extra_pos_emb is not None:
        x_B_T_H_W_D = x_B_T_H_W_D + extra_pos_emb

    if block.use_adaln_lora:
        s1, sc1, g1 = (block.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora).chunk(3, dim=-1)
        s2, sc2, g2 = (block.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora).chunk(3, dim=-1)
        s3, sc3, g3 = (block.adaln_modulation_mlp(emb_B_T_D) + adaln_lora).chunk(3, dim=-1)
    else:
        s1, sc1, g1 = block.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
        s2, sc2, g2 = block.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
        s3, sc3, g3 = block.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

    def _fn(_x, _norm, _scale, _shift):
        return _norm(_x) * (1 + rearrange(_scale, "b t d -> b t 1 1 d")) + rearrange(_shift, "b t d -> b t 1 1 d")

    B, T, H, W, D = x_B_T_H_W_D.shape

    # 1. Self-attention (identical to original)
    normed = _fn(x_B_T_H_W_D, block.layer_norm_self_attn, sc1, s1)
    result = rearrange(
        block.self_attn(
            rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"),
            None, rope_emb=rope_emb, transformer_options=transformer_options,
        ),
        "b (t h w) d -> b t h w d", t=T, h=H, w=W,
    )
    x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g1, "b t d -> b t 1 1 d").to(residual_dtype) * result.to(residual_dtype)

    # 2. Cross-attention (text) + IP-Adapter injection
    normed = _fn(x_B_T_H_W_D, block.layer_norm_cross_attn, sc2, s2)
    normed_flat = rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d")

    text_result = rearrange(
        block.cross_attn(normed_flat, crossattn_emb, rope_emb=rope_emb, transformer_options=transformer_options),
        "b (t h w) d -> b t h w d", t=T, h=H, w=W,
    )

    gate = rearrange(g2, "b t d -> b t 1 1 d")

    # Apply IP-Adapter only within [start_at, end_at] step range
    if start_at <= current_sigma <= end_at and weight > 0:
        # Move ip_adapter and image_tokens to match input device/dtype at call time
        ip_adapter = _move_to_match(ip_adapter, x_B_T_H_W_D.device, compute_dtype)
        image_tokens = _move_to_match(image_tokens, x_B_T_H_W_D.device, compute_dtype)
        # Expand image_tokens to match batch size (CFG uses B=2)
        if image_tokens.shape[0] < B:
            image_tokens = image_tokens.expand(B, -1, -1)
        ip_out = ip_adapter.forward_block(block_idx, normed_flat, image_tokens)
        ip_out = rearrange(ip_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = gate.to(residual_dtype) * (text_result.to(residual_dtype) + weight * ip_out.to(residual_dtype)) + x_B_T_H_W_D
    else:
        # Identical to original Block.forward
        x_B_T_H_W_D = gate.to(residual_dtype) * text_result.to(residual_dtype) + x_B_T_H_W_D

    # 3. MLP (identical to original)
    normed = _fn(x_B_T_H_W_D, block.layer_norm_mlp, sc3, s3)
    result = block.mlp(normed.to(compute_dtype))
    x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g3, "b t d -> b t 1 1 d").to(residual_dtype) * result.to(residual_dtype)

    return x_B_T_H_W_D


class IPAdapterHook:
    """Hooks into Anima's DiT to inject IP-Adapter during sampling."""

    def __init__(self, ip_adapter, image_tokens, weight, start_at, end_at):
        self.ip_adapter = ip_adapter
        self.image_tokens = image_tokens
        self.weight = weight
        self.start_at = start_at
        self.end_at = end_at
        self._patches = []

    def attach(self, model_patcher):
        """Monkey-patch each DiT block's forward method."""
        dit = model_patcher.model.diffusion_model
        for block_idx, block in enumerate(dit.blocks):
            orig_forward = block.forward

            def make_forward(of, idx, blk):
                def new_forward(x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs):
                    transformer_options = kwargs.get("transformer_options", {})
                    current_sigma = transformer_options.get("sigmas", None)
                    if current_sigma is not None:
                        if hasattr(current_sigma, 'item'):
                            current_sigma = current_sigma.item()
                        elif hasattr(current_sigma, '__len__'):
                            current_sigma = float(current_sigma[0]) if len(current_sigma) > 0 else 1.0
                    else:
                        current_sigma = 1.0  # fallback: always apply

                    return _block_forward_with_ip(
                        blk, idx, self.ip_adapter, self.image_tokens,
                        self.weight, self.start_at, self.end_at, current_sigma,
                        x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs,
                    )
                return new_forward

            block.forward = make_forward(orig_forward, block_idx, block)
            self._patches.append((block, orig_forward))

    def detach(self):
        """Restore original forward methods."""
        for block, orig_forward in self._patches:
            block.forward = orig_forward
        self._patches.clear()


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

        # Auto-detect architecture from state dict
        num_blocks = max(int(k.split(".")[1]) for k in state if k.startswith("ip_cross_attns.")) + 1
        ip_dim = state["ip_cross_attns.0.q_proj.weight"].shape[0]
        num_ip_heads = ip_dim // 64  # head_dim=64
        num_tokens = state["resampler.latents"].shape[0]
        emb_dim = state["resampler.latents"].shape[1]

        ip_adapter = IPAdapter(
            emb_dim=emb_dim,
            x_dim=2048,
            num_blocks=num_blocks,
            ip_dim=ip_dim,
            num_ip_heads=num_ip_heads,
            ip_head_dim=64,
            num_tokens=num_tokens,
            num_perceiver_layers=2,
            num_perceiver_heads=4,
        )
        ip_adapter.load_state_dict(state)
        ip_adapter.eval()
        for p in ip_adapter.parameters():
            p.requires_grad_(False)

        return (ip_adapter,)


class AnimaIPAdapterApply:
    """Apply IP-Adapter to an Anima model with a reference image embedding."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("ANIMA_IPADAPTER",),
                "image_emb": ("ANIMA_IMAGE_EMB", {"tooltip": "Qwen3-VL embedding of reference image [1, 1024]"}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of sampling steps to apply IP-Adapter"}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of sampling steps to apply IP-Adapter"}),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "Global scale for IP-Adapter influence"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "anima_ipadapter"

    def apply(self, model, ipadapter, image_emb, start_at, end_at, weight):
        dit = model.model.diffusion_model
        dtype = dit.dtype
        device = next(dit.parameters()).device

        # Move ipadapter to model's dtype and device before resample
        ipadapter = ipadapter.to(dtype=dtype, device=device)

        # Resample embedding → image tokens
        with torch.no_grad():
            if image_emb.ndim == 1:
                image_emb = image_emb.unsqueeze(0)  # [1, 1024]
            if image_emb.ndim == 2:
                image_emb = image_emb.unsqueeze(1)  # [1, 1, 1024]
            image_tokens = ipadapter.resample(image_emb.to(dtype=dtype, device=device))  # [1, 16, 1024]

        # Hook into model — pass the moved ipadapter to hook
        hook = IPAdapterHook(ipadapter, image_tokens, weight, start_at, end_at)
        hook.attach(model)

        # Store hook for cleanup
        if not hasattr(model, "_ip_adapter_hooks"):
            model._ip_adapter_hooks = []
        model._ip_adapter_hooks.append(hook)

        return (model,)


class AnimaImageEmbLoader:
    """Load a pre-computed Qwen3-VL image embedding from a .pt file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emb_path": ("STRING", {"default": "", "tooltip": "Path to .pt file containing image embedding"}),
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


class AnimaQwenVLLoader:
    """Load Qwen3-VL-Embedding-2B for image embedding extraction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "Qwen/Qwen3-VL-Embedding-2B", "tooltip": "HuggingFace model ID or local path to Qwen3-VL-Embedding-2B"}),
                "dtype": (["bf16", "fp8", "int8"], {"default": "bf16", "tooltip": "Model precision"}),
            }
        }

    RETURN_TYPES = ("QWEN_VL_MODEL",)
    FUNCTION = "load"
    CATEGORY = "anima_ipadapter"

    def load(self, model_path, dtype):
        from pathlib import Path
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        if Path(model_path).exists():
            model_path = str(Path(model_path).resolve())

        if dtype == "bf16":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, dtype=torch.bfloat16, trust_remote_code=True,
            ).cuda().eval()
        elif dtype == "fp8":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, dtype=torch.float8_e4m3fn, trust_remote_code=True,
            ).cuda().eval()
        elif dtype == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, quantization_config=quant_config, trust_remote_code=True,
            ).eval()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return ({"model": model, "processor": processor},)


class AnimaQwenVLEncodeImage:
    """Extract 1024-dim Qwen3-VL embedding from a reference image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "qwen_vl_model": ("QWEN_VL_MODEL",),
            }
        }

    RETURN_TYPES = ("ANIMA_IMAGE_EMB",)
    FUNCTION = "encode"
    CATEGORY = "anima_ipadapter"

    @staticmethod
    def _pooling_last(hidden_state, attention_mask):
        """Pool by taking the last token position indicated by attention_mask."""
        flipped = attention_mask.flip(dims=[1])
        last_pos = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def encode(self, image, qwen_vl_model):
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        model = qwen_vl_model["model"]
        processor = qwen_vl_model["processor"]
        device = next(model.parameters()).device

        # ComfyUI IMAGE: [B, H, W, 3] float [0,1] → PIL
        if image.ndim == 4:
            batch = image
        else:
            batch = image.unsqueeze(0)

        embeddings = []
        for i in range(batch.shape[0]):
            img_np = (batch[i].cpu().numpy() * 255).clip(0, 255).astype("uint8")
            pil_img = PILImage.fromarray(img_np)

            messages = [
                {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img, "min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28},
                    {"type": "text", "text": "Describe this image."},
                ]},
            ]

            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images, _, _ = process_vision_info([messages], return_video_kwargs=True)
            inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_state = outputs.hidden_states[-1]
            emb = self._pooling_last(hidden_state, inputs["attention_mask"])
            emb = F.normalize(emb[:, :1024], p=2, dim=-1)  # Matryoshka truncation + L2 norm
            embeddings.append(emb.squeeze(0).cpu())

        result = torch.stack(embeddings) if len(embeddings) > 1 else embeddings[0]
        return (result,)


NODE_CLASS_MAPPINGS = {
    "AnimaIPAdapterLoader": AnimaIPAdapterLoader,
    "AnimaIPAdapterApply": AnimaIPAdapterApply,
    "AnimaImageEmbLoader": AnimaImageEmbLoader,
    "AnimaQwenVLLoader": AnimaQwenVLLoader,
    "AnimaQwenVLEncodeImage": AnimaQwenVLEncodeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaIPAdapterLoader": "Anima IP-Adapter Loader",
    "AnimaIPAdapterApply": "Anima IP-Adapter Apply",
    "AnimaImageEmbLoader": "Anima Image Embedding Loader",
    "AnimaQwenVLLoader": "Anima Qwen3-VL Loader",
    "AnimaQwenVLEncodeImage": "Anima Qwen3-VL Encode Image",
}
