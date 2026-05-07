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


def _block_forward_with_ip(block, block_idx, ip_adapter, vl_emb,
                            weight, start_at, end_at, current_sigma,
                            x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs):
    """Block.forward with IP AdaLN modulation after MLP.

    Identical to original Block.forward, with IP modulation added after MLP:
        x = x * (1 + weight * scale) + weight * shift
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

    # 1. Self-attention
    normed = _fn(x_B_T_H_W_D, block.layer_norm_self_attn, sc1, s1)
    result = rearrange(
        block.self_attn(
            rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"),
            None, rope_emb=rope_emb, transformer_options=transformer_options,
        ),
        "b (t h w) d -> b t h w d", t=T, h=H, w=W,
    )
    x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g1, "b t d -> b t 1 1 d").to(residual_dtype) * result.to(residual_dtype)

    # 2. Cross-attention (text)
    normed = _fn(x_B_T_H_W_D, block.layer_norm_cross_attn, sc2, s2)
    normed_flat = rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d")
    text_result = rearrange(
        block.cross_attn(normed_flat, crossattn_emb, rope_emb=rope_emb, transformer_options=transformer_options),
        "b (t h w) d -> b t h w d", t=T, h=H, w=W,
    )
    gate = rearrange(g2, "b t d -> b t 1 1 d")
    x_B_T_H_W_D = gate.to(residual_dtype) * text_result.to(residual_dtype) + x_B_T_H_W_D

    # 3. MLP
    normed = _fn(x_B_T_H_W_D, block.layer_norm_mlp, sc3, s3)
    result = block.mlp(normed.to(compute_dtype))
    x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g3, "b t d -> b t 1 1 d").to(residual_dtype) * result.to(residual_dtype)

    # 4. IP AdaLN modulation (after full block, within [start_at, end_at])
    if start_at <= current_sigma <= end_at and weight > 0:
        ip_adapter = _move_to_match(ip_adapter, x_B_T_H_W_D.device, compute_dtype)
        vl_emb_moved = _move_to_match(vl_emb, x_B_T_H_W_D.device, x_B_T_H_W_D.dtype)
        # Expand VL emb to match batch size (CFG uses B=2)
        if vl_emb_moved.shape[0] < B:
            vl_emb_moved = vl_emb_moved.expand(B, -1)
        ip_scale, ip_shift = ip_adapter.get_modulation(block_idx, vl_emb_moved)
        ip_scale = ip_scale.to(residual_dtype).reshape(B, 1, 1, 1, D)
        ip_shift = ip_shift.to(residual_dtype).reshape(B, 1, 1, 1, D)
        x_B_T_H_W_D = x_B_T_H_W_D * (1.0 + weight * ip_scale) + weight * ip_shift

    return x_B_T_H_W_D


class IPAdapterHook:
    """Hooks into Anima's DiT to inject IP-Adapter AdaLN modulation during sampling."""

    def __init__(self, ip_adapter, vl_emb, weight, start_at, end_at):
        self.ip_adapter = ip_adapter
        self.vl_emb = vl_emb
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
                        blk, idx, self.ip_adapter, self.vl_emb,
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
        # v4 (AdaLN): has block_mods.0.proj.weight
        # v3 (MLP projector): has token_projector.proj_up.weight
        # v1/v2 (Perceiver): has resampler.latents
        # v1 (cross-attn): has ip_cross_attns.0.q_proj.weight

        num_blocks = None
        if "block_mods.0.proj.weight" in state:
            # v4: AdaLN modulation
            vl_dim = state["block_mods.0.proj.weight"].shape[1]
            modulation_hidden = state["block_mods.0.proj.weight"].shape[0]
            block_dim = state["block_mods.0.mod.weight"].shape[0] // 2
            num_blocks = sum(1 for k in state if k.startswith("block_mods.") and k.endswith(".proj.weight"))
            ip_adapter = IPAdapter(
                vl_dim=vl_dim, block_dim=block_dim, num_blocks=num_blocks,
                modulation_hidden=modulation_hidden,
            )
        elif "token_projector.proj_up.weight" in state:
            # v3: MLP Token Projector (cross-attn) — backward compat
            num_blocks = max(int(k.split(".")[1]) for k in state if k.startswith("ip_cross_attns.")) + 1
            num_tokens = state["token_projector.proj_down.weight"].shape[0]
            emb_dim = state["token_projector.proj_down.weight"].shape[1]
            hidden_dim = state["token_projector.proj_up.weight"].shape[0]
            mlp_hidden_mult = hidden_dim // (num_tokens * emb_dim)
            ip_adapter = IPAdapter(
                vl_dim=emb_dim, block_dim=2048, num_blocks=num_blocks,
                modulation_hidden=256,
            )
            # v3 uses different architecture, can't load into v4
            raise RuntimeError(
                "v3 checkpoint format is not compatible with v4 AdaLN architecture. "
                "Please use a v4 checkpoint or revert to an older version of the nodes."
            )
        elif "resampler.latents" in state:
            # v1/v2: Perceiver — backward compat
            raise RuntimeError(
                "v1/v2 checkpoint format is not compatible with v4 AdaLN architecture. "
                "Please use a v4 checkpoint or revert to an older version of the nodes."
            )
        else:
            raise KeyError("Unknown checkpoint format — cannot detect architecture")

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

        # Move ipadapter to model's dtype and device
        ipadapter = ipadapter.to(dtype=dtype, device=device)

        # Prepare VL embedding: [1, 1024]
        with torch.no_grad():
            if image_emb.ndim == 1:
                vl_emb = image_emb.unsqueeze(0)  # [1, 1024]
            elif image_emb.ndim == 3:
                vl_emb = image_emb.squeeze(1)  # [1, 1, 1024] → [1, 1024]
            else:
                vl_emb = image_emb  # [1, 1024] or [B, 1024]
            vl_emb = vl_emb.to(dtype=dtype, device=device)

        # Hook into model
        hook = IPAdapterHook(ipadapter, vl_emb, weight, start_at, end_at)
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


class AnimaCCIPEncodeImage:
    """Extract 768-dim CCIP feature from a reference image.

    Uses local ONNX model from ComfyUI/models/onnx/ccip-caformer-24-randaug-pruned/
    """

    _sessions = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "ccip_model": (cls.list_models(), {"tooltip": "CCIP ONNX model folder under models/onnx/"}),
            }
        }

    RETURN_TYPES = ("ANIMA_IMAGE_EMB",)
    FUNCTION = "encode"
    CATEGORY = "anima_ipadapter"

    @staticmethod
    def list_models():
        import folder_paths
        onnx_dir = os.path.join(folder_paths.models_dir, "onnx")
        if not os.path.isdir(onnx_dir):
            return ["none"]
        models = []
        # Check subdirectories for model_feat.onnx
        for entry in os.listdir(onnx_dir):
            path = os.path.join(onnx_dir, entry)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "model_feat.onnx")):
                models.append(entry)
        # Also check root onnx dir for model_feat.onnx
        if os.path.exists(os.path.join(onnx_dir, "model_feat.onnx")):
            models.append("(root) model_feat.onnx")
        return models if models else ["none"]

    def _get_session(self, model_name):
        if model_name in self._sessions:
            return self._sessions[model_name]

        import onnxruntime
        import folder_paths

        if model_name in ("(root) model_feat.onnx", "."):
            model_path = os.path.join(folder_paths.models_dir, "onnx", "model_feat.onnx")
        else:
            model_path = os.path.join(folder_paths.models_dir, "onnx", model_name, "model_feat.onnx")

        providers = onnxruntime.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = onnxruntime.InferenceSession(model_path, providers=providers)
        self._sessions[model_name] = sess
        return sess

    def encode(self, image, ccip_model):
        import numpy as np
        import onnxruntime
        from PIL import Image as PILImage

        if ccip_model == "none":
            raise RuntimeError("No CCIP model found. Download from https://huggingface.co/deepghs/ccip_onnx "
                             "and place model_feat.onnx in ComfyUI/models/onnx/ccip-caformer-24-randaug-pruned/")

        # ComfyUI IMAGE: [B, H, W, 3] float [0,1] → preprocessed tensor
        if image.ndim == 4:
            batch = image
        else:
            batch = image.unsqueeze(0)

        B = batch.shape[0]
        # Resize to 384x384 and normalize (ImageNet stats)
        preprocessed = torch.zeros(B, 3, 384, 384, dtype=torch.float32)
        for i in range(B):
            img_np = (batch[i].cpu().numpy() * 255).clip(0, 255).astype("uint8")
            pil = PILImage.fromarray(img_np).resize((384, 384), PILImage.LANCZOS)
            arr = np.array(pil, dtype=np.float32) / 255.0
            # ImageNet normalization
            arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            preprocessed[i] = torch.from_numpy(arr.transpose(2, 0, 1))

        # ONNX inference
        sess = self._get_session(ccip_model)
        input_name = sess.get_inputs()[0].name
        feats = sess.run(None, {input_name: preprocessed.numpy()})[0]  # [B, 768]
        return (torch.from_numpy(feats).float(),)


NODE_CLASS_MAPPINGS = {
    "AnimaIPAdapterLoader": AnimaIPAdapterLoader,
    "AnimaIPAdapterApply": AnimaIPAdapterApply,
    "AnimaImageEmbLoader": AnimaImageEmbLoader,
    "AnimaQwenVLLoader": AnimaQwenVLLoader,
    "AnimaQwenVLEncodeImage": AnimaQwenVLEncodeImage,
    "AnimaCCIPEncodeImage": AnimaCCIPEncodeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaIPAdapterLoader": "Anima IP-Adapter Loader",
    "AnimaIPAdapterApply": "Anima IP-Adapter Apply",
    "AnimaImageEmbLoader": "Anima Image Embedding Loader",
    "AnimaQwenVLLoader": "Anima Qwen3-VL Loader",
    "AnimaQwenVLEncodeImage": "Anima Qwen3-VL Encode Image",
    "AnimaCCIPEncodeImage": "Anima CCIP Encode Image",
}
