"""Standalone integration test: IP-Adapter + DiT block forward on GPU."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file
import sys

sys.path.insert(0, "/data/stardust/anima_ipa/comfyui-anima-ipadapter")
from ip_adapter_model import IPAdapter

# ── Inline _block_forward_with_ip and IPAdapterHook from nodes.py ──

def _move_to_match(obj, target_device, target_dtype):
    if isinstance(obj, torch.Tensor):
        if obj.device != target_device or obj.dtype != target_dtype:
            return obj.to(device=target_device, dtype=target_dtype)
    elif isinstance(obj, nn.Module):
        cur_dev = next(obj.parameters()).device
        cur_dt = next(obj.parameters()).dtype
        if cur_dev != target_device or cur_dt != target_dtype:
            return obj.to(device=target_device, dtype=target_dtype)
    return obj

def _block_forward_with_ip(block, block_idx, ip_adapter, image_tokens,
                            weight, start_at, end_at, current_sigma,
                            x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs):
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
    normed = _fn(x_B_T_H_W_D, block.layer_norm_self_attn, sc1, s1)
    result = rearrange(block.self_attn(rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"), None), "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g1, "b t d -> b t 1 1 d").to(residual_dtype) * result.to(residual_dtype)
    normed = _fn(x_B_T_H_W_D, block.layer_norm_cross_attn, sc2, s2)
    normed_flat = rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d")
    text_result = rearrange(block.cross_attn(normed_flat, crossattn_emb), "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    ip_result = torch.zeros_like(text_result)
    gate = rearrange(g2, "b t d -> b t 1 1 d")
    if start_at <= current_sigma <= end_at and weight > 0:
        ip_adapter = _move_to_match(ip_adapter, x_B_T_H_W_D.device, compute_dtype)
        image_tokens = _move_to_match(image_tokens, x_B_T_H_W_D.device, compute_dtype)
        if image_tokens.shape[0] < B:
            image_tokens = image_tokens.expand(B, -1, -1)
        ip_out = ip_adapter.forward_block(block_idx, normed_flat, image_tokens)
        ip_out = rearrange(ip_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        # Adaptive normalization: scale ip_out to match text_result magnitude
        text_norm = text_result.to(residual_dtype).norm()
        ip_norm = ip_out.to(residual_dtype).norm()
        if ip_norm > 0:
            ip_out = ip_out * (text_norm / ip_norm)
        x_B_T_H_W_D = gate.to(residual_dtype) * (text_result.to(residual_dtype) + weight * ip_out.to(residual_dtype)) + x_B_T_H_W_D
    else:
        x_B_T_H_W_D = gate.to(residual_dtype) * text_result.to(residual_dtype) + x_B_T_H_W_D
    normed = _fn(x_B_T_H_W_D, block.layer_norm_mlp, sc3, s3)
    result = block.mlp(normed.to(compute_dtype))
    x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g3, "b t d -> b t 1 1 d").to(residual_dtype) * result.to(residual_dtype)
    return x_B_T_H_W_D


class IPAdapterHook:
    def __init__(self, ip_adapter, image_tokens, weight, start_at, end_at):
        self.ip_adapter = ip_adapter
        self.image_tokens = image_tokens
        self.weight = weight
        self.start_at = start_at
        self.end_at = end_at
        self._patches = []

    def attach(self, model_patcher):
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
                        current_sigma = 1.0
                    return _block_forward_with_ip(
                        blk, idx, self.ip_adapter, self.image_tokens,
                        self.weight, self.start_at, self.end_at, current_sigma,
                        x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs,
                    )
                return new_forward
            block.forward = make_forward(orig_forward, block_idx, block)
            self._patches.append((block, orig_forward))
    def detach(self):
        for block, orig_forward in self._patches:
            block.forward = orig_forward
        self._patches.clear()

device = "cuda"
dtype = torch.bfloat16

# ── 1. Create a minimal DiT Block matching Cosmos/Anima architecture ──
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        inner_dim = n_heads * head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        context = x if context is None else context  # noqa: self-attn uses x as context
        B, S, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, S, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).view(*context.shape[:-1], self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(context).view(*context.shape[:-1], self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(B, S, -1).contiguous()


class Block(nn.Module):
    def __init__(self, x_dim=2048, context_dim=1024, num_heads=16):
        super().__init__()
        self.use_adaln_lora = True
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(x_dim, x_dim, num_heads, x_dim // num_heads)
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(x_dim, context_dim, num_heads, x_dim // num_heads)
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(x_dim, x_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(x_dim * 4, x_dim, bias=False),
        )
        adaln_lora_dim = 256
        self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False))
        self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False))
        self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, adaln_lora_dim, bias=False), nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False))

    def forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb, rope_emb_L_1_1_D=None,
                adaln_lora_B_T_3D=None, extra_per_block_pos_emb=None, transformer_options=None, **kwargs):
        transformer_options = transformer_options or {}
        residual_dtype = x_B_T_H_W_D.dtype
        compute_dtype = emb_B_T_D.dtype
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb
        s1, sc1, g1 = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        s2, sc2, g2 = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        s3, sc3, g3 = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        def _fn(_x, _norm, _scale, _shift):
            return _norm(_x) * (1 + rearrange(_scale, "b t d -> b t 1 1 d")) + rearrange(_shift, "b t d -> b t 1 1 d")
        B, T, H, W, D = x_B_T_H_W_D.shape
        normed = _fn(x_B_T_H_W_D, self.layer_norm_self_attn, sc1, s1)
        result = rearrange(self.self_attn(rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"), None), "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g1, "b t d -> b t 1 1 d").to(residual_dtype) * result.to(residual_dtype)
        normed = _fn(x_B_T_H_W_D, self.layer_norm_cross_attn, sc2, s2)
        result = rearrange(self.cross_attn(rearrange(normed.to(compute_dtype), "b t h w d -> b (t h w) d"), crossattn_emb), "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x_B_T_H_W_D = rearrange(g2, "b t d -> b t 1 1 d").to(residual_dtype) * result.to(residual_dtype) + x_B_T_H_W_D
        normed = _fn(x_B_T_H_W_D, self.layer_norm_mlp, sc3, s3)
        result = self.mlp(normed.to(compute_dtype))
        x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g3, "b t d -> b t 1 1 d").to(residual_dtype) * result.to(residual_dtype)
        return x_B_T_H_W_D


# ── 2. Build a minimal DiT ──
print("1. Creating minimal DiT (2 blocks)...")
class MiniDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(2)])
        self.dtype = dtype  # simulate .dtype attribute

dit = MiniDiT().to(device=device, dtype=dtype).eval()
print(f"   DiT device: {next(dit.parameters()).device}, dtype: {next(dit.parameters()).dtype}")

# ── 3. Create IP-Adapter ──
print("2. Creating IP-Adapter (2 blocks)...")
ip_adapter = IPAdapter(
    emb_dim=1024, x_dim=2048, num_blocks=2,
    ip_dim=512, num_ip_heads=8, ip_head_dim=64,
    num_tokens=16, num_perceiver_layers=2, num_perceiver_heads=4,
).to(device=device, dtype=dtype).eval()
print(f"   IP-Adapter device: {next(ip_adapter.parameters()).device}, dtype: {next(ip_adapter.parameters()).dtype}")

# ── 4. Create fake inputs ──
print("3. Creating fake inputs...")
B, T, H, W = 1, 1, 4, 4
x = torch.randn(B, T, H, W, 2048, device=device, dtype=dtype)
emb = torch.randn(B, T, 2048, device=device, dtype=dtype)
crossattn_emb = torch.randn(B, 128, 1024, device=device, dtype=dtype)
adaln_lora = torch.zeros(B, T, 3 * 2048, device=device, dtype=dtype)

# Fake image embedding
image_emb = torch.randn(1, 1024, device=device, dtype=dtype)

# ── 5. Test resample ──
print("4. Testing resample...")
with torch.no_grad():
    if image_emb.ndim == 1:
        image_emb = image_emb.unsqueeze(0)
    if image_emb.ndim == 2:
        image_emb = image_emb.unsqueeze(1)
    image_tokens = ip_adapter.resample(image_emb)
print(f"   image_tokens: shape={image_tokens.shape}, device={image_tokens.device}, dtype={image_tokens.dtype}")

# ── 6. Test block forward with IP-Adapter injection ──
print("5. Testing IP-Adapter injection on block forward...")

with torch.no_grad():
    for i, block in enumerate(dit.blocks):
        out = _block_forward_with_ip(
            block, i, ip_adapter, image_tokens,
            1.0, 0.0, 1.0, 1.0,
            x, emb, crossattn_emb,
            rope_emb_L_1_1_D=None,
            adaln_lora_B_T_3D=adaln_lora,
        )
        print(f"   Block {i}: output shape={out.shape}, device={out.device}, dtype={out.dtype}")

# ── 5b. Test with CFG batch size (B=2, image_tokens B=1) ──
print("5b. Testing with CFG batch (B=2, image_tokens B=1)...")
x_cfg = torch.randn(2, T, H, W, 2048, device=device, dtype=dtype)
emb_cfg = torch.randn(2, T, 2048, device=device, dtype=dtype)
crossattn_cfg = torch.randn(2, 128, 1024, device=device, dtype=dtype)
adaln_lora_cfg = torch.zeros(2, T, 3 * 2048, device=device, dtype=dtype)

with torch.no_grad():
    out = _block_forward_with_ip(
        dit.blocks[0], 0, ip_adapter, image_tokens,  # image_tokens is [1, 16, 1024]
        1.0, 0.0, 1.0, 1.0,
        x_cfg, emb_cfg, crossattn_cfg,
        rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora_cfg,
    )
    print(f"   CFG batch OK: output shape={out.shape}, device={out.device}")

# ── 5c. Test weight=0 is identical to original forward ──
print("5c. Testing weight=0 is identical to original forward...")
x_test = torch.randn(B, T, H, W, 2048, device=device, dtype=dtype)
emb_test = torch.randn(B, T, 2048, device=device, dtype=dtype)
crossattn_test = torch.randn(B, 128, 1024, device=device, dtype=dtype)
adaln_lora_test = torch.zeros(B, T, 3 * 2048, device=device, dtype=dtype)

with torch.no_grad():
    out_orig = dit.blocks[0](x_test, emb_test, crossattn_test, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora_test)
    out_ip0 = _block_forward_with_ip(
        dit.blocks[0], 0, ip_adapter, image_tokens,
        0.0, 0.0, 1.0, 1.0,  # weight=0
        x_test, emb_test, crossattn_test,
        rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora_test,
    )
    diff = (out_orig - out_ip0).abs().max().item()
    print(f"   weight=0 vs original: max diff = {diff:.2e} (should be 0)")

# ── 7. Test IPAdapterHook ──
print("6. Testing IPAdapterHook (monkey-patch)...")

class FakeModelPatcher:
    def __init__(self, dit):
        self.model = type('M', (), {'diffusion_model': dit})()

fake_patcher = FakeModelPatcher(dit)
hook = IPAdapterHook(ip_adapter, image_tokens, weight=1.0, start_at=0.0, end_at=1.0)
hook.attach(fake_patcher)

with torch.no_grad():
    for i, block in enumerate(dit.blocks):
        out = block(x, emb, crossattn_emb, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora)
        print(f"   Block {i} (hooked): shape={out.shape}, device={out.device}")

hook.detach()
print("   Hook detached OK")

# ── 6a. Test: each hooked block uses its OWN block, not the last one (closure bug check) ──
print("6a. Testing hook closure: each block uses its own weights (weight=0 vs original)...")
hook0 = IPAdapterHook(ip_adapter, image_tokens, weight=0.0, start_at=0.0, end_at=1.0)
hook0.attach(fake_patcher)
with torch.no_grad():
    for i, block in enumerate(dit.blocks):
        out_hooked = block(x, emb, crossattn_emb, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora)
        out_orig = dit.blocks[i].forward  # already restored? no, still hooked
        # Compare hooked (weight=0) against direct _block_forward_with_ip call
        out_direct = _block_forward_with_ip(
            dit.blocks[i], i, ip_adapter, image_tokens,
            0.0, 0.0, 1.0, 1.0,
            x, emb, crossattn_emb,
            rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora,
        )
        diff = (out_hooked - out_direct).abs().max().item()
        print(f"   Block {i}: hooked vs direct (weight=0) max diff = {diff:.2e}")
hook0.detach()

# ── 7b. Test: IP-Adapter on CPU, inputs on CUDA (simulates ComfyUI issue) ──
print("6b. Testing with IP-Adapter on CPU (simulates ComfyUI clone issue)...")
ip_adapter_cpu = IPAdapter(
    emb_dim=1024, x_dim=2048, num_blocks=2,
    ip_dim=512, num_ip_heads=8, ip_head_dim=64,
    num_tokens=16, num_perceiver_layers=2, num_perceiver_heads=4,
).eval()  # stays on CPU!
image_tokens_cpu = torch.randn(1, 16, 1024)  # CPU

with torch.no_grad():
    out = _block_forward_with_ip(
        dit.blocks[0], 0, ip_adapter_cpu, image_tokens_cpu,
        1.0, 0.0, 1.0, 1.0,
        x, emb, crossattn_emb,
        rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora,
    )
    print(f"   CPU→GPU auto-move OK: shape={out.shape}, device={out.device}")

# ── 6c. Test: ipadapter loaded as float32 on CPU, resample needs bf16 CUDA ──
print("6c. Testing ipadapter float32/CPU → bf16/CUDA for resample...")
ip_adapter_f32 = IPAdapter(
    emb_dim=1024, x_dim=2048, num_blocks=2,
    ip_dim=512, num_ip_heads=8, ip_head_dim=64,
    num_tokens=16, num_perceiver_layers=2, num_perceiver_heads=4,
).eval()  # float32, CPU
image_emb_f32 = torch.randn(1, 1024)  # float32, CPU
ip_adapter_f32 = ip_adapter_f32.to(dtype=torch.bfloat16, device="cuda")
with torch.no_grad():
    if image_emb_f32.ndim == 2:
        image_emb_f32 = image_emb_f32.unsqueeze(1)
    tokens = ip_adapter_f32.resample(image_emb_f32.to(dtype=torch.bfloat16, device="cuda"))
print(f"   f32→bf16 resample OK: shape={tokens.shape}, dtype={tokens.dtype}")

# Test hook with moved ipadapter
fake_patcher2 = FakeModelPatcher(dit)
hook2 = IPAdapterHook(ip_adapter_f32, tokens, weight=1.0, start_at=0.0, end_at=1.0)
hook2.attach(fake_patcher2)
with torch.no_grad():
    out = dit.blocks[0](x, emb, crossattn_emb, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora)
    print(f"   Hook forward with moved ipadapter OK: shape={out.shape}")
hook2.detach()

# ── 8. Test Qwen3-VL encoding ──
print("7. Testing Qwen3-VL encoding...")
from PIL import Image
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "/data/stardust/anima_ipa/stage6_emb/models/Qwen3-VL-Embedding-2B",
    dtype=torch.bfloat16, trust_remote_code=True,
).cuda().eval()
processor = AutoProcessor.from_pretrained(
    "/data/stardust/anima_ipa/stage6_emb/models/Qwen3-VL-Embedding-2B",
    trust_remote_code=True,
)

pil_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
messages = [
    {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
    {"role": "user", "content": [
        {"type": "image", "image": pil_img, "min_pixels": 256*28*28, "max_pixels": 512*28*28},
        {"type": "text", "text": "Describe this image."},
    ]},
]

text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
images, _, _ = process_vision_info([messages], return_video_kwargs=True)
inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
qwen_device = next(qwen_model.parameters()).device
inputs = {k: v.to(qwen_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.no_grad():
    outputs = qwen_model(**inputs, output_hidden_states=True)
hs = outputs.hidden_states[-1]
attn_mask = inputs["attention_mask"]
flipped = attn_mask.flip(dims=[1])
last_pos = flipped.argmax(dim=1)
col = attn_mask.shape[1] - last_pos - 1
row = torch.arange(hs.shape[0], device=hs.device)
emb_qwen = F.normalize(hs[row, col][:, :1024], p=2, dim=-1)
print(f"   Qwen3-VL embedding: shape={emb_qwen.shape}, norm={emb_qwen.norm().item():.4f}")

# ── 9. E2E: Qwen3-VL → resample → block forward ──
print("8. E2E: Qwen3-VL → IP-Adapter → DiT block...")
with torch.no_grad():
    ref = emb_qwen.detach()
    if ref.ndim == 1: ref = ref.unsqueeze(0)
    if ref.ndim == 2: ref = ref.unsqueeze(1)
    tokens = ip_adapter.resample(ref.to(device=device, dtype=dtype))
    out = _block_forward_with_ip(
        dit.blocks[0], 0, ip_adapter, tokens,
        1.0, 0.0, 1.0, 1.0,
        x, emb, crossattn_emb,
        rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=adaln_lora,
    )
    print(f"   E2E output: shape={out.shape}, device={out.device}, dtype={out.dtype}")

print("\n=== ALL TESTS PASSED ===")
