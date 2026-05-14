"""Test IPA (cross-attn) with UNTRAINED weights, kohya pipeline, patch cross_attn."""
import sys, argparse, os, json, random
from pathlib import Path
ROOT = Path("/data/stardust/anima_ipa")
sys.path.insert(0, str(ROOT / "sd-scripts"))
sys.path.insert(0, str(ROOT / "stage7_train/code"))

import torch
torch.manual_seed(42)
dtype = torch.bfloat16; dev = torch.device("cuda:0")

from library import anima_utils, strategy_anima, strategy_base, hunyuan_image_utils, qwen_image_autoencoder_kl
from ip_adapter_siglip import IPAdapterSigLIP

anima = anima_utils.load_anima_model(dev,
    f"{ROOT}/stage7_train/models/anima-preview3-base.safetensors",
    "torch", True, dev, dtype, False)
anima = anima.to(dev, dtype=dtype).eval().requires_grad_(False)

te_model, _ = anima_utils.load_qwen3_text_encoder(
    "/data/stardust/anima_ipa/stage6_emb/models/.hf_cache/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    dtype=dtype, device=dev); te_model.eval()

tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
    qwen3_path="/data/stardust/anima_ipa/stage6_emb/models/.hf_cache/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    t5_tokenizer_path=None, qwen3_max_length=512, t5_max_length=512)
strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

from transformers import SiglipVisionModel, AutoImageProcessor
siglip = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-512",
    torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
siglip_proc = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-512", trust_remote_code=True)

ipa = IPAdapterSigLIP(input_dim=768, dit_dim=2048, num_blocks=28,
    num_queries=32, resampler_depth=2, resampler_heads=16,
    resampler_internal=1024, ip_heads=16, ip_bottleneck=256)
ipa.to(dev, dtype=dtype).eval()
print(f"IPA: {sum(p.numel() for p in ipa.parameters())/1e6:.1f}M")
print(f"ip_scales: {[s.item() for s in ipa.ip_scales[:3]]}...")

from PIL import Image as PILImage
img_dir = ROOT / "stage7_train/images_ar"
def _pad_to_square(pil_img, size=512):
    w, h = pil_img.size
    if w == h: return pil_img.resize((size, size))
    if w > h: nw, nh = size, int(h*size/w)
    else: nw, nh = int(w*size/h), size
    img = pil_img.resize((nw, nh))
    canvas = PILImage.new("RGB", (size, size), (255,255,255))
    canvas.paste(img, ((size-nw)//2, (size-nh)//2)); return canvas

pairs = []
with open(f"{ROOT}/dataset_build/stage4_out/training_pairs_final2.jsonl") as f:
    for line in f:
        p = json.loads(line)
        if (img_dir / f"{p['ref_id']}.jpg").exists(): pairs.append(p)
        if len(pairs) >= 100: break
s = random.choice(pairs)
ref_pil = PILImage.open(img_dir / f"{s['ref_id']}.jpg").convert("RGB")
rp = _pad_to_square(ref_pil, 512)
si = siglip_proc(images=rp, return_tensors="pt", do_resize=False)
si = {k: v.to(dev, dtype=dtype) for k, v in si.items()}
with torch.no_grad(): sf = siglip(**si).last_hidden_state.to(dev, dtype=dtype)
image_tokens = ipa.encode_ref(sf)
print(f"Prompt: {s['prompt'][:60]}")

class FA: pass
fa = FA(); fa.prompt = s["prompt"]; fa.negative_prompt = ""; fa.text_encoder_cpu = False
fa.lora_weight = None; fa.lora_multiplier = 1.0
fa.text_encoder = "/data/stardust/anima_ipa/stage6_emb/models/.hf_cache/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
from anima_minimal_inference import prepare_text_inputs
ctx, ctx_null = prepare_text_inputs(fa, dev, anima)
embed = ctx["embed"][0].to(dev, dtype=dtype)
neg_embed = ctx_null["embed"][0].to(dev, dtype=dtype)

# Patch cross_attn to inject IPA output
_ca_orig = {}
_ca_idx = {}
for idx, block in enumerate(anima.blocks):
    ca = block.cross_attn
    _ca_orig[idx] = ca.forward
    _ca_idx[idx] = idx
    bi = idx
    def make_fwd(ca_mod=ca, bidx=bi, ip=ipa, tok=image_tokens):
        orig_fwd = ca_mod.forward
        _debug_count = [0]
        def new_fwd(x, attn_params, context=None, rope_emb=None):
            out = orig_fwd(x, attn_params, context, rope_emb)
            ip_out = ip.forward_block(bidx, x, tok)
            if _debug_count[0] == 0 and bidx == 0:
                print(f"  [DEBUG] block0 ip_out norm={ip_out.norm():.6f}, out norm={out.norm():.6f}, ip_scale={ip.ip_scales[bidx].item():.6f}", flush=True)
                _debug_count[0] += 1
            return out + ip_out.to(out.dtype)
        return new_fwd
    ca.forward = make_fwd()

# Sample
n_steps = 30; flow_shift = 5.0; cfg = 4.5
timesteps, sigmas = hunyuan_image_utils.get_timesteps_sigmas(n_steps, flow_shift, dev)
timesteps /= 1000; timesteps = timesteps.to(dev, dtype=dtype)
h, w = 1024//8, 1024//8
latents = torch.randn(1, 16, 1, h, w, device=dev, dtype=dtype)
pad_mask = torch.zeros(1, 1, h, w, dtype=dtype, device=dev)

print("Sampling...", flush=True)
with torch.no_grad():
    for i, t in enumerate(timesteps):
        ts = t.expand(1).reshape(1, 1)
        npc = anima(latents, ts, embed, padding_mask=pad_mask)
        npu = anima(latents, ts, neg_embed, padding_mask=pad_mask)
        latents = hunyuan_image_utils.step(latents, npu+cfg*(npc-npu), sigmas, i).to(latents.dtype)

# Restore
for idx, block in enumerate(anima.blocks):
    if idx in _ca_orig:
        block.cross_attn.forward = _ca_orig[idx]

# Decode
from anima_minimal_inference import decode_latent, save_images
vae = qwen_image_autoencoder_kl.load_vae(
    "/home/stardust/wenaka/ComfyUI/models/vae/qwen_image_vae.safetensors",
    device="cpu", disable_mmap=True); vae.to(dtype).eval()
out_dir = f"{ROOT}/stage7_train/eval_ipa_init"
os.makedirs(out_dir, exist_ok=True)
ref_pil.save(f"{out_dir}/ref.png")
pixels = decode_latent(vae, latents.cpu(), torch.device("cpu"))
save_images(pixels, argparse.Namespace(save_path=out_dir, seed=42, prompt="test", negative_prompt=""))
print(f"Done: {out_dir}", flush=True)
