"""Kohya-ss pipeline + CharLLLite. Test with multiplier=0 (baseline)."""
import sys, argparse, os
from pathlib import Path
ROOT = Path("/data/stardust/anima_ipa")
sys.path.insert(0, str(ROOT / "sd-scripts"))
sys.path.insert(0, str(ROOT / "stage7_train/code"))

import torch
torch.manual_seed(42)
dtype = torch.bfloat16; dev = torch.device("cuda:0")

from library import anima_models, anima_utils, hunyuan_image_utils, strategy_anima, strategy_base, qwen_image_autoencoder_kl

tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
    qwen3_path="/data/stardust/anima_ipa/stage6_emb/models/.hf_cache/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    t5_tokenizer_path=None, qwen3_max_length=512, t5_max_length=512)
strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

anima = anima_utils.load_anima_model(dev,
    f"{ROOT}/stage7_train/models/anima-preview3-base.safetensors",
    "torch", True, dev, dtype, False)
anima = anima.to(dev, dtype=dtype)
anima.eval().requires_grad_(False)

te_model, _ = anima_utils.load_qwen3_text_encoder(
    "/data/stardust/anima_ipa/stage6_emb/models/.hf_cache/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    dtype=dtype, device=dev)
te_model.eval()

# ─── CharLLLite ───
from char_lllite import CharLLLite
lllite = CharLLLite(anima, siglip_dim=768, cond_dim=32, mlp_dim=64)
lllite.to(dev, dtype=dtype).eval()
lllite.set_multiplier(0.0)  # baseline: no LLLite effect

# Create a dummy cond for testing
dummy_cond = torch.zeros(1, 1, 32, device=dev, dtype=dtype)
lllite.set_cond(dummy_cond)

print(f"LLLite blocks: {len(lllite.lllite_mods)}")
print(f"up.weight[0] norm: {lllite.lllite_mods[0].up.weight.norm():.6f}")

# ─── Text encoding ───
class FakeArgs:
    prompt = "1girl, masterpiece, white hair, blue eyes, school uniform, smile"
    negative_prompt = ""
    text_encoder = "/data/stardust/anima_ipa/stage6_emb/models/.hf_cache/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
    text_encoder_cpu = False
    lora_weight = None; lora_multiplier = 1.0

from anima_minimal_inference import prepare_text_inputs
context, context_null = prepare_text_inputs(FakeArgs(), dev, anima)
embed = context["embed"][0].to(dev, dtype=dtype)
negative_embed = context_null["embed"][0].to(dev, dtype=dtype)

# ─── Inference with LLLite ───
n_steps = 30; flow_shift = 5.0; cfg = 4.5
timesteps, sigmas = hunyuan_image_utils.get_timesteps_sigmas(n_steps, flow_shift, dev)
timesteps /= 1000; timesteps = timesteps.to(dev, dtype=dtype)

h, w = 1024//8, 1024//8
latents = torch.randn(1, 16, 1, h, w, device=dev, dtype=dtype)
padding_mask = torch.zeros(1, 1, h, w, dtype=dtype, device=dev)

print("Sampling with LLLite...", flush=True)
do_cfg = cfg != 1.0
with torch.no_grad():
    for i, t in enumerate(timesteps):
        lllite.apply_to()
        noise_pred = anima(latents, t.expand(1), embed, padding_mask=padding_mask)
        lllite.restore()
        if do_cfg:
            lllite.apply_to()
            uncond_noise_pred = anima(latents, t.expand(1), negative_embed, padding_mask=padding_mask)
            lllite.restore()
            noise_pred = uncond_noise_pred + cfg * (noise_pred - uncond_noise_pred)
        latents = hunyuan_image_utils.step(latents, noise_pred, sigmas, i).to(latents.dtype)

# ─── VAE decode ───
from anima_minimal_inference import decode_latent, save_images
vae_for_decode = qwen_image_autoencoder_kl.load_vae(
    "/home/stardust/wenaka/ComfyUI/models/vae/qwen_image_vae.safetensors",
    device="cpu", disable_mmap=True)
vae_for_decode.to(dtype).eval()

out_dir = f"{ROOT}/stage7_train/eval_lllite_baseline"
os.makedirs(out_dir, exist_ok=True)
pixels = decode_latent(vae_for_decode, latents, dev)
save_images(pixels, argparse.Namespace(save_path=out_dir, seed=42, prompt="test", negative_prompt=""))
print(f"Done: {out_dir}", flush=True)
