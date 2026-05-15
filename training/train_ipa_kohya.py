"""Train IPA with InstantCharacter-style architecture and 3-stage training.

Per arXiv:2504.12395:
- Architecture: SigLIP deep+shallow → CrossLayerEncoder → TimeResampler → IPCrossAttn
- Stage 1 (unpaired): ref→self reconstruction at 512px, learn identity consistency
- Stage 2 (paired): ref+prompt→different scene at 512px, learn text controllability
- Stage 3 (joint): paired+unpaired at 1024px, improve visual fidelity

Currently implements Stage 2 (paired training with text prompts).
"""
import argparse, io, json, os, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file

ROOT = Path("/data/stardust/anima_ipa")
sys.path.insert(0, str(ROOT / "sd-scripts"))
sys.path.insert(0, str(ROOT / "stage7_train/code"))

from library import anima_utils, strategy_anima, strategy_base, hunyuan_image_utils
from ip_adapter_siglip import IPAdapterSigLIP


def _pad_to_square(pil_img, size=512):
    w, h = pil_img.size
    if w == h: return pil_img.resize((size, size))
    from PIL import Image as PILImage
    if w > h: nw, nh = size, int(h*size/w)
    else: nw, nh = int(w*size/h), size
    img = pil_img.resize((nw, nh))
    canvas = PILImage.new("RGB", (size, size), (255,255,255))
    canvas.paste(img, ((size-nw)//2, (size-nh)//2)); return canvas


class TrainDataset(Dataset):
    """Paired dataset: (ref_image, target_image, prompt)."""
    def __init__(self, pairs_path, image_dir, max_pairs=None, resolution=1024):
        self.pairs = []
        with open(pairs_path) as f:
            for line in f: self.pairs.append(json.loads(line))
        if max_pairs: self.pairs = self.pairs[:max_pairs]
        self.image_dir = Path(image_dir)
        self.resolution = resolution

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        from PIL import Image; from torchvision import transforms
        tgt_path = self.image_dir / f"{p['tgt_id']}.jpg"
        R = self.resolution
        img_t = torch.zeros(3, 1, R, R)
        try:
            pil_tgt = Image.open(str(tgt_path)).convert("RGB")
            if pil_tgt is not None and pil_tgt.size[0] > 0:
                img_t = transforms.Compose([
                    transforms.Resize((R, R)), transforms.ToTensor(),
                    transforms.Normalize([0.5],[0.5]),
                ])(pil_tgt).unsqueeze(1)
        except Exception: pass
        ref_path = self.image_dir / f"{p['ref_id']}.jpg"
        ref_bytes = b""
        try:
            with open(str(ref_path), "rb") as f: ref_bytes = f.read()
        except Exception: pass
        return {"ref_img_bytes": ref_bytes, "img": img_t, "prompt": p["prompt"]}


def flow_match_sample(x1, sigma):
    x0 = torch.randn_like(x1)
    sr = sigma.reshape(-1, *([1]*(x1.ndim-1)))
    return (1-sr)*x1 + sr*x0, x0-x1


def train(args):
    dev0 = torch.device("cuda:0"); dev1 = torch.device("cuda:1"); dtype = torch.bfloat16
    if args.wandb: import wandb; wandb.init(project="anima-ipa", config=vars(args))

    # DiT
    print("[1/5] DiT...", flush=True)
    anima = anima_utils.load_anima_model(dev1, f"{ROOT}/stage7_train/models_v1/anima-base-v1.0.safetensors",
        "torch", True, dev1, dtype, False)
    anima = anima.to(dev1, dtype=dtype).eval().requires_grad_(False)

    # TE
    print("[2/5] TE...", flush=True)
    te_model, _ = anima_utils.load_qwen3_text_encoder(
        "/data/stardust/anima_ipa/stage6_emb/models/.hf_cache/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        dtype=dtype, device=dev1); te_model.eval()
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path="/data/stardust/anima_ipa/stage6_emb/models/.hf_cache/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        t5_tokenizer_path=None, qwen3_max_length=512, t5_max_length=512)
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
    encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

    # Pre-compute unconditional text embedding
    print("  Unconditional embedding...", flush=True)
    tn = tokenize_strategy.tokenize("")
    en = encoding_strategy.encode_tokens(tokenize_strategy, [te_model], tn)
    uncond_emb = anima._preprocess_text_embeds(en[0].to(dev1), en[2].to(dev1), en[3].to(dev1), en[1].to(dev1))
    uncond_emb[~en[3].to(dev1).bool()] = 0
    if uncond_emb.shape[1] < 512:
        uncond_emb = F.pad(uncond_emb, (0,0,0,512-uncond_emb.shape[1]))

    # SigLIP
    print("[3/5] SigLIP...", flush=True)
    from transformers import SiglipVisionModel, AutoImageProcessor
    siglip = SiglipVisionModel.from_pretrained("google/siglip2-base-patch16-512",
        torch_dtype=dtype, trust_remote_code=True).to(dev0).eval()
    for p in siglip.parameters(): p.requires_grad_(False)
    siglip_proc = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-512", trust_remote_code=True)

    # VAE
    print("[4/5] VAE...", flush=True)
    from library import qwen_image_autoencoder_kl
    vae = qwen_image_autoencoder_kl.load_vae(
        "/home/stardust/wenaka/ComfyUI/models/vae/qwen_image_vae.safetensors",
        device=dev0, disable_mmap=True); vae.to(dtype).eval()
    for p in vae.parameters(): p.requires_grad_(False)

    # IPA
    print("[5/5] IPA...", flush=True)
    # SigLIP2-base has 26 transformer layers; shallow = layers [7, 13], deep = last_hidden_state
    siglip_num_layers = len(siglip.encoder.layers)
    siglip_shallow_indices = [siglip_num_layers // 4, siglip_num_layers // 2]  # ~layer 7, 13

    ipa = IPAdapterSigLIP(
        siglip_dim=768, siglip_shallow_dim=768,
        dit_dim=2048, num_blocks=28,
        num_queries=32, resampler_depth=4, resampler_heads=16,
        resampler_dim=1024, resampler_dim_head=64,
        intermediate_dim=768, intermediate_layers=4, intermediate_heads=12,
        ip_heads=16, time_embed_dim=320,
        use_intermediate_encoder=True,
    )
    ipa.to(dev1, dtype=dtype)
    print(f"  {sum(p.numel() for p in ipa.parameters())/1e6:.1f}M params", flush=True)

    # Param groups
    resampler_params = [p for n, p in ipa.named_parameters() if n.startswith("resampler.")]
    ipca_params = [p for n, p in ipa.named_parameters() if n.startswith("ip_cross_attns.")]
    ip_scale_params = [p for n, p in ipa.named_parameters() if n.startswith("ip_scales.")]
    inter_params = [p for n, p in ipa.named_parameters() if n.startswith("intermediate_encoder.")]

    opt = torch.optim.AdamW([
        {"params": inter_params, "lr": args.lr},
        {"params": resampler_params, "lr": args.lr},
        {"params": ipca_params, "lr": args.lr},
        {"params": ip_scale_params, "lr": args.lr * 10},
    ], weight_decay=0.01)

    dataset = TrainDataset(args.pairs_path, args.image_dir, max_pairs=args.max_pairs)
    print(f"  {len(dataset):,} pairs", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image as PILImage
    global_step = 0; t0 = time.time(); running_loss = 0.0; n_loss = 0
    data_iter = iter(loader)
    def next_batch():
        nonlocal data_iter
        try: return next(data_iter)
        except StopIteration: data_iter = iter(loader); return next(data_iter)

    ipa.train()
    print(f"\n{args.num_steps} steps, lr={args.lr}, bs={args.batch_size}\n", flush=True)

    init_params = {n: p.data.clone() for n, p in ipa.named_parameters()}

    def extract_siglip_features(pil_images):
        """Extract deep + shallow features from SigLIP2."""
        rp = [_pad_to_square(r, 512) for r in pil_images]
        si = siglip_proc(images=rp, return_tensors="pt", do_resize=False)
        si = {k: v.to(dev0, dtype=dtype) for k, v in si.items()}
        with torch.no_grad():
            outputs = siglip(**si, output_hidden_states=True)
            deep_features = outputs.last_hidden_state.to(dev1, dtype=dtype)
            # Concatenate shallow layer hidden states
            shallow_features = torch.cat([
                outputs.hidden_states[i].to(dev1, dtype=dtype)
                for i in siglip_shallow_indices
            ], dim=1)
        return deep_features, shallow_features

    def sample_image(ref_pils, prompt, step):
        if not ref_pils or ref_pils[0] is None: return None
        try:
            ipa.eval()
            deep_f, shallow_f = extract_siglip_features([ref_pils[0]])
            tokens = tokenize_strategy.tokenize(prompt)
            emb = encoding_strategy.encode_tokens(tokenize_strategy, [te_model], tokens)
            ce = anima._preprocess_text_embeds(emb[0].to(dev1), emb[2].to(dev1), emb[3].to(dev1), emb[1].to(dev1))
            ce[~emb[3].to(dev1).bool()] = 0
            cn = uncond_emb.clone()
            n_steps = 20; cfg_v = 4.5
            _s_orig = {}
            cur_tokens = [None]
            for idx, blk in enumerate(anima.blocks):
                _s_orig[idx] = blk.cross_attn.forward; bi = idx
                def _smf(ca_mod=blk.cross_attn, bidx=bi, ipa_mod=ipa):
                    of = ca_mod.forward
                    def _sf(x, attn_params, context=None, rope_emb=None):
                        o = of(x, attn_params, context, rope_emb)
                        return o + ipa_mod.forward_block(bidx, x, cur_tokens[0], scale_override=None).to(o.dtype)
                    return _sf
                blk.cross_attn.forward = _smf()
            tsteps, sigs = hunyuan_image_utils.get_timesteps_sigmas(n_steps, 5.0, dev1)
            tsteps = tsteps.to(dev1, dtype=dtype) / 1000
            h, w = 1024//8, 1024//8
            lat = torch.randn(1, 16, 1, h, w, device=dev1, dtype=dtype)
            pm = torch.zeros(1, 1, h, w, dtype=dtype, device=dev1)
            with torch.no_grad():
                for i, t_ in enumerate(tsteps):
                    ts = t_.expand(1).reshape(1, 1)
                    cur_tokens[0] = ipa.encode_ref(deep_f, timestep=ts.flatten(), shallow_features=shallow_f)
                    npc = anima(lat, ts, ce, padding_mask=pm)
                    npu = anima(lat, ts, cn, padding_mask=pm)
                    lat = hunyuan_image_utils.step(lat, npu+cfg_v*(npc-npu), sigs, i).to(lat.dtype)
            for idx, blk in enumerate(anima.blocks):
                if idx in _s_orig: blk.cross_attn.forward = _s_orig[idx]
            vae.to("cpu")
            with torch.no_grad():
                dec = vae.decode_to_pixels(lat.cpu().to(dtype=dtype))
            vae.to(dev0)
            if dec.ndim == 5: dec = dec.squeeze(2)
            img = dec[0].float().clamp(-1,1)
            img = ((img+1)*127.5).clamp(0,255).to(torch.uint8).cpu().numpy().transpose(1,2,0)
            ipa.train()
            return PILImage.fromarray(img)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [sample] err: {e}", flush=True); ipa.train(); return None

    while global_step < args.num_steps:
        batch = next_batch()
        B = len(batch["prompt"]); prompts = batch["prompt"]; ref_bytes_list = batch["ref_img_bytes"]

        # 1. SigLIP deep + shallow features
        ref_pils = [PILImage.open(io.BytesIO(rb)).convert("RGB") if rb
                    else PILImage.new("RGB", (512,512), (128,128,128)) for rb in ref_bytes_list]
        deep_features, shallow_features = extract_siglip_features(ref_pils)

        # 2. VAE encode
        with torch.no_grad():
            x1 = vae.encode_pixels_to_latents(batch["img"].to(dev0, dtype=dtype)).to(dev1, dtype=dtype)

        # 3. Flow match
        sigma = torch.rand(B, device=dev1, dtype=dtype)
        xt, target = flow_match_sample(x1, sigma)

        # 4. Encode ref with timestep + shallow features
        image_tokens = ipa.encode_ref(deep_features, timestep=sigma, shallow_features=shallow_features)

        # Image token dropout 10%
        img_drop_mask = (torch.rand(B, device=dev1) > 0.1).to(dtype).view(B, 1, 1)
        image_tokens = image_tokens * img_drop_mask

        # 5. Text with 10% unconditional dropout
        text_embeds = []
        for p in prompts:
            tokens = tokenize_strategy.tokenize(p)
            emb = encoding_strategy.encode_tokens(tokenize_strategy, [te_model], tokens)
            ce = anima._preprocess_text_embeds(emb[0].to(dev1), emb[2].to(dev1), emb[3].to(dev1), emb[1].to(dev1))
            ce[~emb[3].to(dev1).bool()] = 0; text_embeds.append(ce)
        text_emb = torch.cat(text_embeds, dim=0).to(dev1, dtype=dtype)
        text_drop = (torch.rand(B, device=dev1) > 0.1).to(dtype).view(B, 1, 1)
        text_emb = text_emb * text_drop + uncond_emb.expand_as(text_emb) * (1 - text_drop)
        if text_emb.shape[1] < 512: text_emb = F.pad(text_emb, (0,0,0,512-text_emb.shape[1]))

        # 6. Forward with IPA
        ts = sigma.unsqueeze(1)
        pad_mask = torch.zeros(B, 1, xt.shape[-2], xt.shape[-1], device=dev1, dtype=dtype)

        _ipa_ca_orig = {}
        for idx, block in enumerate(anima.blocks):
            ca = block.cross_attn
            _ipa_ca_orig[idx] = ca.forward
            bi = idx
            def make_fwd(ca_mod=ca, bidx=bi, ip=ipa, tok=image_tokens):
                orig_fwd = ca_mod.forward
                def new_fwd(x, attn_params, context=None, rope_emb=None):
                    out = orig_fwd(x, attn_params, context, rope_emb)
                    ip_out = ip.forward_block(bidx, x, tok, scale_override=None)
                    return out + ip_out.to(out.dtype)
                return new_fwd
            ca.forward = make_fwd()

        anima.requires_grad_(True)
        pred = anima(xt, ts, text_emb, padding_mask=pad_mask)

        for idx, block in enumerate(anima.blocks):
            if idx in _ipa_ca_orig:
                block.cross_attn.forward = _ipa_ca_orig[idx]
        anima.requires_grad_(False)

        # 7. Loss
        loss = F.mse_loss(pred.float(), target.float())
        opt.zero_grad()
        loss.backward()
        for p in anima.parameters():
            if p.grad is not None:
                p.grad = None
        torch.nn.utils.clip_grad_norm_(
            inter_params + resampler_params + ipca_params + ip_scale_params, 1.0)
        opt.step()

        global_step += 1; running_loss += loss.item(); n_loss += 1
        if args.wandb: wandb.log({"loss": loss.item(), "step": global_step})

        if global_step % 100 == 1:
            grad_norms = {}
            for n, p in ipa.named_parameters():
                if p.grad is not None:
                    key = n.split(".")[0]
                    grad_norms[key] = grad_norms.get(key, 0) + p.grad.norm().item()
            for k, v in grad_norms.items():
                print(f"  [grad] {k}: {v:.4f}", flush=True)
            delta = {}
            for n, p in ipa.named_parameters():
                if n in init_params:
                    d = (p.data.float() - init_params[n].float()).norm().item()
                    key = n.split(".")[0]
                    delta[key] = delta.get(key, 0) + d
            for k, v in delta.items():
                print(f"  [delta] {k}: {v:.4f}", flush=True)
            scales = [p.item() for p in ip_scale_params]
            print(f"  [scales] min={min(scales):.4f} max={max(scales):.4f} mean={sum(scales)/len(scales):.4f}", flush=True)

        if global_step % args.log_every == 0:
            print(f"step {global_step}/{args.num_steps} | loss={running_loss/n_loss:.4f} | "
                  f"rate={global_step/(time.time()-t0):.2f} steps/s", flush=True)
            running_loss = 0.0; n_loss = 0
        if global_step % args.save_every == 0:
            sf_path = output_dir / f"ipa_step{global_step}.safetensors"
            save_file(ipa.state_dict(), str(sf_path))
            print(f"Saved: {sf_path}", flush=True)
        if args.wandb and global_step % 100 == 0:
            gen = sample_image(ref_pils, prompts[0], global_step)
            if gen:
                try:
                    rf = ref_pils[0].resize((512,512))
                    combined = PILImage.new("RGB", (1024,512))
                    combined.paste(rf, (0,0)); combined.paste(gen.resize((512,512)), (512,0))
                    sample_dir = output_dir / "samples"
                    sample_dir.mkdir(exist_ok=True)
                    combined.save(sample_dir / f"step{global_step:06d}.png")
                    with open(sample_dir / f"step{global_step:06d}.txt", "w") as f: f.write(prompts[0])
                    wandb.log({"sample/ref_gen": wandb.Image(combined, caption=prompts[0]),
                               "step": global_step})
                except Exception: pass
        if global_step >= args.num_steps: break
    print(f"\nDone in {(time.time()-t0)/60:.1f}m", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-path", default=f"{ROOT}/dataset_build/stage4_out/training_pairs_final2.jsonl")
    p.add_argument("--image-dir", default=f"{ROOT}/stage7_train/images_ar")
    p.add_argument("--output-dir", default=f"{ROOT}/stage7_train/out_ipa_v3")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--wandb", action="store_true")
    train(p.parse_args())

if __name__ == "__main__": main()
