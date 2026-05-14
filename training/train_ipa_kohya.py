"""Train IPA (cross-attn injection) with kohya-ss model class."""
import argparse, io, json, os, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import save_file
from einops import rearrange

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
    def __init__(self, pairs_path, image_dir, max_pairs=None):
        self.pairs = []
        with open(pairs_path) as f:
            for line in f: self.pairs.append(json.loads(line))
        if max_pairs: self.pairs = self.pairs[:max_pairs]
        self.image_dir = Path(image_dir)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]
        from PIL import Image; from torchvision import transforms
        tgt_path = self.image_dir / f"{p['tgt_id']}.jpg"
        img_t = torch.zeros(3, 1, 1024, 1024)
        try:
            pil_tgt = Image.open(str(tgt_path)).convert("RGB")
            if pil_tgt is not None and pil_tgt.size[0] > 0:
                img_t = transforms.Compose([
                    transforms.Resize((1024,1024)), transforms.ToTensor(),
                    transforms.Normalize([0.5],[0.5]),
                ])(pil_tgt).unsqueeze(1)
        except Exception: pass
        ref_path = self.image_dir / f"{p['ref_id']}.jpg"
        ref_bytes = b""
        try:
            with open(str(ref_path), "rb") as f: ref_bytes = f.read()
        except Exception: pass
        return {"ref_img_bytes": ref_bytes, "img": img_t, "prompt": p["prompt"]}


def flow_match_sample(x1, t):
    x0 = torch.randn_like(x1)
    tr = t.reshape(-1, *([1]*(x1.ndim-1)))
    return (1-tr)*x0 + tr*x1, x1-x0


def train(args):
    dev0 = torch.device("cuda:0"); dev1 = torch.device("cuda:1"); dtype = torch.bfloat16
    if args.wandb: import wandb; wandb.init(project="anima-ipa", config=vars(args))

    # DiT
    print("[1/5] DiT...", flush=True)
    anima = anima_utils.load_anima_model(dev1, f"{ROOT}/stage7_train/models/anima-preview3-base.safetensors",
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
    ipa = IPAdapterSigLIP(input_dim=768, dit_dim=2048, num_blocks=28,
        num_queries=32, resampler_depth=2, resampler_heads=16,
        resampler_internal=1024, ip_heads=16, ip_bottleneck=256)
    ipa.to(dev1, dtype=dtype)
    print(f"  {sum(p.numel() for p in ipa.parameters())/1e6:.1f}M", flush=True)

    # Separate optimizer: ip_scales get ultra-low LR (they're 28 scalars, grow too fast)
    ip_scale_params = list(ipa.ip_scales)
    other_params = [p for n, p in ipa.named_parameters() if not n.startswith("ip_scales")]
    opt = torch.optim.AdamW([
        {"params": other_params, "lr": args.lr},
        {"params": ip_scale_params, "lr": 1e-5},  # 100x smaller
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

    def sample_image(ref_pils, prompt, step):
        if not ref_pils or ref_pils[0] is None: return None
        try:
            ipa.eval()
            rp = [_pad_to_square(ref_pils[0], 512)]
            si = siglip_proc(images=rp, return_tensors="pt", do_resize=False)
            si = {k: v.to(dev0, dtype=dtype) for k, v in si.items()}
            with torch.no_grad(): sf = siglip(**si).last_hidden_state.to(dev1, dtype=dtype)
            it = ipa.encode_ref(sf)
            # Text
            tokens = tokenize_strategy.tokenize(prompt)
            emb = encoding_strategy.encode_tokens(tokenize_strategy, [te_model], tokens)
            ce = anima._preprocess_text_embeds(emb[0].to(dev1), emb[2].to(dev1), emb[3].to(dev1), emb[1].to(dev1))
            ce[~emb[3].to(dev1).bool()] = 0
            # Encode negative
            tn = tokenize_strategy.tokenize("")
            en = encoding_strategy.encode_tokens(tokenize_strategy, [te_model], tn)
            cn = anima._preprocess_text_embeds(en[0].to(dev1), en[2].to(dev1), en[3].to(dev1), en[1].to(dev1))
            # Sample — patch once, run all steps, restore once
            n_steps = 20; cfg_v = 4.5
            tsteps, sigs = hunyuan_image_utils.get_timesteps_sigmas(n_steps, 5.0, dev1)
            tsteps = tsteps.to(dev1, dtype=dtype) / 1000
            h, w = 1024//8, 1024//8
            lat = torch.randn(1, 16, 1, h, w, device=dev1, dtype=dtype)
            pm = torch.zeros(1, 1, h, w, dtype=dtype, device=dev1)
            # Patch once before sampling loop
            _ipa_sample_orig = {}
            for idx, blk in enumerate(anima.blocks):
                _ipa_sample_orig[idx] = blk.cross_attn.forward
                bi = idx
                def _make_fwd2(ca_mod=blk.cross_attn, bidx=bi, ipa_mod=ipa, tok=it):
                    of = ca_mod.forward
                    def _fwd2(x, attn_params, context=None, rope_emb=None):
                        o = of(x, attn_params, context, rope_emb)
                        ipo = ipa_mod.forward_block(bidx, x, tok)
                        return o + ipo.to(o.dtype)
                    return _fwd2
                blk.cross_attn.forward = _make_fwd2()
            with torch.no_grad():
                for i, t_ in enumerate(tsteps):
                    ts = t_.expand(1).reshape(1, 1)
                    npc = anima(lat, ts, ce, padding_mask=pm)
                    npu = anima(lat, ts, cn, padding_mask=pm)
                    lat = hunyuan_image_utils.step(lat, npu+cfg_v*(npc-npu), sigs, i).to(lat.dtype)
            # Restore
            for idx, blk in enumerate(anima.blocks):
                if idx in _ipa_sample_orig:
                    blk.cross_attn.forward = _ipa_sample_orig[idx]
            dec = vae.decode_to_pixels(lat.to(dev0, dtype=dtype))
            if dec.ndim == 5: dec = dec.squeeze(2)
            img = dec[0].float().clamp(-1,1)
            img = ((img+1)*127.5).clamp(0,255).to(torch.uint8).cpu().numpy().transpose(1,2,0)
            ipa.train()
            return PILImage.fromarray(img)
        except Exception as e:
            print(f"  [sample] err: {e}", flush=True); ipa.train(); return None

    _ipa_ca_orig = {}
    def _ipa_patch(blocks, ip_adapter, im_tokens):
        for idx, block in enumerate(blocks):
            ca = block.cross_attn
            if idx not in _ipa_ca_orig:
                _ipa_ca_orig[idx] = ca.forward
            bi = idx
            def make_fwd(ca_mod=ca, bidx=bi, ip=ip_adapter, tok=im_tokens):
                orig_fwd = ca_mod.forward
                def new_fwd(x, attn_params, context=None, rope_emb=None):
                    out = orig_fwd(x, attn_params, context, rope_emb)
                    ip_out = ip.forward_block(bidx, x, tok)
                    return out + ip_out.to(out.dtype)
                return new_fwd
            ca.forward = make_fwd()
    def _ipa_restore(blocks):
        for idx, block in enumerate(blocks):
            if idx in _ipa_ca_orig:
                block.cross_attn.forward = _ipa_ca_orig[idx]

    while global_step < args.num_steps:
        batch = next_batch()
        B = len(batch["prompt"]); prompts = batch["prompt"]; ref_bytes_list = batch["ref_img_bytes"]

        # 1. SigLIP
        ref_pils = [PILImage.open(io.BytesIO(rb)).convert("RGB") if rb
                    else PILImage.new("RGB", (512,512), (128,128,128)) for rb in ref_bytes_list]
        rp = [_pad_to_square(r, 512) for r in ref_pils]
        si = siglip_proc(images=rp, return_tensors="pt", do_resize=False)
        si = {k: v.to(dev0, dtype=dtype) for k, v in si.items()}
        with torch.no_grad(): sf = siglip(**si).last_hidden_state.to(dev1, dtype=dtype)
        image_tokens = ipa.encode_ref(sf)

        # 2. VAE
        with torch.no_grad():
            x1 = vae.encode(batch["img"].to(dev0, dtype=dtype))["latent_dist"].sample().to(dev1, dtype=dtype)

        # 3. Flow match
        t = torch.rand(B, device=dev1, dtype=dtype)
        xt, target = flow_match_sample(x1, t)

        # 4. Text
        text_embeds = []
        for p in prompts:
            tokens = tokenize_strategy.tokenize(p)
            emb = encoding_strategy.encode_tokens(tokenize_strategy, [te_model], tokens)
            ce = anima._preprocess_text_embeds(emb[0].to(dev1), emb[2].to(dev1), emb[3].to(dev1), emb[1].to(dev1))
            ce[~emb[3].to(dev1).bool()] = 0; text_embeds.append(ce)
        text_emb = torch.cat(text_embeds, dim=0).to(dev1, dtype=dtype)
        # Text dropout: 20% chance to zero out text, forcing IPA to carry info
        drop_mask = (torch.rand(B, device=dev1) > 0.2).to(dtype).view(B, 1, 1)
        text_emb = text_emb * drop_mask
        if text_emb.shape[1] < 512: text_emb = F.pad(text_emb, (0,0,0,512-text_emb.shape[1]))

        # 5. Forward with IPA
        _ipa_patch(anima.blocks, ipa, image_tokens)
        ts = (t*1000).unsqueeze(1)
        pad_mask = torch.zeros(B, 1, xt.shape[-2], xt.shape[-1], device=dev1, dtype=dtype)
        pred = anima(xt, ts, text_emb, padding_mask=pad_mask)
        _ipa_restore(anima.blocks)

        # 6. Loss
        loss = F.mse_loss(pred.float(), target.float())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ipa.parameters(), 1.0)
        opt.step()

        global_step += 1; running_loss += loss.item(); n_loss += 1
        if args.wandb: wandb.log({"loss": loss.item(), "step": global_step})
        if global_step % args.log_every == 0:
            print(f"step {global_step}/{args.num_steps} | loss={running_loss/n_loss:.4f} | "
                  f"rate={global_step/(time.time()-t0):.2f} steps/s", flush=True)
            running_loss = 0.0; n_loss = 0
        if global_step % args.save_every == 0:
            sf_path = output_dir / f"ipa_step{global_step}.safetensors"
            save_file(ipa.state_dict(), str(sf_path))
            print(f"Saved: {sf_path}", flush=True)
        if args.wandb and global_step % 200 == 0:
            gen = sample_image(ref_pils, prompts[0], global_step)
            if gen:
                try:
                    rf = ref_pils[0].resize((512,512))
                    combined = PILImage.new("RGB", (1024,512))
                    combined.paste(rf, (0,0)); combined.paste(gen.resize((512,512)), (512,0))
                    wandb.log({"sample/ref_gen": wandb.Image(combined, caption=f"step{global_step} {prompts[0][:50]}"),
                               "step": global_step})
                except Exception: pass
        if global_step >= args.num_steps: break
    print(f"\nDone in {(time.time()-t0)/60:.1f}m", flush=True)


def _block_ipa(orig_fwd, block, block_idx, ip_adapter, image_tokens,
               x_B_T_H_W_D, emb_B_T_D, crossattn_emb, **kwargs):
    """Block forward with IPA cross-attn injection."""
    residual_dtype = x_B_T_H_W_D.dtype; compute_dtype = emb_B_T_D.dtype
    rope_emb = kwargs.get("rope_emb_L_1_1_D", None)
    adaln_lora = kwargs.get("adaln_lora_B_T_3D", None)
    extra_pos_emb = kwargs.get("extra_per_block_pos_emb", None)
    use_fp32 = kwargs.get("use_fp32", False)

    if extra_pos_emb is not None: x_B_T_H_W_D = x_B_T_H_W_D + extra_pos_emb
    B, T, H, W, D = x_B_T_H_W_D.shape

    # Self-attn
    s1, sc1, g1 = (block.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora).chunk(3, dim=-1)
    def _fn(_x, _norm, _scale, _shift):
        return _norm(_x)*(1+rearrange(_scale,"b t d -> b t 1 1 d"))+rearrange(_shift,"b t d -> b t 1 1 d")
    normed = _fn(x_B_T_H_W_D, block.layer_norm_self_attn, sc1, s1)
    result = rearrange(block.self_attn(
        rearrange(normed.to(compute_dtype),"b t h w d -> b (t h w) d"),
        None, rope_emb=rope_emb),
        "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g1,"b t d -> b t 1 1 d").to(residual_dtype)*result.to(residual_dtype)

    # Cross-attn (text) + IPA
    s2, sc2, g2 = (block.adaln_modulation_cross_attn(emb_B_T_D)+adaln_lora).chunk(3,dim=-1)
    normed = _fn(x_B_T_H_W_D, block.layer_norm_cross_attn, sc2, s2)
    nf = rearrange(normed.to(compute_dtype),"b t h w d -> b (t h w) d")
    tr = rearrange(block.cross_attn(nf, crossattn_emb, rope_emb=rope_emb),
                   "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    # IPA injection
    ip_out = ip_adapter.forward_block(block_idx, nf, image_tokens)
    ip_out = rearrange(ip_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
    gate = rearrange(g2,"b t d -> b t 1 1 d")
    x_B_T_H_W_D = gate.to(residual_dtype)*(tr.to(residual_dtype)+ip_out.to(residual_dtype))+x_B_T_H_W_D

    # MLP
    s3, sc3, g3 = (block.adaln_modulation_mlp(emb_B_T_D)+adaln_lora).chunk(3,dim=-1)
    normed = _fn(x_B_T_H_W_D, block.layer_norm_mlp, sc3, s3)
    result = block.mlp(normed.to(compute_dtype))
    x_B_T_H_W_D = x_B_T_H_W_D + rearrange(g3,"b t d -> b t 1 1 d").to(residual_dtype)*result.to(residual_dtype)
    return x_B_T_H_W_D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-path", default=f"{ROOT}/dataset_build/stage4_out/training_pairs_final2.jsonl")
    p.add_argument("--image-dir", default=f"{ROOT}/stage7_train/images_ar")
    p.add_argument("--output-dir", default=f"{ROOT}/stage7_train/out_ipa")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--wandb", action="store_true")
    train(p.parse_args())

if __name__ == "__main__": main()
