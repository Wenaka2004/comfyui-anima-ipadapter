"""Diagnose: does the resampler produce different outputs for different refs?"""
import torch, json, sys
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, "/data/stardust/anima_ipa/comfyui-anima-ipadapter")
from ip_adapter_model import IPAdapter
from safetensors.torch import load_file

sd = load_file("/data/stardust/anima_ipa/stage7_train/out/ip_adapter_step9000.safetensors")
ip = IPAdapter(emb_dim=1024, x_dim=2048, num_blocks=28, ip_dim=512, num_ip_heads=8, ip_head_dim=64, num_tokens=16, num_perceiver_layers=2, num_perceiver_heads=4)
ip.load_state_dict(sd)
ip.cuda().bfloat16().eval()

# Load 5 real VL embeddings
emb_dir = Path("/data/stardust/anima_ipa/stage6_emb/out")
pairs = []
with open("/data/stardust/anima_ipa/dataset_build/stage4_out/training_pairs_final2.jsonl") as f:
    for line in f:
        pairs.append(json.loads(line))
unique_ids = []
seen = set()
for p in pairs:
    if p["ref_id"] not in seen:
        unique_ids.append(p["ref_id"])
        seen.add(p["ref_id"])
    if len(unique_ids) >= 5:
        break

embs = {}
for shard in sorted(emb_dir.glob("emb_shard_*.parquet")):
    t = pq.read_table(shard, columns=["id", "emb"])
    for i, e in zip(t.column("id").to_pylist(), t.column("emb").to_pylist()):
        if i in set(unique_ids):
            embs[i] = torch.tensor(e, dtype=torch.float16)
    if len(embs) >= 5:
        break

# 1. Raw VL embedding pairwise cosine similarity
print("=== Raw VL embeddings (before resampler) ===")
raw = []
for idx, rid in enumerate(unique_ids[:5]):
    e = embs.get(rid)
    if e is None:
        continue
    raw.append(e.cuda().to(torch.bfloat16))
    print(f"  Ref{idx}: norm={e.norm():.4f}")

for i in range(len(raw)):
    for j in range(i + 1, len(raw)):
        cs = torch.nn.functional.cosine_similarity(raw[i].flatten().unsqueeze(0), raw[j].flatten().unsqueeze(0)).item()
        print(f"  Raw Ref{i} vs Ref{j}: cos_sim={cs:.4f}")

# 2. Resampler output
print("\n=== Resampler output (image_tokens) ===")
all_tok = []
for idx, rid in enumerate(unique_ids[:5]):
    e = embs.get(rid)
    if e is None:
        continue
    inp = e.unsqueeze(0).unsqueeze(0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        tok = ip.resample(inp)
    all_tok.append(tok)
    print(f"  Ref{idx}: tokens norm={tok.norm():.4f}, std={tok.std():.6f}")

for i in range(len(all_tok)):
    for j in range(i + 1, len(all_tok)):
        cs = torch.nn.functional.cosine_similarity(all_tok[i].flatten().unsqueeze(0), all_tok[j].flatten().unsqueeze(0)).item()
        print(f"  Tokens Ref{i} vs Ref{j}: cos_sim={cs:.4f}")

# 3. Latent dominance
print("\n=== Latent dominance check ===")
latents = ip.resampler.latents.data
print(f"  Latents norm={latents.norm():.4f}, std={latents.std():.6f}")
print(f"  Latents mean_abs={latents.abs().mean():.6f}")
if len(all_tok) > 0:
    print(f"  Tokens mean_abs={all_tok[0].abs().mean():.6f}")
    cs = torch.nn.functional.cosine_similarity(latents.flatten().unsqueeze(0), all_tok[0].flatten().cpu().unsqueeze(0)).item()
    print(f"  Latents vs Tokens[0] cos_sim={cs:.4f}")

# 4. Zero embedding
print("\n=== Zero embedding resampler output ===")
with torch.no_grad():
    zero_tok = ip.resample(torch.zeros(1, 1, 1024, device="cuda", dtype=torch.bfloat16))
print(f"  Zero tokens norm={zero_tok.norm():.4f}")
if len(all_tok) > 0:
    cs = torch.nn.functional.cosine_similarity(zero_tok.flatten().unsqueeze(0), all_tok[0].flatten().unsqueeze(0)).item()
    print(f"  Zero vs Ref0 tokens cos_sim={cs:.4f}")

# 5. Per-layer check: how much does cross-attn change the latents?
print("\n=== Per-layer resampler info flow ===")
with torch.no_grad():
    for ref_idx in [0]:
        e = raw[ref_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, 1024]
        x = ip.resampler.latents.unsqueeze(0).expand(1, -1, -1)  # [1, 16, 1024]
        for li, layer in enumerate(ip.resampler.layers):
            x_pre = x.clone()
            x = layer(x, e)
            delta = (x - x_pre).norm().item()
            print(f"  Ref{ref_idx} Layer {li}: delta_norm={delta:.4f}, x_norm={x.norm():.4f}")
