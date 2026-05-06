"""v4 AdaLN modulation verification: discriminability + gradient flow."""
import torch, json, sys
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, "/data/stardust/anima_ipa/stage7_train/code")
from ip_adapter import IPAdapter

device = torch.device("cuda:0")
ip = IPAdapter(
    vl_dim=1024, block_dim=2048, num_blocks=2,
    modulation_hidden=256,
).to(device, dtype=torch.bfloat16)

print(f"IPAdapter v4: {sum(p.numel() for p in ip.parameters())/1e6:.1f}M params")
print(f"ip_scales init values: {[s.item() for s in ip.ip_scales]}")

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
        unique_ids.append(p["ref_id"]); seen.add(p["ref_id"])
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

print("\n=== 1. Modulation discriminability (per-block scale/shift) ===")
all_mods = []
for idx, rid in enumerate(unique_ids[:5]):
    e = embs.get(rid)
    if e is None: continue
    vl = e.cuda().to(torch.bfloat16)  # [1024]
    with torch.no_grad():
        scale0, shift0 = ip.get_modulation(0, vl)
        scale1, shift1 = ip.get_modulation(1, vl)
    all_mods.append((scale0, shift0, scale1, shift1))
    print(f"  Ref{idx}: block0 scale std={scale0.std():.4f}, shift mean={shift0.mean():.4f}")

collapsed = False
for i in range(len(all_mods)):
    for j in range(i+1, len(all_mods)):
        # Compare block 0 scale modulations
        cs = torch.nn.functional.cosine_similarity(
            all_mods[i][0].flatten().unsqueeze(0), all_mods[j][0].flatten().unsqueeze(0)
        ).item()
        if cs > 0.99: collapsed = True
        print(f"  Block0 scale Ref{i} vs Ref{j}: cos_sim={cs:.4f}{' !!!' if cs > 0.99 else ''}")

# Zero test
with torch.no_grad():
    zero_scale, zero_shift = ip.get_modulation(0, torch.zeros(1024, device=device, dtype=torch.bfloat16))
if all_mods:
    cs = torch.nn.functional.cosine_similarity(
        zero_scale.flatten().unsqueeze(0), all_mods[0][0].flatten().unsqueeze(0)
    ).item()
    print(f"  Zero vs Ref0 block0 scale: cos_sim={cs:.4f} (should differ)")

print(f"\n{'==> COLLAPSED!' if collapsed else '==> NOT COLLAPSED - AdaLN modulations differ per ref!'}")

print("\n=== 2. Gradient flow test ===")
ip.train()
vl = torch.randn(2, 1024, device=device, dtype=torch.bfloat16)
loss = torch.zeros(1, device=device)
for bi in range(2):
    s, sh = ip.get_modulation(bi, vl)
    loss = loss + s.sum() + sh.sum()
loss.backward()

has_grad = 0
total = 0
for name, param in ip.named_parameters():
    total += 1
    g = param.grad is not None and param.grad.abs().sum() > 0
    if g: has_grad += 1
print(f"  {has_grad}/{total} params have gradients")
for bi in range(2):
    print(f"  ip_scales[{bi}] grad: {ip.ip_scales[bi].grad.item():.6f}")

print(f"\n{'==> ALL OK - AdaLN modulation ready for training!' if has_grad == total and not collapsed else '==> FAILED'}")
