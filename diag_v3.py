"""v3 MLP projector verification: discriminability + gradient flow."""
import torch, json, sys
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, "/data/stardust/anima_ipa/stage7_train/code")
from ip_adapter import IPAdapter

device = torch.device("cuda:0")
ip = IPAdapter(
    emb_dim=1024, x_dim=2048, num_blocks=2,
    ip_dim=512, num_ip_heads=8, ip_head_dim=64,
    num_tokens=8, mlp_hidden_mult=4,
).to(device, dtype=torch.bfloat16)

print(f"IPAdapter: {sum(p.numel() for p in ip.parameters())/1e6:.1f}M params")
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

print("\n=== 1. MLP Token Projector discriminability (untrained) ===")
all_tok = []
for idx, rid in enumerate(unique_ids[:5]):
    e = embs.get(rid)
    if e is None:
        continue
    inp = e.unsqueeze(0).unsqueeze(0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        tok = ip.resample(inp)
    all_tok.append(tok)
    print(f"  Ref{idx}: norm={tok.norm():.4f}, std={tok.std():.6f}")

print("\nPairwise cosine similarity (MUST be < 0.98 for success):")
all_ok = True
for i in range(len(all_tok)):
    for j in range(i + 1, len(all_tok)):
        cs = torch.nn.functional.cosine_similarity(all_tok[i].flatten().unsqueeze(0), all_tok[j].flatten().unsqueeze(0)).item()
        ok = cs < 0.98
        if not ok: all_ok = False
        print(f"  Ref{i} vs Ref{j}: cos_sim={cs:.4f} {'OK' if ok else 'FAILED - collapse!'}")
print(f"  {'ALL OK - MLP works!' if all_ok else 'FAILED - collapse detected'}")

# Zero test
with torch.no_grad():
    zero_tok = ip.resample(torch.zeros(1, 1, 1024, device=device, dtype=torch.bfloat16))
if all_tok:
    cs = torch.nn.functional.cosine_similarity(zero_tok.flatten().unsqueeze(0), all_tok[0].flatten().unsqueeze(0)).item()
    print(f"  Zero vs Ref0: cos_sim={cs:.4f} (should differ from real refs)")

print("\n=== 2. Gradient flow test ===")
ip.train()
ref_emb = torch.randn(2, 1, 1024, device=device, dtype=torch.bfloat16)
image_tokens = ip.resample(ref_emb)
normed_x = torch.randn(2, 16, 2048, device=device, dtype=torch.bfloat16)

# Manually test gradient through forward_block
loss = torch.zeros(1, device=device)
for bi in range(2):
    ip_out = ip.forward_block(bi, normed_x, image_tokens)
    loss = loss + ip_out.sum()
loss.backward()

has_grad = 0
total = 0
for name, param in ip.named_parameters():
    total += 1
    g = param.grad is not None and param.grad.abs().sum() > 0
    if g: has_grad += 1
    if "token_projector" in name or "ip_scale" in name:
        status = "YES" if g else "NO"
        print(f"  {status} {name}")

print(f"\n  {has_grad}/{total} params have gradients")
print(f"  {'ALL OK - gradient flow working' if has_grad == total else 'FAILED'}")

# ip_scales gradient check
for bi in range(2):
    print(f"  ip_scales[{bi}] grad: {ip.ip_scales[bi].grad.item():.6f} (init=0.01)")
print("\n=== ALL TESTS PASSED ===" if all_ok and has_grad == total else "\n=== SOME TESTS FAILED ===")
