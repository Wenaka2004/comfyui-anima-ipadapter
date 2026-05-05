"""Quick diagnostic: v2 resampler discriminability test (untrained)."""
import torch, json, sys
from pathlib import Path
import pyarrow.parquet as pq

sys.path.insert(0, "/data/stardust/anima_ipa/stage7_train/code")
from ip_adapter import IPAdapter

ip = IPAdapter(
    emb_dim=1024, x_dim=2048, num_blocks=2,  # 2 blocks for quick test
    ip_dim=512, num_ip_heads=8, ip_head_dim=64,
    num_tokens=8, num_perceiver_layers=2, num_perceiver_heads=4,
    num_input_tokens=8,
).cuda().bfloat16().eval()

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

print("=== v2 Resampler (untrained) — discriminability test ===")
print("\nRaw VL embeddings pairwise cosine similarity:")
raw = []
for idx, rid in enumerate(unique_ids[:5]):
    e = embs.get(rid)
    if e is None:
        continue
    raw.append(e.cuda().to(torch.bfloat16))

for i in range(len(raw)):
    for j in range(i + 1, len(raw)):
        cs = torch.nn.functional.cosine_similarity(raw[i].flatten().unsqueeze(0), raw[j].flatten().unsqueeze(0)).item()
        print(f"  Ref{i} vs Ref{j}: {cs:.4f}")

print("\nv2 Resampler output pairwise cosine similarity (should be LOWER than v1's 0.996):")
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

for i in range(len(all_tok)):
    for j in range(i + 1, len(all_tok)):
        cs = torch.nn.functional.cosine_similarity(all_tok[i].flatten().unsqueeze(0), all_tok[j].flatten().unsqueeze(0)).item()
        print(f"  Ref{i} vs Ref{j}: {cs:.4f}")

# Zero vs real
print("\nZero embedding test:")
with torch.no_grad():
    zero_tok = ip.resample(torch.zeros(1, 1, 1024, device="cuda", dtype=torch.bfloat16))
print(f"  Zero tokens norm={zero_tok.norm():.4f}")
if all_tok:
    cs = torch.nn.functional.cosine_similarity(zero_tok.flatten().unsqueeze(0), all_tok[0].flatten().unsqueeze(0)).item()
    print(f"  Zero vs Ref0: {cs:.4f}")
