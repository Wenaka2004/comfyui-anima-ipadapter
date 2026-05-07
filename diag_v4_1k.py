"""v4 1k checkpoint discriminability test (CCIP 768-d input)."""
import torch, json, sys
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, "/data/stardust/anima_ipa/comfyui-anima-ipadapter")
from ip_adapter_model import IPAdapter
from safetensors.torch import load_file

sd = load_file("/data/stardust/anima_ipa/stage7_train/out_v4/ip_adapter_step1000.safetensors")
num_blocks = sum(1 for k in sd if k.startswith("block_mods.") and k.endswith(".proj.weight"))
vl_dim = sd["block_mods.0.proj.weight"].shape[1]
mod_hidden = sd["block_mods.0.proj.weight"].shape[0]
block_dim = sd["block_mods.0.mod.weight"].shape[0] // 2
print(f"Detected: num_blocks={num_blocks}, vl_dim={vl_dim}, mod_hidden={mod_hidden}, block_dim={block_dim}")

ip = IPAdapter(vl_dim=vl_dim, block_dim=block_dim, num_blocks=num_blocks, modulation_hidden=mod_hidden)
ip.load_state_dict(sd)
ip.cuda().bfloat16().eval()

# Load 5 real CCIP features
pairs = []
with open("/data/stardust/anima_ipa/dataset_build/stage4_out/training_pairs_final2.jsonl") as f:
    for _ in range(200):
        pairs.append(json.loads(f.readline()))

unique_ids = list(set(p["ref_id"] for p in pairs))[:5]

ccip_dir = Path("/data/stardust/anima_ipa/stage7_train/ccip_features")
ccip_feats = {}
id_set = set(unique_ids)
for shard in sorted(ccip_dir.glob("ccip_shard_*.parquet")):
    t = pq.read_table(shard)
    for i, feat_bytes in zip(t.column("id").to_pylist(), t.column("feat").to_pylist()):
        if i in id_set:
            ccip_feats[i] = torch.from_numpy(np.frombuffer(feat_bytes, dtype=np.float32)).float()
    if len(ccip_feats) >= len(id_set):
        break

# If not enough, use any IDs from the first shard
if len(ccip_feats) < 5:
    t = pq.read_table(sorted(ccip_dir.glob("ccip_shard_*.parquet"))[0])
    for i, feat_bytes in zip(t.column("id").to_pylist()[:10], t.column("feat").to_pylist()[:10]):
        ccip_feats[i] = torch.from_numpy(np.frombuffer(feat_bytes, dtype=np.float32)).float()
    unique_ids = list(ccip_feats.keys())[:5]

print(f"\nLoaded {len(ccip_feats)} CCIP features")

# Test discriminability with CCIP as input
print("\n=== v4 AdaLN modulation discriminability (CCIP 768-d input, 1k steps) ===")
all_ok = True
for bi in [0, 7, 14, 21, 27]:
    mods = []
    for rid in unique_ids:
        if rid not in ccip_feats:
            continue
        vl = ccip_feats[rid].cuda().to(torch.bfloat16)
        with torch.no_grad():
            s, _ = ip.get_modulation(bi, vl)
        mods.append(s.flatten())

    print(f"\n  Block {bi} scale modulations:")
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            cs = torch.nn.functional.cosine_similarity(mods[i].unsqueeze(0), mods[j].unsqueeze(0)).item()
            ok = cs < 0.98
            if not ok:
                all_ok = False
            print(f"    Ref{i} vs Ref{j}: cos_sim={cs:.4f}{' !!!' if not ok else ''}")

print(f"\n{'==> NOT COLLAPSED — CCIP input works!' if all_ok else '==> COLLAPSED'}")
