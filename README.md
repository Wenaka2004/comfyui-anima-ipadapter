# ComfyUI Anima IP-Adapter

IP-Adapter custom node for [Anima](https://github.com/circlestone-labs/Anima) in ComfyUI.

Injects reference image features into Anima's DiT via decoupled cross-attention, enabling character-consistent image generation.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/circlestone-labs/comfyui-anima-ipadapter.git
pip install -r comfyui-anima-ipadapter/requirements.txt
```

## Nodes

### Anima IP-Adapter Loader
Loads an IP-Adapter checkpoint (.safetensors).

### Anima IP-Adapter Apply
Applies IP-Adapter to an Anima model with a reference image embedding.

**Inputs:**
- `model` — Anima model
- `ipadapter` — IP-Adapter from loader
- `image_emb` — Qwen3-VL embedding of reference image [1, 1024]
- `start_at` / `end_at` — Sampling step range (0.0–1.0)
- `weight` — Global IP-Adapter scale (default 1.0)

### Anima Image Embedding Loader
Loads a pre-computed Qwen3-VL image embedding from a .pt file.

## Usage

1. Load your Anima model as usual
2. Load IP-Adapter: `Anima IP-Adapter Loader` → select .safetensors
3. Load reference image embedding: `Anima Image Embedding Loader` → select .pt file
4. Apply: `Anima IP-Adapter Apply` → connect model + ipadapter + image_emb
5. Sample as usual with KSampler

## Pre-computing Image Embeddings

Use [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) to extract 1024-dim embeddings from reference images:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16
).cuda()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")

# Extract embedding
inputs = processor(images=[pil_image], text="Describe this image.", return_tensors="pt").to("cuda")
with torch.no_grad():
    emb = model(**inputs).last_hidden_state.mean(dim=1)  # [1, 1024]
torch.save(emb.squeeze().cpu(), "my_image_emb.pt")
```

## Architecture

- **Perceiver Resampler**: Qwen3-VL emb (1024-dim) → 16 tokens
- **28 decoupled cross-attention layers**: Injected after each DiT block's text cross-attention
- **Injection formula**: `gate * (text_attn + λ * ip_attn) + residual`

## License

Apache-2.0
