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

### Anima Qwen3-VL Loader
Loads [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) for on-the-fly image embedding extraction. Accepts a HuggingFace model ID or local path.

### Anima Qwen3-VL Encode Image
Extracts a 1024-dim L2-normalized embedding from a reference image using Qwen3-VL. Connect the output to `Anima IP-Adapter Apply`'s `image_emb` input.

## Usage

### Option A: Pre-computed embedding (no extra VRAM)

1. Load your Anima model as usual
2. Load IP-Adapter: `Anima IP-Adapter Loader` → select .safetensors
3. Load reference image embedding: `Anima Image Embedding Loader` → select .pt file
4. Apply: `Anima IP-Adapter Apply` → connect model + ipadapter + image_emb
5. Sample as usual with KSampler

### Option B: On-the-fly encoding (~4GB extra VRAM)

1. Load your Anima model as usual
2. Load IP-Adapter: `Anima IP-Adapter Loader` → select .safetensors
3. Load Qwen3-VL: `Anima Qwen3-VL Loader` → model path (default: `Qwen/Qwen3-VL-Embedding-2B`)
4. Encode image: `Anima Qwen3-VL Encode Image` → connect Qwen3-VL model + ComfyUI IMAGE
5. Apply: `Anima IP-Adapter Apply` → connect model + ipadapter + image_emb
6. Sample as usual with KSampler

## Pre-computing Image Embeddings

If you prefer to pre-compute embeddings outside ComfyUI, use [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B):

```python
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-Embedding-2B", torch_dtype=torch.bfloat16
).cuda().eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")

messages = [
    {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
    {"role": "user", "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": "Describe this image."},
    ]},
]

text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
images, _, _ = process_vision_info([messages], return_video_kwargs=True)
inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

# Pool by last token, truncate to 1024-dim (Matryoshka), L2 normalize
attn_mask = inputs["attention_mask"]
last_pos = attn_mask.flip(dims=[1]).argmax(dim=1)
col = attn_mask.shape[1] - last_pos - 1
row = torch.arange(outputs.last_hidden_state.shape[0], device="cuda")
emb = outputs.last_hidden_state[row, col]
emb = F.normalize(emb[:, :1024], p=2, dim=-1)

torch.save(emb.squeeze().cpu(), "my_image_emb.pt")
```

## Architecture

- **Perceiver Resampler**: Qwen3-VL emb (1024-dim) → 16 tokens
- **28 decoupled cross-attention layers**: Injected after each DiT block's text cross-attention
- **Injection formula**: `gate * (text_attn + λ * ip_attn) + residual`

## License

Apache-2.0
