import os
import gzip
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from PIL import Image
import open_clip
from torchvision import transforms

# ---- Config ----
DATA_DIR = Path(os.path.dirname(__file__)) / 'Traces' / 'stone_pickaxe_easy'
OUTPUT_DIR = DATA_DIR / 'clip_features'    # output directory for CLIP features
IMG_SHAPE = (274, 274, 3)                  # expected input shape
BATCH_SIZE = 64
CLIP_MODEL = "ViT-B-32"                    # e.g., "ViT-B-32", "ViT-L-14", "ViT-H-14"
CLIP_PRETRAINED = "openai"                 # weights tag (e.g., "openai", "laion2b_s34b_b79k")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load CLIP model + preprocess ----
clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
    model_name=CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device
)
clip_model.eval()

# ToPIL for arrays; accepts float32 [0,1] or uint8
to_pil = transforms.ToPILImage()

# ---- Utility: robustly strip .pkl.gz -> .npy keeping base name identical ----
def output_name_for(input_path: Path) -> Path:
    stem = input_path.name
    if stem.endswith('.pkl.gz'):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem
    return OUTPUT_DIR / f"{stem}.npy"

# ---- Process each file independently ----
raw_dir = DATA_DIR / 'raw_data'
if not raw_dir.exists():
    raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

for in_path in tqdm(sorted(raw_dir.glob('*.pkl.gz'))):
    out_path = output_name_for(in_path)

    if out_path.exists():
        continue

    # Load images from pickle
    with gzip.open(in_path, 'rb') as f:
        data = pickle.load(f)
        if 'all_obs' not in data:
            raise KeyError(f"'all_obs' key missing in {in_path}")
        imgs = data['all_obs']  # list/array (N, H, W, C) float32 in [0,1]

    imgs = np.asarray(imgs, dtype=np.float32)
    n, h, w, c = imgs.shape
    if (h, w, c) != IMG_SHAPE:
        print(f"Warning: {in_path.name} has shape {(h,w,c)} != {IMG_SHAPE}")

    feats_list = []
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = imgs[start:end]  # (B, H, W, C)

            # Convert each frame -> PIL -> CLIP preprocess -> tensor
            pil_imgs = [to_pil(frame) for frame in batch]  # ToPILImage handles float32 [0,1]
            tensors = [clip_preprocess(img) for img in pil_imgs]  # (3, 224/336, 224/336)
            x = torch.stack(tensors, dim=0).to(device)

            # Encode with CLIP and L2-normalize
            z = clip_model.encode_image(x)  # (B, D)
            z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            feats_list.append(z.cpu().float().numpy())

    X_feats = np.concatenate(feats_list, axis=0)  # (N, D) e.g., D=512 for ViT-B/32

    np.save(out_path, X_feats)
    print(f"Saved {out_path.name}, shape={X_feats.shape} (model={CLIP_MODEL}, weights={CLIP_PRETRAINED})")