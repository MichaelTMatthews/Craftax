import os
import gzip
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from torchvision import models, transforms
from PIL import Image

# ---- Config ----
DATA_DIR = Path(os.path.dirname(__file__)) / 'Traces' / 'Test'
OUTPUT_DIR = DATA_DIR / 'resnet_features'   # new output directory
IMG_SHAPE = (274, 274, 3)                   # expected input shape
BATCH_SIZE = 512

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load ResNet50 pretrained on ImageNet ----
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = torch.nn.Identity()  # remove classification head → outputs 2048-D
resnet.eval().to(device)

# ---- Preprocessing transform (ImageNet) ----
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

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
        imgs = data['all_obs']

    imgs = np.asarray(imgs, dtype=np.float32)  # (N, H, W, C)
    n, h, w, c = imgs.shape
    if (h, w, c) != IMG_SHAPE:
        print(f"Warning: {in_path.name} has shape {(h,w,c)} != {IMG_SHAPE}")

    # ---- Run through ResNet in batches ----
    feats_list = []
    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = imgs[start:end]  # (B, H, W, C)

            # Apply transforms to each image
            tensors = [transform(frame.astype(np.uint8)) for frame in batch]
            x = torch.stack(tensors, dim=0).to(device)  # (B, 3, 224, 224)

            feats = resnet(x)  # (B, 2048)
            feats_list.append(feats.cpu().numpy())

    X_feats = np.concatenate(feats_list, axis=0)  # (N, 2048)

    # Save per-file features
    np.save(out_path, X_feats)
    print(f"Saved {out_path.name}, shape={X_feats.shape}")