#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import joblib
import imageio.v2 as imageio  # imageio>=2 supports animated GIFs

# -------- CLI --------
parser = argparse.ArgumentParser(description="Reconstruct images from PCA features and save as GIF.")
parser.add_argument("features_path", type=Path, help="Path to the .npy file with PCA features (N, 512).")
parser.add_argument("--model", type=Path, default=Path("pca_model.joblib"),
                    help="Path to saved PCA+scaler joblib (default: pca_model.joblib).")
parser.add_argument("--out", type=Path, default=None,
                    help="Output GIF path. Default: same basename as features with .gif extension.")
parser.add_argument("--batch_size", type=int, default=256, help="Reconstruction batch size.")
parser.add_argument("--fps", type=int, default=10, help="GIF frames per second.")
parser.add_argument("--clip_min", type=float, default=0.0, help="Min clip after inverse transform.")
parser.add_argument("--clip_max", type=float, default=1.0, help="Max clip after inverse transform.")
args = parser.parse_args()

features_path: Path = args.features_path
out_path: Path = args.out or features_path.with_suffix(".gif")

# -------- Load model --------
artifacts = joblib.load(args.model)
pca = artifacts["pca"]
scaler = artifacts["scaler"]
img_shape = tuple(artifacts.get("img_shape", (274, 274, 3)))  # fallback

latent = np.load(features_path)  # shape (N, n_components)
N, n_latent = latent.shape
if n_latent != getattr(pca, "n_components_", n_latent):
    raise ValueError(f"Feature width {n_latent} does not match PCA.n_components_={pca.n_components_}")

D_expected = scaler.mean_.shape[0]
H, W, C = img_shape
if D_expected != H * W * C:
    raise ValueError(f"Scaler/PCA trained for {D_expected} features but img_shape={img_shape} -> {H*W*C}")

# -------- Reconstruct & write GIF (streaming) --------
out_path.parent.mkdir(parents=True, exist_ok=True)
duration = 1.0 / max(args.fps, 1)  # seconds per frame

with imageio.get_writer(out_path.as_posix(), mode="I", duration=duration, loop=0) as writer:
    for start in range(0, N, args.batch_size):
        end = min(start + args.batch_size, N)
        Z = latent[start:end]  # (B, n_components)

        # PCA inverse -> centered feature space
        X_centered = pca.inverse_transform(Z)  # (B, D)

        # Undo mean-centering -> original pixel space
        X = scaler.inverse_transform(X_centered)  # (B, D) in approx [0,1]
        X = np.clip(X, args.clip_min, args.clip_max)

        # Reshape & convert to uint8 for GIF
        X_imgs = (X.reshape(-1, H, W, C) * 255.0 + 0.5).astype(np.uint8)

        # Append frames
        for frame in X_imgs:
            # GIF expects either (H, W) or (H, W, 3) uint8; RGB is fine.
            writer.append_data(frame)

print(f"Saved GIF -> {out_path}  ({N} frames @ {args.fps} fps)")