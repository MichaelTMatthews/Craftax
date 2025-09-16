import os
import json
import gzip
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt

# Optional (only if you want to save a quick visual of reconstructions):
# import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), 'Traces/stone_pickaxe_easy')
IMG_SHAPE = (274, 274, 3)
COMPONENTS = 1000
os.makedirs(DATA_DIR + '/pca_models', exist_ok=True)

# -----------------------
# 1) Load images
# -----------------------
all_images = []
for filename in tqdm(os.listdir(DATA_DIR + '/raw_data')):
    if filename.endswith('.pkl.gz'):
        file_path = os.path.join(DATA_DIR + '/raw_data', filename)
        with gzip.open(file_path, 'rb') as f:
            data = pickle.load(f)
            all_images.extend(data['all_obs'])

all_images = np.array(all_images, dtype=np.float32)
print("All images loaded", all_images.shape)  # (N, 274, 274, 3)

# -----------------------
# 2) Flatten to (N, D)
# -----------------------
N = all_images.shape[0]
all_images = all_images.reshape(N, -1)
print("All images flattened", all_images.shape)  # (N, 225252)

# -----------------------
# 3) Mean-center (keep scaler for inverse)
# -----------------------
scaler = StandardScaler(with_std=False)
all_images_centered = scaler.fit_transform(all_images)

# -----------------------
# 4) Shuffle (keep same permutation for originals)
# -----------------------
rng = np.random.default_rng(0)
perm = rng.permutation(N)
all_images_centered = all_images_centered[perm]
all_images_orig = all_images[perm]  # uncentered originals aligned to the same order

# -----------------------
# 5) PCA fit
# -----------------------
pca = PCA(n_components=COMPONENTS)  # or set n_components=0.95 to target 95% variance
X_pca = pca.fit_transform(all_images_centered)

# -----------------------
# 6) Print variance captured
# -----------------------
expl_ratio = pca.explained_variance_ratio_
cum_var = np.cumsum(expl_ratio)
print(f"Kept components: {pca.n_components_}")
print(f"Explained variance captured (sum): {cum_var[-1]:.4f}")
print("First 10 component variance ratios:", np.round(expl_ratio[:10], 4))
print("Cumulative after 10 comps:", round(cum_var[min(9, len(cum_var)-1)], 4))

# -----------------------
# 7) Save model compressed
# -----------------------
# A) Save the full objects (handy if you want to inverse_transform later)
joblib.dump(
    {
        'pca': pca,
        'scaler': scaler,
        'img_shape': IMG_SHAPE
    },
    f'{DATA_DIR}/pca_models/pca_model_{COMPONENTS}.joblib',
    compress=3
)
print("Saved PCA model + scaler -> pca_model.joblib")

# B) Also save a lightweight NPZ (portable across environments)
np.savez_compressed(
    f'{DATA_DIR}/pca_models/pca_artifacts_{COMPONENTS}.npz',
    components=pca.components_,
    mean=scaler.mean_,
    explained_variance=pca.explained_variance_,
    explained_variance_ratio=pca.explained_variance_ratio_,
    singular_values=pca.singular_values_,
    n_components=np.array([pca.n_components_]),
    img_shape=np.array(IMG_SHAPE)
)
print("Saved PCA artifacts -> pca_artifacts.npz")

# Free memory if desired
del all_images

# -----------------------
# 8) Reconstruct 5 random images and report error
# -----------------------
k = 5
idx = rng.choice(X_pca.shape[0], size=k, replace=False)

# Reconstruct from PCA space -> centered feature space
recon_centered = pca.inverse_transform(X_pca[idx])

# Undo centering -> original pixel space in [~0,1]
recon = scaler.inverse_transform(recon_centered)

# Clip to [0,1] just in case of slight negative/overflow due to projection
recon = np.clip(recon, 0.0, 1.0)

# Reshape both originals and reconstructions to image tensors
recon_imgs = recon.reshape(k, *IMG_SHAPE)
orig_imgs = all_images_orig[idx].reshape(k, *IMG_SHAPE)

# Compute simple metrics
mse = np.mean((recon_imgs - orig_imgs) ** 2, axis=(1,2,3))
psnr = 10.0 * np.log10(1.0 / np.maximum(mse, 1e-12))

print("Reconstruction sample indices:", idx.tolist())
for i in range(k):
    print(f"[{i}] MSE={mse[i]:.6f}  PSNR={psnr[i]:.2f} dB")

# -----------------------
# 9) (Optional) Save a side-by-side montage as a PNG
# -----------------------
# Uncomment this block if you want a quick visual saved to disk.

import math
rows = k
fig, axes = plt.subplots(rows, 2, figsize=(6, 3*rows))
if rows == 1:
    axes = np.array([axes])
for i in range(rows):
    axes[i,0].imshow(orig_imgs[i])
    axes[i,0].set_title(f"Original #{idx[i]}")
    axes[i,0].axis('off')
    axes[i,1].imshow(recon_imgs[i])
    axes[i,1].set_title(f"Reconstruction #{idx[i]}")
    axes[i,1].axis('off')
plt.tight_layout()
out_path = DATA_DIR + f"/pca_models/pca_recon_samples_{COMPONENTS}.png"
plt.savefig(out_path, dpi=300)
print(f"Saved reconstruction montage -> {out_path}")

