import os
import gzip
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import joblib

# ---- Config ----
DATA_DIR = Path(os.path.dirname(__file__)) / 'Traces' / 'Test' 
OUTPUT_DIR = DATA_DIR / 'pca_features'   # change to DATA_DIR if you want them alongside inputs
MODEL_PATH = DATA_DIR / 'pca_model.joblib'  # path to your saved model
IMG_SHAPE = (274, 274, 3)                # for sanity checks (optional)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("model dir:", MODEL_PATH)

# ---- Load model (scaler + PCA) ----
artifacts = joblib.load(MODEL_PATH)
scaler = artifacts['scaler']   # StandardScaler(with_std=False)
pca = artifacts['pca']         # PCA(n_components=512)
n_features_expected = scaler.mean_.shape[0]
print(f"Loaded model: PCA comps={pca.n_components_}, expected features={n_features_expected}")

# ---- Utility: robustly strip .pkl.gz -> .npy keeping base name identical ----
def output_name_for(input_path: Path) -> Path:
    stem = input_path.name
    if stem.endswith('.pkl.gz'):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem
    return OUTPUT_DIR / f"{stem}.npy"

# ---- Process each file independently (no shuffling) ----
raw_dir = DATA_DIR / 'raw_data'
if not raw_dir.exists():
    raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

for in_path in tqdm(sorted(raw_dir.glob('*.pkl.gz'))):
    out_path = output_name_for(in_path)

    if out_path.exists():
        continue

    with gzip.open(in_path, 'rb') as f:
        data = pickle.load(f)
        if 'all_obs' not in data:
            raise KeyError(f"'all_obs' key missing in {in_path}")
        imgs = data['all_obs']

    imgs = np.asarray(imgs, dtype=np.float32)
    n, h, w, c = imgs.shape
    if (h, w, c) != IMG_SHAPE:
        print(f"Warning: {in_path.name} has shape {(h,w,c)} != {IMG_SHAPE}")

    X = imgs.reshape(n, -1)
    if X.shape[1] != n_features_expected:
        raise ValueError(
            f"Feature size mismatch for {in_path.name}: got {X.shape[1]}, "
            f"model expects {n_features_expected}"
        )

    X_centered = scaler.transform(X)
    X_feats = pca.transform(X_centered)

    np.save(out_path, X_feats)