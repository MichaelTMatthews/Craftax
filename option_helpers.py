import torch
import torch.nn.functional as F
import numpy as np
import os 
import joblib 
import json 
from joblib import load as joblib_load


# --- must match your training definitions ---
class ImageNormalizer:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3,1,1)
        self.std  = torch.clamp(torch.tensor(std, dtype=torch.float32).view(3,1,1), min=1e-3)
    def __call__(self, x):  # x: [3,H,W] in [0,1]
        return (x - self.mean) / self.std

class ConvBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn   = torch.nn.BatchNorm2d(c_out)  # or GroupNorm if you switched
        self.act  = torch.nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class PolicyCNN(torch.nn.Module):
    def __init__(self, n_actions=16):
        super().__init__()
        self.stem = torch.nn.Sequential(
            ConvBlock(3, 32, k=7, s=2, p=3),
            ConvBlock(32, 32),
            torch.nn.MaxPool2d(2),
        )
        self.stage2 = torch.nn.Sequential(
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            torch.nn.MaxPool2d(2),
        )
        self.stage3 = torch.nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            torch.nn.MaxPool2d(2),
        )
        self.stage4 = torch.nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.head = torch.nn.Linear(256, n_actions)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.head(x)

# ---- inference helpers ----

def load_policy(ckpt_path, device=None):
    """Load model + normalizer from a saved training checkpoint."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device)
    n_actions = int(ckpt['n_actions'])
    model = PolicyCNN(n_actions=n_actions).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    normalizer = ImageNormalizer(ckpt['mean'], ckpt['std'])
    return model, normalizer, device, n_actions

def preprocess_frame(frame_hw3, normalizer, target=256):
    """
    frame_hw3: numpy array [H,W,3], float32 in [0,1]
    returns torch tensor [1,3,target,target]
    """
    assert frame_hw3.ndim == 3 and frame_hw3.shape[2] == 3
    x = torch.from_numpy(np.transpose(frame_hw3, (2,0,1))).float()   # [3,H,W]
    x = F.interpolate(x.unsqueeze(0), size=(target, target), mode='bilinear', align_corners=False).squeeze(0)  # [3,T,T]
    x = normalizer(x)
    return x.unsqueeze(0)  # [1,3,T,T]

@torch.no_grad()
def act_greedy(model, normalizer, device, frame_hw3):
    """
    Returns (action_id, probs) where probs is a numpy array length n_actions.
    """
    x = preprocess_frame(frame_hw3, normalizer)            # [1,3,256,256]
    x = x.to(device)
    logits = model(x)                                      # [1,n_actions]
    probs = torch.softmax(logits, dim=-1).squeeze(0)       # [n_actions]
    action = int(torch.argmax(probs).item())
    return action, probs.cpu().numpy()

@torch.no_grad()
def act_sample(model, normalizer, device, frame_hw3, temperature=1.0):
    x = preprocess_frame(frame_hw3, normalizer).to(device)
    logits = model(x).squeeze(0)
    if temperature != 1.0:
        logits = logits / max(1e-6, float(temperature))
    probs = torch.softmax(logits, dim=-1)
    action = int(torch.multinomial(probs, num_samples=1).item())
    return action, probs.cpu().numpy()

def load_pu_start_models(models_dir: str):
    """
    Load (skill, clf, threshold, meta) tuples from <models_dir>.
    Expects files: <skill>_clf.joblib and <skill>_meta.json
    Returns: list[dict] with keys: skill, clf, thr, meta
    """
    models = []
    for fname in os.listdir(models_dir):
        if not fname.endswith("_meta.json"):
            continue
        skill = fname[:-10]  # strip "_meta.json"
        meta_path  = os.path.join(models_dir, f"{skill}_meta.json")
        model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
        if not os.path.exists(model_path):
            continue
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            thr = float(meta["threshold"])
            clf = joblib_load(model_path)
            models.append({"skill": skill, "clf": clf, "thr": thr, "meta": meta})
        except Exception as e:
            print(f"[WARN] Skipping {skill}: {e}")
    return models

def applicable_pu_start_models(models, state, *, return_details=False, eps=0.0):
    """
    Given a list from load_pu_models(...) and a state feature vector (shape [d] or [1,d]),
    return/print skills whose probability >= threshold (+eps).
    - return_details=True returns a list of dicts with scores/margins
    - eps lets you demand a small margin above threshold (e.g., eps=0.02).
    """
    # Accept 1D or 2D input
    state = np.asarray(state)
    if state.ndim == 1:
        X = state.reshape(1, -1)
    elif state.ndim == 2 and state.shape[0] == 1:
        X = state
    else:
        raise ValueError("`state` must be a single feature vector of shape [d] or [1,d].")

    rows = []
    for m in models:
        prob = float(m["clf"].predict_proba(X)[:, 1][0])
        thr  = float(m["thr"])
        margin = prob - thr
        is_applicable = prob >= (thr + eps)
        rows.append({
            "skill": m["skill"],
            "prob": prob,
            "thr": thr,
            "margin": margin,
            "applicable": is_applicable
        })

    # Sort by confidence margin (best first)
    rows.sort(key=lambda r: r["margin"], reverse=True)

    # Print list of applicable models
    applicable = [r for r in rows if r["applicable"]]
    # if applicable:
    #     print("Applicable models (prob ≥ threshold):")
    #     for r in applicable:
    #         print(f"  - {r['skill']}: p={r['prob']:.3f}  thr={r['thr']:.3f}  margin={r['margin']:.3f}")
    # else:
    #     print("No applicable models for this state.")

    return rows if return_details else [r["skill"] for r in applicable]

def load_pu_end_model(models_dir: str, skill: str):
    """
    Load a single PU model (classifier + metadata) for a given skill.

    Looks for:
      - <models_dir>/<skill>_clf.joblib
      - <models_dir>/<skill>_meta.json

    Returns:
      dict with keys:
        - skill: str
        - clf:   fitted classifier (expects .predict_proba)
        - thr:   float threshold from meta["threshold"]
        - meta:  dict (entire meta JSON)
    """
    model_path = os.path.join(models_dir, f"{skill}_clf.joblib")
    meta_path  = os.path.join(models_dir, f"{skill}_meta.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta file:  {meta_path}")

    clf = joblib_load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    thr = float(meta["threshold"])
    return {"skill": skill, "clf": clf, "thr": thr, "meta": meta}


def predict_pu_end_state(model: dict, state) -> dict:
    """
    Score a single state with a loaded PU model dict from load_pu_model(...).

    Args:
      - model: dict with keys {"skill","clf","thr","meta"}
      - state: shape [d] or [1, d]

    Returns:
      dict: {prob, threshold, is_end, margin}
    """
    # Accept 1D or 2D single-row input
    state = np.asarray(state)
    if state.ndim == 1:
        X = state.reshape(1, -1)
    elif state.ndim == 2 and state.shape[0] == 1:
        X = state
    else:
        raise ValueError("`state` must be a single feature vector of shape [d] or [1, d].")

    # Compute positive-class probability
    prob = float(model["clf"].predict_proba(X)[:, 1][0])

    thr = float(model["thr"])
    margin = prob - thr
    return {
        "prob": prob,
        "threshold": thr,
        "is_end": bool(prob >= thr),
        "margin": margin,
    }

def load_all_models(skill_list = ['wood', 'stone', 'wood_pickaxe', 'stone_pickaxe', 'table']):
    bc_models = {}
    for skill in skill_list:
        ckpt_path = os.path.join('Traces/stone_pickaxe_easy', 'bc_checkpoints', f'{skill}_policy_cnn.pt')
        bc_models[skill] = load_policy(ckpt_path)

    artifacts = joblib.load('Traces/stone_pickaxe_easy/pca_models/pca_model_750.joblib')
    scaler = artifacts['scaler']
    pca = artifacts['pca']
    n_features_expected = scaler.mean_.shape[0]

    pu_start_models = load_pu_start_models('Traces/stone_pickaxe_easy/pu_start_models')

    pu_end_models = {}
    for skill in skill_list:
        try:
            pu_end_models[skill] = load_pu_end_model('Traces/stone_pickaxe_easy/pu_end_models', skill)
        except FileNotFoundError:
            print(f"[WARN] No PU end model for skill '{skill}'")

    return {
        "skills": skill_list,  # <—— canonical order
        "bc_models": bc_models,
        "termination_models": pu_end_models,
        "start_models": pu_start_models,
        "pca_model": {'scaler': scaler, 'pca': pca, 'n_features_expected': n_features_expected}
    }

def available_skills(models, state):
    # state: uint8 or float32 flat vector -> PCA space
    state = np.asarray(state).astype(np.float32)
    if state.max() > 1.0:  # allow uint8 input
        state = state / 255.0

    X = state.reshape(1, -1)
    Xc = models["pca_model"]['scaler'].transform(X)
    Xf = models["pca_model"]['pca'].transform(Xc)

    rows = applicable_pu_start_models(models["start_models"], Xf, return_details=True, eps=0.0)
    applicable = {r["skill"] for r in rows if r["applicable"]}
    order = models["skills"]
    return np.array([s in applicable for s in order], dtype=bool)

def should_terminate(models, state, skill): 
    state = np.asarray(state).astype(np.float32) / 255.0

    X = state.reshape(1, -1)
    X_centered = models["pca_model"]['scaler'].transform(X)
    X_feats = models["pca_model"]['pca'].transform(X_centered)

    return predict_pu_end_state(models["termination_models"][skill], X_feats)["is_end"]



def bc_policy(models, state, skill):

    assert state.max() > 1.0 
     # allow uint8 input

    state = np.asarray(state).astype(np.float32) / 255.0

    model, normalizer, device, n_actions = models["bc_models"][skill]
    action, probs = act_greedy(model, normalizer, device, state)
    return action


