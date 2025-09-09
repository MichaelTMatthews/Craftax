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

DATA_DIR = Path(os.path.dirname(__file__)) / 'Traces' / 'Test'

all_files = [f.replace('.pkl.gz', '') for f in os.listdir(DATA_DIR / 'raw_data') if f.endswith('.pkl.gz')]


for file in all_files:
    with gzip.open(DATA_DIR / 'raw_data' / (file + '.pkl.gz'), 'rb') as f:
        data = pickle.load(f)
        raw_obs = data['all_obs']
        raw_truth = data['all_truths']
        raw_actions = data['all_actions']

    pca_feats = np.load(DATA_DIR / 'pca_features' / (file + '.npy'))
    resnet_feats = np.load(DATA_DIR / 'resnet_features' / (file + '.npy'))
    clip_feats = np.load(DATA_DIR / 'clip_features' / (file + '.npy'))

    with open(DATA_DIR / 'groundTruth' / (file + ''), 'r') as truth_file:
        ground_truth_text = truth_file.read()
        #split to a list
        ground_truth_list = ground_truth_text.splitlines()
    

    #make sure all lists are the same length
    assert len(raw_obs) == len(raw_truth) == len(raw_actions) == len(pca_feats) == len(resnet_feats) == len(clip_feats) == len(ground_truth_list), f"Length mismatch in file {file}"

print("All files verified successfully.")