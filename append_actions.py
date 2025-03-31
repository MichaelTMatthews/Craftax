import numpy as np
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Incremental PCA to image data and save transformed features.")
    
    parser.add_argument("--folder-path", type=str, required=True, 
                        help="Base folder path containing image data.")
    
    args = parser.parse_args()
    os.makedirs(args.folder_path + "features_with_actions", exist_ok=True)

    files = os.listdir(args.folder_path + "features")
    
    for file in files:
        features = np.load(os.path.join(args.folder_path, "features", file))  # Shape: M x N
        actions = np.load(os.path.join(args.folder_path, "actions", file))    # Shape: (N,)

        one_hot_actions = np.eye(17)[actions] 

        augmented_features = np.concatenate([features, one_hot_actions], axis=1)

        np.save(os.path.join(args.folder_path, "features_with_actions", file), augmented_features)

    


    
