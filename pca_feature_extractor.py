import numpy as np
import os
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
import psutil
import argparse
import json

def get_reconstructed_images(folder, pca, max_episodes=30):
    """
    Load images from a specified folder, apply PCA transformation and reconstruction.
    
    Args:
        folder (str): Path to the folder containing .npy image files.
        pca (IncrementalPCA): Trained PCA model.
        max_episodes (int): Maximum number of image files to process.
    
    Returns:
        tuple: Original images and reconstructed images.
    """
    images = []
    t = 0
    for filename in os.listdir(folder):
        if t >= max_episodes:
            break
        if filename.endswith(".npy"):
            file_path = os.path.join(folder, filename)
            img_array = np.load(file_path, mmap_mode='r')  # Load using memory mapping
            images.append(img_array)
            t += 1
    
    combined_images = np.concatenate(images, axis=0)  # Concatenate along first axis
    flattened_images = combined_images.reshape(combined_images.shape[0], -1)  # Flatten images
    
    X_pca = pca.transform(flattened_images)
    X_reconstructed = pca.inverse_transform(X_pca)
    X_reconstructed = X_reconstructed.reshape(-1, 274, 274, 3)
    
    return combined_images, X_reconstructed

def visualize_images(original, reconstructed, path, num_samples=5):
    samples = np.random.choice(len(original), num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2))
    for i, img in enumerate(samples):
        axes[i, 0].imshow(original[img])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow((reconstructed[img] * 255).clip(0, 255).astype(np.uint8))
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(path, "reconstructed_images.pdf"))


def apply_incremental_pca(folder, n_components=600, batch_size=40):
    """
    Apply Incremental PCA to images in the given folder.
    
    Args:
        folder (str): Path to the folder containing .npy image files.
        n_components (int): Number of PCA components.
        batch_size (int): Batch size for IncrementalPCA.
    
    Returns:
        IncrementalPCA: Trained PCA model.
    """
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    files = os.listdir(folder)
    
    for i in range(0, len(files), batch_size):
        print(f"Processing batch {i}, RAM usage: {psutil.virtual_memory().percent}%")
        batch_files = files[i:i + batch_size]
        batch_images = [np.load(os.path.join(folder, f)).reshape(-1, 274*274*3) for f in batch_files]
        batch_images = np.concatenate(batch_images, axis=0)
        pca.partial_fit(batch_images)  # Incrementally fit PCA
        del batch_images 
    
    return pca

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply Incremental PCA to image data and save transformed features.")
    
    parser.add_argument("--folder-path", type=str, required=True, 
                        help="Base folder path containing image data.")
    parser.add_argument("--n-components", type=int, default=600, 
                        help="Number of principal components for PCA (default: 600).")
    parser.add_argument("--batch-size", type=int, default=40, 
                        help="Batch size for Incremental PCA fitting (default: 40).")
    parser.add_argument("--max-episodes", type=int, default=30, 
                        help="Maximum number of episodes to process for reconstruction (default: 30).")
    parser.add_argument("--num-samples", type=int, default=5, 
                        help="Number of images to visualize (default: 5).")
    
    args = parser.parse_args()
    
    print(f"Starting PCA, RAM usage: {psutil.virtual_memory().percent}%")
    
    pca = apply_incremental_pca(os.path.join(args.folder_path, "top_down_states/"), 
                                n_components=args.n_components, 
                                batch_size=args.batch_size)
    
    print(f"Variance captured: {np.sum(pca.explained_variance_ratio_)}")
    print(f"PCA done, RAM usage: {psutil.virtual_memory().percent}%")
    
    np.save(os.path.join(args.folder_path, "pca_model.npy"), pca)
    
    X, X_reconstructed = get_reconstructed_images(os.path.join(args.folder_path, "top_down_states/"), pca, 
                                                   max_episodes=args.max_episodes)
    
    visualize_images(X, X_reconstructed, args.folder_path, num_samples=args.num_samples)
    
    os.makedirs(os.path.join(args.folder_path, "pca_features"), exist_ok=True)
    files = os.listdir(os.path.join(args.folder_path, "top_down_states/"))
    
    for i, file in enumerate(files):
        episode_images = np.load(os.path.join(args.folder_path, "top_down_states/", file))
        episode_images = episode_images.reshape(episode_images.shape[0], -1)
        episode_pca = pca.transform(episode_images)
        np.save(os.path.join(args.folder_path, "pca_features/", file), episode_pca)
    
    print("Saved episode features")

    #Save the config for the file:
    with open(os.path.join(args.folder_path, "pca_config.json"), 'w') as f:
        json.dump(vars(args), f)
