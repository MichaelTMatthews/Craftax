#!/bin/bash
#SBATCH --job-name=gen_craftax          # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/generate_data.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate craftax

# echo "Starting data generation..."
# python gen_truth_stats.py
# echo "Truth stats generated."
# python train_pca_model.py
# echo "PCA model trained."
python get_pca_features.py
echo "PCA features extracted."
# python get_clip_features.py
# echo "CLIP features extracted."
# python get_resnet_features.py
# echo "ResNet features extracted."