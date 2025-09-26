#!/bin/bash
#SBATCH --job-name=ppo_options_craftax        # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/ppo_options_craftax.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python ppo_skills.py
echo "Done"

