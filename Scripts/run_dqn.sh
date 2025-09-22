#!/bin/bash
#SBATCH --job-name=dqn_craftax_actions        # Job name
#SBATCH --partition=bigbatch                     # Replace with your cluster's GPU partition name
#SBATCH --output=/home-mscluster/dharvey/HiSD/Experiments/dqn_actions_stonepick.out  # Standard output and error log
# Load your environment

source ~/.bashrc
conda activate SOTA


python dqn_basic_actions.py
echo "Done"

