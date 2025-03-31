import os
from huggingface_hub import snapshot_download

# Define the target folder
target_dir = "Traces"

# Create the folder if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Download the dataset into the 'Traces' folder
snapshot_download(
    repo_id="dami2106/Craftax-Skill-Data",
    repo_type="dataset",
    local_dir=target_dir,
    local_dir_use_symlinks=False
)

print(f"Dataset downloaded to: {os.path.abspath(target_dir)}")
