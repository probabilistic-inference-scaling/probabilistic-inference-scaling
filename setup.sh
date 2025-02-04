# #!/bin/bash

# # Set the target directory
# TARGET_DIR="/home/lab/shiv"

# # Create the target directory if it doesn't exist
# if [ ! -d "$TARGET_DIR" ]; then
#   echo "Creating target directory: $TARGET_DIR"
#   mkdir -p "$TARGET_DIR"
# fi

# # Navigate to the target directory
# cd "$TARGET_DIR"

# # Clone the repository
# GITHUB_REPO="https://github.com/ishapuri/probabilistic_inference_scaling.git"
# GITHUB_TOKEN="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"

# echo "Cloning the repository..."
# git clone https://$GITHUB_TOKEN@${GITHUB_REPO#https://} || { echo "Failed to clone repository"; exit 1; }

# # Extract the repository name from the URL
# REPO_NAME="$(basename -s .git $GITHUB_REPO)"

# # Navigate into the cloned repository directory
# cd "$REPO_NAME" || { echo "Failed to enter the repository directory"; exit 1; }

# Ensure conda is initialized
echo "Initializing Conda..."
eval "$(conda shell.bash hook)" || { echo "Failed to initialize Conda"; exit 1; }

# Set up the Conda environment
echo "Setting up Conda environment..."
conda create -n sal python=3.10 -y || { echo "Failed to create Conda environment"; exit 1; }
conda activate sal || { echo "Failed to activate Conda environment"; exit 1; }

# Install the required packages
echo "Installing required packages..."
pip install 'vllm==0.6.3'
python -m pip install flash-attn --no-build-isolation
pip install -e .
pip install transformers==4.47.1
pip install openai==1.59.3
pip install huggingface-hub==0.27.0
pip install pydantic==2.10.4
pip install pyarrow==18.1.0

echo "Environment setup complete."