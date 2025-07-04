 #!/bin/bash

# RunPod Environment Setup Script for Qwen Models
echo "Setting up environment for Qwen2.5 models..."

# Update system packages
apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    vim \
    htop \
    screen

# Set environment variables
export PYTHONPATH=/workspace:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Install Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip

# Core ML packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers and related
pip install transformers==4.46.0
pip install accelerate>=0.26.0
pip install tokenizers>=0.15.0

# Qwen specific packages
pip install qwen-vl-utils
pip install transformers-stream-generator

# FastAPI and web server
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart

# Additional utilities
pip install pydantic==2.5.0
pip install pillow>=9.5.0
pip install numpy>=1.24.0
pip install requests>=2.28.0

# Flash Attention (if needed)
pip install flash-attn --no-build-isolation

# Optional: Better performance packages
pip install ninja
pip install einops

echo "Environment setup completed!"

# Create directories
mkdir -p /workspace/models
mkdir -p /workspace/logs
mkdir -p /workspace/configs

# Set permissions
chmod +x /workspace/*.py

echo "Ready to download models and start server!"
echo "Usage:"
echo "  python3 server.py --models qwen2.5-vl qwen2.5-chat"