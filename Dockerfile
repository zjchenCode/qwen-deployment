FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV FORCE_CUDA="1"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flash Attention (if possible)
RUN pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed, continuing without it"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/.cache/huggingface \
    && mkdir -p /workspace/.cache/transformers \
    && mkdir -p /tmp/offload \
    && mkdir -p /workspace/logs

# Set permissions
RUN chmod +x *.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set default environment variables
ENV MODEL_PATH=/workspace/Qwen2.5-VL-72B-Instruct
ENV QWEN_API_KEY=sk-qwen25-vl-72b--demo-key
ENV TORCH_DTYPE=bfloat16
ENV DEVICE_MAP=auto
ENV USE_FLASH_ATTENTION=true

# Startup command
CMD ["python3", "server.py"] 