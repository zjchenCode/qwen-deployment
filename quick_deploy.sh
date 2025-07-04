#!/bin/bash

# Qwen2.5-VL-72B One-Click Deployment Script
# For rapid deployment on rental servers

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default configuration
DEFAULT_API_KEY="sk-qwen25-vl-72b--$(date +%s)"
DEFAULT_PORT="8000"
DEFAULT_QUANTIZATION="8bit"  # none, 8bit, 4bit
WORKSPACE_DIR="/workspace"

# Show welcome message
show_welcome() {
    echo ""
    echo "==============================================="
    echo "🚀 Qwen2.5-VL-72B One-Click Deployment Script"
    echo "==============================================="
    echo ""
    echo "This script will help you quickly deploy Qwen2.5-VL-72B model"
    echo "Suitable for RunPod, Vast.ai, AutoDL and other rental servers"
    echo ""
}

# System check
check_system() {
    print_info "Checking system environment..."
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root detected, recommend using regular user"
    fi
    
    # Check operating system
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_info "Operating System: $NAME $VERSION"
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        python_version=$(python3 --version)
        print_success "Python: $python_version"
    else
        print_error "Python3 not installed"
        exit 1
    fi
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "GPU: $gpu_info"
        
        # Check GPU memory
        gpu_memory=$(echo $gpu_info | cut -d',' -f2 | tr -d ' ')
        if [ "$gpu_memory" -lt 40000 ]; then
            print_warning "GPU memory less than 40GB, strongly recommend using quantization"
            DEFAULT_QUANTIZATION="4bit"
        elif [ "$gpu_memory" -lt 70000 ]; then
            print_warning "GPU memory less than 70GB, recommend using 8bit quantization"
            DEFAULT_QUANTIZATION="8bit"
        fi
    else
        print_error "NVIDIA GPU not detected"
        exit 1
    fi
    
    # Check disk space
    available_space=$(df $WORKSPACE_DIR | awk 'NR==2 {print int($4/1024/1024)}')
    print_info "Available disk space: ${available_space}GB"
    
    if [ "$available_space" -lt 200 ]; then
        print_error "Insufficient disk space, at least 200GB required"
        exit 1
    fi
}

# User configuration
configure_deployment() {
    print_info "Configuring deployment parameters..."
    echo ""
    
    # API key
    read -p "Please set API key (default: $DEFAULT_API_KEY): " API_KEY
    API_KEY=${API_KEY:-$DEFAULT_API_KEY}
    
    # Port
    read -p "Please set API port (default: $DEFAULT_PORT): " PORT
    PORT=${PORT:-$DEFAULT_PORT}
    
    # Quantization options
    echo ""
    echo "Quantization options:"
    echo "1) No quantization (requires ~144GB GPU memory)"
    echo "2) 8-bit quantization (requires ~72GB GPU memory) [Recommended]"
    echo "3) 4-bit quantization (requires ~36GB GPU memory)"
    echo ""
    read -p "Please choose quantization option [1-3] (default: 2): " quant_choice
    
    case $quant_choice in
        1) QUANTIZATION="none" ;;
        3) QUANTIZATION="4bit" ;;
        *) QUANTIZATION="8bit" ;;
    esac
    
    # Confirm configuration
    echo ""
    print_info "Deployment configuration confirmation:"
    echo "  API Key: ${API_KEY:0:20}..."
    echo "  Port: $PORT"
    echo "  Quantization: $QUANTIZATION"
    echo ""
    
    read -p "Confirm to start deployment? [y/N]: " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        print_info "Deployment cancelled"
        exit 0
    fi
}

# Install dependencies
install_dependencies() {
    print_info "Installing system dependencies..."
    
    # Update package manager
    if command -v apt &> /dev/null; then
        apt update -y
        apt install -y curl wget git build-essential
    elif command -v yum &> /dev/null; then
        yum update -y
        yum install -y curl wget git gcc gcc-c++ make
    fi
    
    # Install Python dependencies
    print_info "Installing Python dependencies..."
    pip install --upgrade pip
    
    # Create requirements file
    cat > requirements_quick.txt << EOF
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.37.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
qwen-vl-utils
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
Pillow>=10.0.0
psutil>=5.9.0
requests>=2.31.0
safetensors>=0.4.0
EOF
    
    pip install -r requirements_quick.txt
    
    # Try installing Flash Attention
    print_info "Installing Flash Attention..."
    pip install flash-attn --no-build-isolation || {
        print_warning "Flash Attention installation failed, continuing without Flash Attention"
    }
}

# Download model
download_model() {
    print_info "Downloading Qwen2.5-VL-72B model..."
    
    # Set model directory
    MODEL_DIR="$WORKSPACE_DIR/Qwen2.5-VL-72B-Instruct"
    
    if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
        print_warning "Model directory already exists, skipping download"
        return 0
    fi
    
    # Install huggingface-hub
    pip install huggingface-hub[hf_transfer] tqdm
    
    # Set download optimization
    export HF_HUB_ENABLE_HF_TRANSFER=1
    
    # Download model
    python3 -c "
import os
from huggingface_hub import snapshot_download
from tqdm import tqdm

print('Starting model download...')
try:
    snapshot_download(
        repo_id='Qwen/Qwen2.5-VL-72B-Instruct',
        local_dir='$MODEL_DIR',
        resume_download=True,
        local_dir_use_symlinks=False
    )
    print('Model download completed!')
except Exception as e:
    print(f'Model download failed: {e}')
    exit(1)
"
}

# Create server script
create_server() {
    print_info "Creating API server..."
    
    # Create simplified server script
    cat > qwen_server.py << 'EOF'
import os
import torch
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Union, Dict, Any, Optional
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import time
import uuid

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/Qwen2.5-VL-72B-Instruct")
API_KEY = os.getenv("QWEN_API_KEY", "your-api-key")
QUANTIZATION = os.getenv("QUANTIZATION", "8bit")

# Global variables
model = None
processor = None

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict

# API key verification
def verify_api_key(authorization: str = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization.replace("Bearer ", "")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token

# Load model
def load_model():
    global model, processor
    
    logger.info(f"Loading model from {MODEL_PATH}")
    logger.info(f"Quantization: {QUANTIZATION}")
    
    # Model loading parameters
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    
    # Quantization configuration
    if QUANTIZATION == "8bit":
        model_kwargs["load_in_8bit"] = True
    elif QUANTIZATION == "4bit":
        model_kwargs["load_in_4bit"] = True
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, **model_kwargs
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    logger.info("Model loaded successfully!")

# Create FastAPI application
app = FastAPI(title="Qwen2.5-VL-72B API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "Qwen2.5-VL-72B API Server", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "quantization": QUANTIZATION
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process messages
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        # Build response
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:29]}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": response_text.strip()},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(text.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(text.split()) + len(response_text.split())
            }
        )
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
EOF
}

# Create startup script
create_startup_script() {
    print_info "Creating startup script..."
    
    cat > start_qwen.sh << EOF
#!/bin/bash

# Set environment variables
export MODEL_PATH="$WORKSPACE_DIR/Qwen2.5-VL-72B-Instruct"
export QWEN_API_KEY="$API_KEY"
export QUANTIZATION="$QUANTIZATION"
export PORT="$PORT"

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export OMP_NUM_THREADS="8"
export TOKENIZERS_PARALLELISM="false"

echo "🚀 Starting Qwen2.5-VL-72B API Server..."
echo "Port: $PORT"
echo "Quantization: $QUANTIZATION"
echo "API Key: ${API_KEY:0:20}..."

cd $WORKSPACE_DIR
python3 qwen_server.py
EOF

    chmod +x start_qwen.sh
}

# Create test script
create_test_script() {
    print_info "Creating test script..."
    
    cat > test_api.sh << EOF
#!/bin/bash

API_KEY="$API_KEY"
PORT="$PORT"

echo "🧪 Testing API server..."

# Health check
echo "1. Health check..."
curl -s http://localhost:$PORT/health | jq .

echo -e "\n2. Text conversation test..."
curl -X POST http://localhost:$PORT/v1/chat/completions \\
  -H "Authorization: Bearer \$API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen/qwen2.5-vl-72b-instruct",
    "messages": [
      {"role": "user", "content": "Hello, please introduce yourself briefly."}
    ],
    "max_tokens": 512
  }' | jq .

echo -e "\n✅ Test completed!"
echo "API Documentation: http://localhost:$PORT/docs"
EOF

    chmod +x test_api.sh
}

# Start service
start_service() {
    print_info "Starting service..."
    
    cd $WORKSPACE_DIR
    
    print_success "🎉 Deployment completed!"
    echo ""
    echo "==============================================="
    echo "📋 Deployment Information"
    echo "==============================================="
    echo "API Address: http://localhost:$PORT"
    echo "API Key: $API_KEY"
    echo "Quantization Mode: $QUANTIZATION"
    echo "Model Path: $WORKSPACE_DIR/Qwen2.5-VL-72B-Instruct"
    echo ""
    echo "==============================================="
    echo "🚀 Startup Commands"
    echo "==============================================="
    echo "Start service: ./start_qwen.sh"
    echo "Test API: ./test_api.sh"
    echo ""
    
    read -p "Start service now? [y/N]: " start_now
    if [[ $start_now =~ ^[Yy]$ ]]; then
        ./start_qwen.sh
    else
        print_info "Please manually run ./start_qwen.sh to start the service"
    fi
}

# Main function
main() {
    show_welcome
    check_system
    configure_deployment
    
    cd $WORKSPACE_DIR
    
    install_dependencies
    download_model
    create_server
    create_startup_script
    create_test_script
    start_service
}

# Check if script is being executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 