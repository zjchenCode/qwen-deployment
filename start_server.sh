 #!/bin/bash

# Qwen2.5-VL-72B Server Startup Script
# Optimized for high-performance deployment

set -e

# Color codes
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
WORKSPACE="${WORKSPACE:-/workspace}"
MODEL_PATH="${MODEL_PATH:-/workspace/Qwen2.5-VL-72B-Instruct}"
API_KEY="${QWEN_API_KEY:-sk-qwen25-vl-72b--demo-key}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Performance and memory settings
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-false}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
MAX_MEMORY="${MAX_MEMORY:-}"
USE_FLASH_ATTENTION="${USE_FLASH_ATTENTION:-true}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-/tmp/offload}"

# Server settings
WORKERS="${WORKERS:-1}"
MAX_REQUESTS="${MAX_REQUESTS:-50}"
TIMEOUT="${TIMEOUT:-300}"

# Show help
show_help() {
    echo "Qwen2.5-VL-72B Server Startup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Model Configuration:"
    echo "  --model-path PATH          Model directory path (default: $MODEL_PATH)"
    echo "  --api-key KEY             API key for authentication (default: from env)"
    echo "  --torch-dtype TYPE        Torch dtype: bfloat16|float16|float32 (default: $TORCH_DTYPE)"
    echo "  --device-map MAP          Device mapping strategy (default: $DEVICE_MAP)"
    echo "  --max-memory MEM          Max memory per GPU, e.g., '20GiB,20GiB' (default: auto)"
    echo ""
    echo "Quantization Options:"
    echo "  --load-in-8bit            Enable 8-bit quantization"
    echo "  --load-in-4bit            Enable 4-bit quantization (saves more memory)"
    echo "  --no-flash-attention      Disable Flash Attention"
    echo ""
    echo "Server Options:"
    echo "  --host HOST               Host to bind to (default: $HOST)"
    echo "  --port PORT               Port to bind to (default: $PORT)"
    echo "  --workers N               Number of worker processes (default: $WORKERS)"
    echo "  --max-requests N          Max concurrent requests (default: $MAX_REQUESTS)"
    echo "  --timeout SECONDS         Request timeout (default: $TIMEOUT)"
    echo ""
    echo "Advanced Options:"
    echo "  --offload-folder PATH     CPU offload directory (default: $OFFLOAD_FOLDER)"
    echo "  --workspace PATH          Workspace directory (default: $WORKSPACE)"
    echo "  --debug                   Enable debug mode"
    echo "  --profile                 Enable performance profiling"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Start with defaults"
    echo "  $0 --load-in-8bit --port 8080        # Start with 8-bit quantization"
    echo "  $0 --max-memory '20GiB,20GiB'        # Limit memory per GPU"
    echo "  $0 --load-in-4bit --workers 2        # 4-bit quant with 2 workers"
}

# Parse command line arguments
DEBUG_MODE=false
PROFILE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --torch-dtype)
            TORCH_DTYPE="$2"
            shift 2
            ;;
        --device-map)
            DEVICE_MAP="$2"
            shift 2
            ;;
        --max-memory)
            MAX_MEMORY="$2"
            shift 2
            ;;
        --load-in-8bit)
            LOAD_IN_8BIT=true
            shift
            ;;
        --load-in-4bit)
            LOAD_IN_4BIT=true
            shift
            ;;
        --no-flash-attention)
            USE_FLASH_ATTENTION=false
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --max-requests)
            MAX_REQUESTS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --offload-folder)
            OFFLOAD_FOLDER="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --profile)
            PROFILE_MODE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# System check function
check_system() {
    print_info "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not found"
        exit 1
    fi
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        print_success "Found $gpu_count GPU(s)"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | \
        while IFS=, read -r name memory driver; do
            print_info "GPU: $name, Memory: ${memory}MB, Driver: $driver"
        done
    else
        print_warning "NVIDIA GPU not detected - will use CPU (very slow for 72B model)"
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        ram_gb=$(free -g | awk '/^Mem:/{print $2}')
        print_info "System RAM: ${ram_gb}GB"
        if [ "$ram_gb" -lt 32 ]; then
            print_warning "Low system RAM detected. Consider using quantization options."
        fi
    fi
    
    # Check disk space
    if [ -d "$MODEL_PATH" ]; then
        model_size=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1 || echo "Unknown")
        print_info "Model size: $model_size"
    else
        print_error "Model path not found: $MODEL_PATH"
        print_info "Run the download script first: bash download_models.sh"
        exit 1
    fi
}

# Environment setup
setup_environment() {
    print_info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p "$OFFLOAD_FOLDER"
    mkdir -p "$WORKSPACE/.cache"
    
    # Set environment variables
    export QWEN_API_KEY="$API_KEY"
    export MODEL_PATH="$MODEL_PATH"
    export TORCH_DTYPE="$TORCH_DTYPE"
    export DEVICE_MAP="$DEVICE_MAP"
    export LOAD_IN_8BIT="$LOAD_IN_8BIT"
    export LOAD_IN_4BIT="$LOAD_IN_4BIT"
    export USE_FLASH_ATTENTION="$USE_FLASH_ATTENTION"
    export OFFLOAD_FOLDER="$OFFLOAD_FOLDER"
    export MAX_MEMORY="$MAX_MEMORY"
    
    # PyTorch optimizations
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
    
    # HuggingFace cache
    export TRANSFORMERS_CACHE="$WORKSPACE/.cache/transformers"
    export HF_HOME="$WORKSPACE/.cache/huggingface"
    
    # Performance optimizations
    export OMP_NUM_THREADS="8"
    export TOKENIZERS_PARALLELISM="false"
    
    if [ "$DEBUG_MODE" = true ]; then
        export CUDA_LAUNCH_BLOCKING="1"
        export TORCH_USE_CUDA_DSA="1"
    fi
    
    print_success "Environment configured"
}

# Dependency check
check_dependencies() {
    print_info "Checking Python dependencies..."
    
    # Check if requirements are installed
    python3 -c "
import torch
import transformers
import fastapi
import uvicorn
from qwen_vl_utils import process_vision_info
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
" || {
        print_error "Missing dependencies. Installing..."
        pip install -r requirements.txt
    }
    
    # Check Flash Attention if enabled
    if [ "$USE_FLASH_ATTENTION" = true ]; then
        python3 -c "import flash_attn" 2>/dev/null || {
            print_warning "Flash Attention not installed. Installing..."
            pip install flash-attn --no-build-isolation || {
                print_warning "Flash Attention installation failed. Disabling..."
                USE_FLASH_ATTENTION=false
                export USE_FLASH_ATTENTION=false
            }
        }
    fi
}

# Start server function
start_server() {
    print_info "Starting Qwen2.5-VL-72B API server..."
    
    # Server startup configuration
    local uvicorn_args=(
        "server:app"
        "--host" "$HOST"
        "--port" "$PORT"
        "--workers" "$WORKERS"
        "--timeout-keep-alive" "$TIMEOUT"
        "--access-log"
        "--log-level" "info"
    )
    
    if [ "$DEBUG_MODE" = true ]; then
        uvicorn_args+=("--reload" "--log-level" "debug")
    fi
    
    # Print configuration summary
    echo ""
    print_info "=== Server Configuration ==="
    echo "Model Path: $MODEL_PATH"
    echo "API Key: ${API_KEY:0:10}..."
    echo "Host: $HOST"
    echo "Port: $PORT"
    echo "Workers: $WORKERS"
    echo "Torch Dtype: $TORCH_DTYPE"
    echo "Device Map: $DEVICE_MAP"
    echo "8-bit Quantization: $LOAD_IN_8BIT"
    echo "4-bit Quantization: $LOAD_IN_4BIT"
    echo "Flash Attention: $USE_FLASH_ATTENTION"
    echo "Max Memory: ${MAX_MEMORY:-auto}"
    echo "Debug Mode: $DEBUG_MODE"
    echo ""
    
    # Change to workspace directory
    cd "$WORKSPACE"
    
    # Start the server
    if [ "$PROFILE_MODE" = true ]; then
        print_info "Starting with profiling enabled..."
        python3 -m cProfile -o server_profile.prof server.py
    else
        python3 -m uvicorn "${uvicorn_args[@]}"
    fi
}

# Health check function
health_check() {
    local max_attempts=30
    local attempt=1
    
    print_info "Waiting for server to start..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
            print_success "Server is healthy and ready!"
            print_info "API Documentation: http://$HOST:$PORT/docs"
            print_info "Health Check: http://$HOST:$PORT/health"
            return 0
        fi
        
        sleep 2
        ((attempt++))
    done
    
    print_error "Server failed to start within expected time"
    return 1
}

# Cleanup function
cleanup() {
    print_info "Cleaning up..."
    # Kill background processes if any
    jobs -p | xargs -r kill
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    print_info "=== Qwen2.5-VL-72B Server Startup ==="
    echo ""
    
    check_system
    setup_environment
    check_dependencies
    
    # Start server in background for health check
    start_server &
    SERVER_PID=$!
    
    # Wait a bit then check health
    sleep 10
    if health_check; then
        print_success "Server started successfully!"
        print_info "Server PID: $SERVER_PID"
        print_info "Press Ctrl+C to stop the server"
        
        # Wait for server process
        wait $SERVER_PID
else
        print_error "Server startup failed"
        kill $SERVER_PID 2>/dev/null
    exit 1
    fi
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi