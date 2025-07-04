 #!/bin/bash

# Qwen2.5-VL-72B Model Download Script
# This script downloads the Qwen2.5-VL-72B-Instruct model from Hugging Face

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
MODEL_NAME="Qwen/Qwen2.5-VL-72B-Instruct"
MODEL_DIR="/workspace/Qwen2.5-VL-72B-Instruct"
HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-"/workspace/.cache/huggingface"}
HF_TOKEN=${HF_TOKEN:-""}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    print_warning "Not running as root. Make sure you have write permissions to $MODEL_DIR"
fi

# Function to check available disk space
check_disk_space() {
    local required_space_gb=200  # 72B model requires approximately 200GB
    local available_space_gb=$(df /workspace | awk 'NR==2 {print int($4/1024/1024)}')
    
    print_info "Checking disk space..."
    print_info "Available space: ${available_space_gb}GB"
    print_info "Required space: ${required_space_gb}GB"

    if [ $available_space_gb -lt $required_space_gb ]; then
        print_error "Insufficient disk space. Need at least ${required_space_gb}GB, but only ${available_space_gb}GB available."
        exit 1
    fi
    print_success "Disk space check passed"
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing/updating dependencies..."
    
    # Update pip
    python -m pip install --upgrade pip
    
    # Install huggingface-hub with download capabilities
    pip install huggingface-hub[hf_transfer] tqdm
    
    # Install hf_transfer for faster downloads (optional but recommended)
    pip install hf_transfer
    export HF_HUB_ENABLE_HF_TRANSFER=1
    
    print_success "Dependencies installed"
}

# Function to authenticate with Hugging Face
setup_authentication() {
    if [ -n "$HF_TOKEN" ]; then
        print_info "Using provided HF_TOKEN for authentication"
        huggingface-cli login --token $HF_TOKEN
    else
        print_warning "No HF_TOKEN provided. Some models may not be accessible."
        print_info "To set token: export HF_TOKEN='your_token_here'"
    fi
}

# Function to download model using huggingface-hub
download_with_hf_hub() {
    print_info "Starting download of $MODEL_NAME to $MODEL_DIR"
    print_info "This will take a significant amount of time due to the large model size (72B parameters)"
    
    # Create target directory
    mkdir -p "$MODEL_DIR"
    
    # Set cache directory
    export HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
    
    # Download with resume capability and progress tracking
        python3 -c "
import os
from huggingface_hub import snapshot_download
from tqdm import tqdm
import sys

def progress_callback(filename, bytes_downloaded, total_bytes):
    if total_bytes > 0:
        percentage = (bytes_downloaded / total_bytes) * 100
        print(f'\r{filename}: {percentage:.1f}% ({bytes_downloaded}/{total_bytes} bytes)', end='')
    else:
        print(f'\r{filename}: {bytes_downloaded} bytes downloaded', end='')

try:
    print('Starting model download...')
    snapshot_download(
        repo_id='$MODEL_NAME',
        local_dir='$MODEL_DIR',
        resume_download=True,
        local_dir_use_symlinks=False,
        token=os.getenv('HF_TOKEN') if os.getenv('HF_TOKEN') else None
    )
    print('\nDownload completed successfully!')
except Exception as e:
    print(f'\nError during download: {e}')
    sys.exit(1)
"
}

# Function to verify download
verify_download() {
    print_info "Verifying download..."
        
    # Check for essential files
    essential_files=(
        "config.json"
        "model.safetensors.index.json"
        "tokenizer.json"
        "tokenizer_config.json"
        "preprocessor_config.json"
    )
    
    for file in "${essential_files[@]}"; do
        if [ ! -f "$MODEL_DIR/$file" ]; then
            print_error "Missing essential file: $file"
            return 1
        fi
    done
    
    # Check for model weight files
    if ! ls "$MODEL_DIR"/model-*.safetensors >/dev/null 2>&1; then
        print_error "No model weight files found (model-*.safetensors)"
        return 1
    fi
    
    print_success "Download verification passed"
    return 0
}

# Function to display download summary
show_summary() {
    local model_size=$(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1 || echo "Unknown")
    local file_count=$(find "$MODEL_DIR" -type f | wc -l)
    
    print_success "Download Summary:"
    echo "  Model: $MODEL_NAME"
    echo "  Location: $MODEL_DIR"
    echo "  Size: $model_size"
    echo "  Files: $file_count"
    echo ""
    print_info "Model is ready for use!"
    print_info "You can now start the server with: bash start_server.sh"
}

# Function to handle cleanup on failure
cleanup_on_failure() {
    print_error "Download failed. Cleaning up partial downloads..."
    if [ -d "$MODEL_DIR" ]; then
        print_warning "Removing incomplete model directory: $MODEL_DIR"
        rm -rf "$MODEL_DIR"
    fi
}

# Main execution
main() {
    print_info "=== Qwen2.5-VL-72B Model Download Script ==="
    print_info "Model: $MODEL_NAME"
    print_info "Target directory: $MODEL_DIR"
    echo ""
    
    # Trap to handle interruption
    trap cleanup_on_failure ERR INT TERM
    
    # Check if model already exists
    if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
        print_warning "Model directory already exists and is not empty: $MODEL_DIR"
        echo "Options:"
        echo "  1) Resume/verify download"
        echo "  2) Delete and re-download"
        echo "  3) Exit"
        echo ""
        read -p "Choose option [1-3]: " choice
        
        case $choice in
            1)
                print_info "Resuming/verifying download..."
                ;;
            2)
                print_warning "Deleting existing model directory..."
                rm -rf "$MODEL_DIR"
                print_info "Starting fresh download..."
                ;;
            3)
                print_info "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid choice. Exiting..."
                exit 1
                ;;
        esac
    fi
    
    # Execute download steps
    check_disk_space
    install_dependencies
    setup_authentication
    download_with_hf_hub
    
    if verify_download; then
        show_summary
    else
        print_error "Download verification failed!"
        exit 1
    fi
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi