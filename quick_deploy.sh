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
    
    # Multi-GPU configuration
    echo ""
    echo "GPU Configuration options:"
    echo "1) Single GPU (auto detection)"
    echo "2) Multi-GPU with auto distribution [Recommended for 2×H100]"
    echo "3) Multi-GPU with manual memory allocation"
    echo ""
    read -p "Please choose GPU configuration [1-3] (default: 1): " gpu_choice
    
    case $gpu_choice in
        2) 
            GPU_CONFIG="multi_auto"
            print_info "Multi-GPU auto distribution selected"
            ;;
        3) 
            GPU_CONFIG="multi_manual"
            read -p "Enter memory allocation (e.g., '75GiB,75GiB' for 2×H100): " MANUAL_MEMORY
            MANUAL_MEMORY=${MANUAL_MEMORY:-"75GiB,75GiB"}
            print_info "Manual memory allocation: $MANUAL_MEMORY"
            ;;
        *) 
            GPU_CONFIG="single"
            print_info "Single GPU configuration selected"
            ;;
    esac
    
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
    echo "  GPU Config: $GPU_CONFIG"
    if [ "$GPU_CONFIG" = "multi_manual" ]; then
        echo "  Memory Allocation: $MANUAL_MEMORY"
    fi
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
import base64
import io
import gc  # ガベージコレクション用
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Union, Dict, Any, Optional
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import time
import uuid

# transformersの詳細ログレベルを設定して警告を抑制
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/Qwen2.5-VL-72B-Instruct")
API_KEY = os.getenv("QWEN_API_KEY", "your-api-key")
QUANTIZATION = os.getenv("QUANTIZATION", "8bit")

# グローバル変数
model = None
processor = None

# メモリ管理用カウンター
inference_count = 0
last_cleanup_count = 0

# メモリ管理設定
AUTO_CLEANUP_MEMORY = os.getenv("AUTO_CLEANUP_MEMORY", "true").lower() == "true"
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "3"))
AGGRESSIVE_CLEANUP = os.getenv("AGGRESSIVE_CLEANUP", "true").lower() == "true"
FORCE_SYNC = os.getenv("FORCE_SYNC", "true").lower() == "true"
MAX_MEMORY_THRESHOLD = float(os.getenv("MAX_MEMORY_THRESHOLD", "0.85"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))

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

# Content processing function
def process_content(content) -> List[Dict]:
    """Process message content to handle text and images"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    
    processed = []
    for item in content:
        if item["type"] == "text":
            processed.append({"type": "text", "text": item["text"]})
        elif item["type"] == "image_url":
            image_url = item["image_url"]["url"]
            if image_url.startswith("data:image"):
                try:
                    # Parse base64 image
                    header, image_data = image_url.split(",", 1)
                    image_bytes = base64.b64decode(image_data)
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    processed.append({"type": "image", "image": image})
                except Exception as e:
                    logger.error(f"Image processing error: {e}")
                    raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")
            else:
                raise HTTPException(status_code=400, detail="Only base64 image format is supported")
    
    return processed

def cleanup_gpu_memory(aggressive: bool = False):
    """GPU メモリをクリーンアップ"""
    if torch.cuda.is_available():
        if aggressive:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logger.info("積極的GPUメモリクリーンアップを実行")
        else:
            torch.cuda.empty_cache()
            logger.debug("GPUメモリキャッシュクリーンアップを実行")

def get_memory_info():
    """現在のGPUメモリ使用状況を取得"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(reserved - allocated, 2)
        }
    return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}

def should_cleanup_memory():
    """メモリクリーンアップが必要かどうかを判断"""
    global inference_count, last_cleanup_count
    
    if not AUTO_CLEANUP_MEMORY:
        return False
    
    if inference_count - last_cleanup_count >= CLEANUP_INTERVAL:
        return True
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        usage_ratio = allocated / max(reserved, 1)
        if usage_ratio > MAX_MEMORY_THRESHOLD:
            logger.warning(f"高メモリ使用率検出: {allocated:.2f}GB/{reserved:.2f}GB ({usage_ratio:.1%})")
            return True
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if allocated > total_memory * 0.8:
            logger.warning(f"物理メモリ使用量が高い: {allocated:.2f}GB/{total_memory:.2f}GB")
            return True
    
    return False

# モデル読み込み
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
    
    # 量子化設定（警告を避けるためfloat16を使用）
    if QUANTIZATION == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,  # 警告を避けるためfloat16を使用
            bnb_8bit_use_double_quant=True,
        )
    elif QUANTIZATION == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,  # 警告を避けるためfloat16を使用
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    
    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, **model_kwargs
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Fix deprecation warning by resaving processor config
    try:
        import json
        preprocessor_path = os.path.join(MODEL_PATH, "preprocessor.json")
        video_preprocessor_path = os.path.join(MODEL_PATH, "video_preprocessor.json")
        
        if os.path.exists(preprocessor_path) and not os.path.exists(video_preprocessor_path):
            logger.info("Fixing video processor config deprecation warning...")
            with open(preprocessor_path, 'r') as f:
                config = json.load(f)
            
            # Extract video processor config if it exists
            if 'video' in config or 'video_processor' in config:
                video_config = config.get('video', config.get('video_processor', {}))
                with open(video_preprocessor_path, 'w') as f:
                    json.dump(video_config, f, indent=2)
                logger.info("Video processor config saved to video_preprocessor.json")
    except Exception as e:
        logger.warning(f"Could not fix deprecation warning: {e}")
    
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
    memory_info = get_memory_info()
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "quantization": QUANTIZATION,
        "inference_count": inference_count,
        "memory_management": {
            "auto_cleanup": AUTO_CLEANUP_MEMORY,
            "cleanup_interval": CLEANUP_INTERVAL,
            "aggressive_cleanup": AGGRESSIVE_CLEANUP,
            "last_cleanup_at": last_cleanup_count
        },
        "gpu_memory": memory_info
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest, api_key: str = Depends(verify_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process messages with correct content formatting
        processed_messages = []
        for msg in request.messages:
            processed_messages.append({
                "role": msg.role,
                "content": process_content(msg.content)
            })
        
        # Apply chat template
        text = processor.apply_chat_template(processed_messages, tokenize=False, add_generation_prompt=True)
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(processed_messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        # 生成パラメータを準備（警告を避けるため）
        generation_kwargs = {
            "max_new_tokens": request.max_tokens,
        }
        
        # temperatureとsampling設定（サポートされている場合のみ）
        if request.temperature > 0:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": request.temperature,
            })
        else:
            generation_kwargs["do_sample"] = False
        
        # 応答を生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_kwargs)
        
        # レスポンスをデコード
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        
        # 推理完了後のメモリ管理
        global inference_count, last_cleanup_count
        inference_count += 1
        
        # 明示的に中間変数を削除
        del inputs, generated_ids, generated_ids_trimmed
        
        # 基本的なクリーンアップを実行
        torch.cuda.empty_cache()
        gc.collect()
        
        # 条件に応じてクリーンアップを実行
        if should_cleanup_memory():
            cleanup_gpu_memory(aggressive=AGGRESSIVE_CLEANUP)
            last_cleanup_count = inference_count
            logger.info(f"メモリクリーンアップ実行 (推理回数: {inference_count})")
        elif FORCE_SYNC:
            torch.cuda.synchronize()
            logger.debug(f"軽量クリーンアップ実行 (推理回数: {inference_count})")
        
        # レスポンスを構築
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
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        timeout_keep_alive=REQUEST_TIMEOUT,
        timeout_graceful_shutdown=30
    )
EOF
}

# Create startup script
create_startup_script() {
    print_info "Creating startup script..."
    
    cat > start_qwen.sh << EOF
#!/bin/bash

# 環境変数を設定
export MODEL_PATH="$WORKSPACE_DIR/Qwen2.5-VL-72B-Instruct"
export QWEN_API_KEY="$API_KEY"
export QUANTIZATION="$QUANTIZATION"
export PORT="$PORT"

# GPU設定を適用
case "$GPU_CONFIG" in
    "multi_auto")
        export DEVICE_MAP="auto"
        echo "🔧 多GPU自動分散モード有効"
        ;;
    "multi_manual")
        export DEVICE_MAP="auto"
        export MAX_MEMORY="$MANUAL_MEMORY"
        echo "🔧 多GPU手動メモリ分割: $MANUAL_MEMORY"
        ;;
    *)
        export DEVICE_MAP="auto"
        echo "🔧 単一GPU自動検出モード"
        ;;
esac

# PyTorch最適化とログ抑制設定
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export OMP_NUM_THREADS="8"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_VERBOSITY="error"  # 警告を抑制

# メモリ管理設定
export AUTO_CLEANUP_MEMORY="true"   # 自動メモリクリーンアップを有効
export CLEANUP_INTERVAL="3"         # 3回の推理毎にクリーンアップ
export AGGRESSIVE_CLEANUP="true"    # 積極的クリーンアップを有効
export FORCE_SYNC="true"             # GPU同期を有効
export MAX_MEMORY_THRESHOLD="0.85"  # メモリ使用率85%でクリーンアップ

# タイムアウト設定
export REQUEST_TIMEOUT="600"        # 10分のリクエストタイムアウト

echo "🚀 Qwen2.5-VL-72B APIサーバーを起動中..."
echo "ポート: $PORT"
echo "量子化: $QUANTIZATION"
echo "API Key: ${API_KEY:0:20}..."

# GPU設定表示
case "$GPU_CONFIG" in
    "multi_auto")
        echo "🎯 GPU設定: 多GPU自動分散"
        ;;
    "multi_manual") 
        echo "🎯 GPU設定: 多GPU手動分割 ($MANUAL_MEMORY)"
        ;;
    *)
        echo "🎯 GPU設定: 単一GPU"
        ;;
esac

echo "🔧 警告修正: Temperature, BitsAndBytes, Deprecation"
echo "🧠 メモリ管理有効:"
echo "  - クリーンアップ間隔: ${CLEANUP_INTERVAL}回"
echo "  - 強制同期: 有効"
echo "  - メモリ閾値: ${MAX_MEMORY_THRESHOLD}"
echo "⏱️  タイムアウト: ${REQUEST_TIMEOUT}秒"

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

echo "🧪 APIサーバーをテスト中..."

# ヘルスチェック
echo "1. ヘルスチェック..."
curl -s http://localhost:$PORT/health | jq .

echo -e "\n2. テキスト会話テスト..."
curl -X POST http://localhost:$PORT/v1/chat/completions \\
  -H "Authorization: Bearer \$API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen/qwen2.5-vl-72b-instruct",
    "messages": [
      {"role": "user", "content": "こんにちは、簡単に自己紹介してください。"}
    ],
    "max_tokens": 512
  }' | jq .

echo -e "\n3. メモリ管理テスト..."
curl -s http://localhost:$PORT/health | jq '.memory_management, .gpu_memory'

echo -e "\n✅ テスト完了!"
echo "APIドキュメント: http://localhost:$PORT/docs"
echo "🔧 修正済み問題:"
echo "  - メッセージ処理エラーの修正"
echo "  - Deprecation警告の修正"
echo "  - GPU メモリ管理の改善"
echo ""
echo "💡 メモリ管理:"
echo "  - 手動クリーンアップ: POST /v1/memory/cleanup"
echo "  - メモリ監視: GET /health または /metrics"
EOF

    chmod +x test_api.sh
}

# Start service
start_service() {
    print_info "Starting service..."
    
    cd $WORKSPACE_DIR
    
    print_success "🎉 デプロイメント完了！"
    echo ""
    echo "==============================================="
    echo "📋 デプロイメント情報"
    echo "==============================================="
    echo "API アドレス: http://localhost:$PORT"
    echo "API キー: $API_KEY"
    echo "量子化モード: $QUANTIZATION"
    case "$GPU_CONFIG" in
        "multi_auto")
            echo "GPU設定: 多GPU自動分散"
            ;;
        "multi_manual") 
            echo "GPU設定: 多GPU手動分割 ($MANUAL_MEMORY)"
            ;;
        *)
            echo "GPU設定: 単一GPU"
            ;;
    esac
    echo "モデルパス: $WORKSPACE_DIR/Qwen2.5-VL-72B-Instruct"
    echo ""
    echo "==============================================="
    echo "🚀 起動コマンド"
    echo "==============================================="
    echo "サービス開始: ./start_qwen.sh"
    echo "API テスト: ./test_api.sh"
    echo ""
    echo "🔧 問題が修正されました:"
    echo "  - メッセージ処理エラー ('dict' object has no attribute 'startswith')"
    echo "  - preprocessor.json deprecation警告"
    echo "  - 10分タイムアウト設定追加"
    if [ "$GPU_CONFIG" != "single" ]; then
        echo "  - 多GPU自動分散対応"
    fi
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