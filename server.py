import os
import base64
import io
import json
import time
import uuid
import logging
import asyncio
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager
from functools import lru_cache

# transformersの詳細ログレベルを設定して警告を抑制
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
import gc  # ガベージコレクション用
from PIL import Image
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設定クラス
class Config:
    API_KEY = os.getenv("QWEN_API_KEY", "sk-qwen25-vl-72b--1751619091")
    MODEL_NAME = "qwen/qwen2.5-vl-72b-instruct"
    MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/Qwen2.5-VL-72B-Instruct")
    MAX_TOKENS_LIMIT = 8192  # トークン制限を増加
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 画像サイズを20MBに増加
    RATE_LIMIT_REQUESTS = 50  # 同時実行制限を削減
    ALLOWED_HOSTS = ["*"]  # 本番環境では制限すべき
    
    # 72Bモデル用GPU設定
    USE_FLASH_ATTENTION = True
    TORCH_DTYPE = torch.bfloat16
    DEVICE_MAP = "auto"  # 自動マルチGPU割り当て
    LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "true").lower() == "true"
    LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
    MAX_MEMORY = os.getenv("MAX_MEMORY", None)  # GPU毎の最大メモリ設定可能
    OFFLOAD_FOLDER = os.getenv("OFFLOAD_FOLDER", "/tmp/offload")  # CPUオフロードディレクトリ
    
    # メモリ管理設定
    AUTO_CLEANUP_MEMORY = os.getenv("AUTO_CLEANUP_MEMORY", "true").lower() == "true"  # 自動メモリクリーンアップ
    CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "3"))  # クリーンアップ間隔
    AGGRESSIVE_CLEANUP = os.getenv("AGGRESSIVE_CLEANUP", "true").lower() == "true"  # 積極的クリーンアップ
    FORCE_SYNC = os.getenv("FORCE_SYNC", "true").lower() == "true"  # GPU同期
    MAX_MEMORY_THRESHOLD = float(os.getenv("MAX_MEMORY_THRESHOLD", "0.85"))  # メモリ使用率閾値
    
    # タイムアウト設定
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "600"))  # 10分のタイムアウト

config = Config()

# グローバル変数
model = None
processor = None
model_lock = asyncio.Lock()

# シンプルなレート制限用リクエストカウンター
request_counts = {}

# メモリ管理用カウンター
inference_count = 0
last_cleanup_count = 0

def cleanup_gpu_memory(aggressive: bool = False):
    """GPU メモリをクリーンアップ"""
    if torch.cuda.is_available():
        pre_allocated = torch.cuda.memory_allocated() / 1024**3
        
        if aggressive:
            # 完全なメモリクリア
            torch.cuda.empty_cache()
            if config.FORCE_SYNC:
                torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            
            # すべてのGPUデバイスをクリア
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    if config.FORCE_SYNC:
                        torch.cuda.synchronize()
            
            post_allocated = torch.cuda.memory_allocated() / 1024**3
            freed = pre_allocated - post_allocated
            logger.info(f"GPUメモリクリーンアップ完了: {freed:.2f}GB解放")
        else:
            # 通常クリーンアップ
            torch.cuda.empty_cache()
            gc.collect()
            if config.FORCE_SYNC:
                torch.cuda.synchronize()
            
            post_allocated = torch.cuda.memory_allocated() / 1024**3
            freed = pre_allocated - post_allocated
            logger.debug(f"GPUメモリクリーンアップ完了: {freed:.2f}GB解放")

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
    
    if not config.AUTO_CLEANUP_MEMORY:
        return False
    
    # 設定されたインターバル毎にクリーンアップ
    if inference_count - last_cleanup_count >= config.CLEANUP_INTERVAL:
        return True
    
    # メモリ使用率チェック
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        usage_ratio = allocated / max(reserved, 1)
        if usage_ratio > config.MAX_MEMORY_THRESHOLD:
            logger.warning(f"高メモリ使用率検出: {allocated:.2f}GB/{reserved:.2f}GB ({usage_ratio:.1%})")
            return True
        
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if allocated > total_memory * 0.8:
            logger.warning(f"物理メモリ使用量が高い: {allocated:.2f}GB/{total_memory:.2f}GB")
            return True
    
    return False

def load_model():
    """Qwen2.5-VLモデルとプロセッサーを読み込み"""
    global model, processor
    
    logger.info("Loading Qwen2.5-VL-72B model...")
    
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(f"Model path not found: {config.MODEL_PATH}")
    
    try:
        # Check GPU availability
        if not torch.cuda.is_available():
            logger.warning("GPU not available, falling back to CPU (much slower)")
        else:
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Prepare loading arguments
        model_kwargs = {
            "torch_dtype": config.TORCH_DTYPE,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        # Configure device mapping and memory management
        if torch.cuda.is_available():
            if config.MAX_MEMORY:
                # Parse max memory configuration, e.g. "20GiB,20GiB" or "auto"
                if config.MAX_MEMORY == "auto":
                    model_kwargs["device_map"] = "auto"
                else:
                    max_memory = {}
                    memory_parts = config.MAX_MEMORY.split(',')
                    for i, memory in enumerate(memory_parts):
                        max_memory[i] = memory.strip()
                    model_kwargs["max_memory"] = max_memory
            else:
                model_kwargs["device_map"] = config.DEVICE_MAP
        
        # 量子化設定（警告を避けるためfloat16を使用）
        if config.LOAD_IN_8BIT:
            logger.info("8bit量子化でモデルを読み込み中")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,  # 警告を避けるためfloat16を使用
                bnb_8bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
        elif config.LOAD_IN_4BIT:
            logger.info("4bit量子化でモデルを読み込み中")
            model_kwargs["load_in_4bit"] = True
            # 4bit量子化設定
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # 警告を避けるためfloat16を使用
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["quantization_config"] = quantization_config
        
        # Flash Attention configuration
        if config.USE_FLASH_ATTENTION:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # CPU offload configuration
        if os.getenv("OFFLOAD_STATE_DICT", "false").lower() == "true":
            model_kwargs["offload_state_dict"] = True
            model_kwargs["offload_folder"] = config.OFFLOAD_FOLDER
            os.makedirs(config.OFFLOAD_FOLDER, exist_ok=True)
        
        # Load model with optimized settings for 72B
        logger.info("Loading model with configuration:")
        for key, value in model_kwargs.items():
            logger.info(f"  {key}: {value}")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.MODEL_PATH,
            **model_kwargs
        )
        
        # プロセッサーを読み込み
        processor = AutoProcessor.from_pretrained(
            config.MODEL_PATH,
            trust_remote_code=True
        )
        
        # deprecation警告を修正
        try:
            import json
            preprocessor_path = os.path.join(config.MODEL_PATH, "preprocessor.json")
            video_preprocessor_path = os.path.join(config.MODEL_PATH, "video_preprocessor.json")
            
            if os.path.exists(preprocessor_path) and not os.path.exists(video_preprocessor_path):
                logger.info("video processor設定のdeprecation警告を修正中...")
                with open(preprocessor_path, 'r') as f:
                    config_data = json.load(f)
                
                # video processor設定を抽出
                if 'video' in config_data or 'video_processor' in config_data:
                    video_config = config_data.get('video', config_data.get('video_processor', {}))
                    with open(video_preprocessor_path, 'w') as f:
                        json.dump(video_config, f, indent=2)
                    logger.info("video processor設定をvideo_preprocessor.jsonに保存しました")
                else:
                    # デフォルトのvideo processor設定を作成
                    default_video_config = {
                        "processor_class": "QwenVideoProcessor",
                        "video_mean": [0.485, 0.456, 0.406],
                        "video_std": [0.229, 0.224, 0.225]
                    }
                    with open(video_preprocessor_path, 'w') as f:
                        json.dump(default_video_config, f, indent=2)
                    logger.info("デフォルトのvideo processor設定を作成しました")
        except Exception as e:
            logger.warning(f"deprecation警告の修正に失敗しました: {e}")
        
        # モデルの暖機運転
        logger.info("Performing model warmup...")
        try:
            warmup_inputs = processor(
                text=["Hello"], 
                images=None, 
                return_tensors="pt"
            )
            if torch.cuda.is_available() and not config.LOAD_IN_8BIT and not config.LOAD_IN_4BIT:
                warmup_inputs = warmup_inputs.to("cuda")
            
            with torch.no_grad():
                model.generate(**warmup_inputs, max_new_tokens=10)
        except Exception as e:
            logger.warning(f"Warmup failed, but continuing: {e}")
        
        logger.info("Model loaded successfully!")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            total_memory = 0
            logger.info("=== GPU Memory Distribution ===")
            for i in range(gpu_count):
                memory = torch.cuda.memory_allocated(i) / 1024**3
                total_memory += memory
                gpu_name = torch.cuda.get_device_name(i)
                gpu_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i} ({gpu_name}): {memory:.2f}GB / {gpu_total:.1f}GB used")
            logger.info(f"Total GPU memory usage: {total_memory:.2f} GB across {gpu_count} GPU(s)")
            
            if gpu_count > 1:
                logger.info("Multi-GPU configuration detected - model distributed automatically")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan management"""
    # Startup
    try:
        load_model()
        logger.info("API server started successfully on port 8000")
        yield
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    # Shutdown cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Create FastAPI app
app = FastAPI(
    title="Qwen2.5-VL-72B API Server",
    description="High-performance API for Qwen2.5-VL-72B Vision-Language Model",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.ALLOWED_HOSTS
)

# Data models
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: Union[str, List[Dict[str, Any]]]

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage] = Field(..., min_items=1)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=config.MAX_TOKENS_LIMIT)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    stream: Optional[bool] = False

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict

class ErrorResponse(BaseModel):
    error: Dict[str, str]

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "http_error"}}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "internal_error"}}
    )

# Rate limiting
async def rate_limit_check(request: Request):
    client_ip = request.client.host
    current_time = time.time()
    minute = int(current_time // 60)
    
    key = f"{client_ip}:{minute}"
    request_counts[key] = request_counts.get(key, 0) + 1
    
    if request_counts[key] > config.RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Clean up old counts
    old_keys = [k for k in request_counts.keys() if int(k.split(':')[1]) < minute - 5]
    for old_key in old_keys:
        del request_counts[old_key]

def verify_api_key(authorization: str = Header(None)):
    """Verify API key from Authorization header"""
    if authorization is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    
    if token != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return token

@lru_cache(maxsize=100)
def validate_image_format(image_data: bytes) -> bool:
    """Validate image format"""
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            return img.format in ['JPEG', 'PNG', 'WEBP']
    except:
        return False

def process_content(content) -> List[Dict]:
    """メッセージ内容を処理してテキストと画像を扱う"""
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
                    # base64画像を解析
                    header, image_data = image_url.split(",", 1)
                    image_bytes = base64.b64decode(image_data)
                    
                    # 画像サイズをチェック
                    if len(image_bytes) > config.MAX_IMAGE_SIZE:
                        raise HTTPException(status_code=400, detail="画像サイズが制限を超えています")
                    
                    # 画像フォーマットを検証
                    if not validate_image_format(image_bytes):
                        raise HTTPException(status_code=400, detail="サポートされていない画像フォーマットです")
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    processed.append({"type": "image", "image": image})
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"画像処理エラー: {str(e)}")
            else:
                raise HTTPException(status_code=400, detail="base64画像フォーマットのみサポートされています")
    
    return processed

async def generate_response(messages: List[Dict], max_tokens: int = 2048, 
                          temperature: float = 0.0, top_p: float = 0.9) -> str:
    """統合VLモデルを使用してモデル応答を生成"""
    global model, processor
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="モデルが読み込まれていません")
    
    # 推論前のメモリ状態をログ記録
    pre_memory = get_memory_info()
    logger.debug(f"推論開始 - メモリ使用量: {pre_memory['allocated_gb']}GB/{pre_memory['reserved_gb']}GB")
    
    async with model_lock:  # 同時推論の問題を防ぐ
        try:
            # マルチモーダル入力用のメッセージを処理
            processed_messages = []
            for msg in messages:
                processed_messages.append({
                    "role": msg["role"],
                    "content": process_content(msg["content"])
                })
            
            # Apply chat template
            text = processor.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process vision information (images/videos)
            image_inputs, video_inputs = process_vision_info(processed_messages)
            
            # Prepare inputs for the unified VL model
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            # 生成パラメータを準備（警告を避けるため）
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "pad_token_id": processor.tokenizer.eos_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            
            # temperatureとsampling設定（サポートされている場合のみ）
            if temperature > 0:
                generation_kwargs.update({
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                })
            else:
                generation_kwargs["do_sample"] = False
            
            # 同一モデルを使用してすべてのモダリティの応答を生成
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **generation_kwargs)
            
            # Extract newly generated tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # レスポンスをデコード
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
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
                cleanup_gpu_memory(aggressive=config.AGGRESSIVE_CLEANUP)
                last_cleanup_count = inference_count
                logger.info(f"メモリクリーンアップ実行 (推理回数: {inference_count})")
            elif config.FORCE_SYNC:
                torch.cuda.synchronize()
                logger.debug(f"軽量クリーンアップ実行 (推理回数: {inference_count})")
            
            # 推論後のメモリ状態をログ記録
            post_memory = get_memory_info()
            memory_delta = pre_memory['allocated_gb'] - post_memory['allocated_gb']
            logger.debug(f"推論完了 - メモリ変化: {memory_delta:+.2f}GB, 現在使用量: {post_memory['allocated_gb']}GB")
            
            return output_text.strip()
            
        except torch.cuda.OutOfMemoryError:
            # OOMエラー時は積極的クリーンアップを実行
            cleanup_gpu_memory(aggressive=True)
            raise HTTPException(status_code=503, detail="GPU out of memory, please try again later")
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Qwen2.5-VL-72B API Server",
        "status": "running",
        "model": config.MODEL_NAME,
        "capabilities": ["chat", "vision", "multimodal"],
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": config.MODEL_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "qwen",
            "capabilities": ["chat", "vision", "multimodal"]
        }]
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    client_request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Unified chat completion endpoint for the Vision-Language model.
    Handles text-only, image-only, and multimodal conversations seamlessly.
    """
    # Rate limiting check
    await rate_limit_check(client_request)
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request messages format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Generate response using the unified VL model
        response_text = await generate_response(
            messages,
            request.max_tokens,
            request.temperature,
            request.top_p
        )
        
        # Build OpenAI-compatible response
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        created_time = int(time.time())
        
        # Simple token counting (could be improved with actual tokenizer)
        prompt_tokens = sum(len(str(msg["content"]).split()) for msg in messages)
        completion_tokens = len(response_text.split())
        total_tokens = prompt_tokens + completion_tokens
        
        return ChatResponse(
            id=completion_id,
            created=created_time,
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Chat completion failed")

@app.get("/health")
async def health():
    """ヘルスチェックエンドポイント"""
    gpu_info = {}
    if torch.cuda.is_available():
        memory_info = get_memory_info()
        gpu_info = {
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_allocated": f"{memory_info['allocated_gb']} GB",
            "gpu_memory_reserved": f"{memory_info['reserved_gb']} GB",
            "gpu_memory_free": f"{memory_info['free_gb']} GB",
            "gpu_utilization": f"{torch.cuda.utilization()}%" if hasattr(torch.cuda, 'utilization') else "N/A"
        }
    else:
        gpu_info = {"gpu_available": False}
    
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "model_name": config.MODEL_NAME,
        "model_type": "vision_language_model",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time(),
        "inference_count": inference_count,
        "memory_management": {
            "auto_cleanup": config.AUTO_CLEANUP_MEMORY,
            "cleanup_interval": config.CLEANUP_INTERVAL,
            "aggressive_cleanup": config.AGGRESSIVE_CLEANUP,
            "last_cleanup_at": last_cleanup_count
        },
        **gpu_info
    }

@app.get("/metrics")
async def metrics(api_key: str = Depends(verify_api_key)):
    """監視メトリクスエンドポイント"""
    memory_info = get_memory_info() if torch.cuda.is_available() else {}
    
    return {
        "active_requests": len(request_counts),
        "model_loaded": model is not None,
        "model_type": "unified_vision_language",
        "gpu_available": torch.cuda.is_available(),
        "inference_stats": {
            "total_inferences": inference_count,
            "cleanups_performed": last_cleanup_count,
            "pending_cleanups": max(0, inference_count - last_cleanup_count)
        },
        "memory_usage": {
            "gpu_allocated_gb": memory_info.get("allocated_gb", 0),
            "gpu_reserved_gb": memory_info.get("reserved_gb", 0),
            "gpu_free_gb": memory_info.get("free_gb", 0)
        },
        "memory_config": {
            "auto_cleanup_enabled": config.AUTO_CLEANUP_MEMORY,
            "cleanup_interval": config.CLEANUP_INTERVAL,
            "aggressive_cleanup": config.AGGRESSIVE_CLEANUP
        },
        "capabilities": ["text", "image", "multimodal_chat"]
    }

@app.post("/v1/memory/cleanup")
async def manual_memory_cleanup(api_key: str = Depends(verify_api_key)):
    """手動でGPUメモリクリーンアップを実行"""
    try:
        memory_before = get_memory_info()
        cleanup_gpu_memory(aggressive=True)
        memory_after = get_memory_info()
        
        global last_cleanup_count
        last_cleanup_count = inference_count
        
        return {
            "status": "success",
            "message": "手動メモリクリーンアップが完了しました",
            "memory_before": memory_before,
            "memory_after": memory_after,
            "freed_gb": round(memory_before.get("allocated_gb", 0) - memory_after.get("allocated_gb", 0), 2)
        }
    except Exception as e:
        logger.error(f"Manual cleanup failed: {e}")
        return {
            "status": "error",
            "message": f"メモリクリーンアップに失敗しました: {str(e)}"
        }

if __name__ == "__main__":
    print("Starting Qwen2.5-VL-72B API Server...")
    print(f"API Key: {config.API_KEY}")
    print(f"Request Timeout: {config.REQUEST_TIMEOUT}秒")
    print("Visit http://localhost:8000/docs for API documentation")
    print("This unified endpoint handles both text and vision seamlessly!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=True,
        log_level="info",
        timeout_keep_alive=config.REQUEST_TIMEOUT,
        timeout_graceful_shutdown=30
    )