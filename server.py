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

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration class
class Config:
    API_KEY = os.getenv("QWEN_API_KEY", "sk-qwen25-vl-72b--demo-key")
    MODEL_NAME = "qwen/qwen2.5-vl-72b-instruct"
    MODEL_PATH = os.getenv("MODEL_PATH", "/workspace/Qwen2.5-VL-72B-Instruct")
    MAX_TOKENS_LIMIT = 8192  # Increase token limit
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # Increase to 20MB
    RATE_LIMIT_REQUESTS = 50  # Reduce concurrency limit
    ALLOWED_HOSTS = ["*"]  # Should be restricted in production
    
    # GPU configuration for 72B model
    USE_FLASH_ATTENTION = True
    TORCH_DTYPE = torch.bfloat16
    DEVICE_MAP = "auto"  # Automatic multi-GPU allocation
    LOAD_IN_8BIT = os.getenv("LOAD_IN_8BIT", "false").lower() == "true"
    LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "false").lower() == "true"
    MAX_MEMORY = os.getenv("MAX_MEMORY", None)  # Can set max memory per GPU
    OFFLOAD_FOLDER = os.getenv("OFFLOAD_FOLDER", "/tmp/offload")  # CPU offload directory

config = Config()

# Global variables
model = None
processor = None
model_lock = asyncio.Lock()

# Request counter for simple rate limiting
request_counts = {}

def load_model():
    """Load Qwen2.5-VL model and processor"""
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
        
        # Quantization configuration
        if config.LOAD_IN_8BIT:
            logger.info("Loading model with 8-bit quantization")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=config.TORCH_DTYPE,
                bnb_8bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = quantization_config
        elif config.LOAD_IN_4BIT:
            logger.info("Loading model with 4-bit quantization")
            model_kwargs["load_in_4bit"] = True
            # 4bit quantization configuration
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=config.TORCH_DTYPE,
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
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            config.MODEL_PATH,
            trust_remote_code=True
        )
        
        # Model warmup
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
            total_memory = 0
            for i in range(torch.cuda.device_count()):
                memory = torch.cuda.memory_allocated(i) / 1024**3
                total_memory += memory
                logger.info(f"GPU {i} memory usage: {memory:.2f} GB")
            logger.info(f"Total GPU memory usage: {total_memory:.2f} GB")
            
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
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
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
                    
                    # Check image size
                    if len(image_bytes) > config.MAX_IMAGE_SIZE:
                        raise HTTPException(status_code=400, detail="Image size exceeds limit")
                    
                    # Validate image format
                    if not validate_image_format(image_bytes):
                        raise HTTPException(status_code=400, detail="Unsupported image format")
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    processed.append({"type": "image", "image": image})
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Image processing error: {str(e)}")
            else:
                raise HTTPException(status_code=400, detail="Only base64 image format is supported")
    
    return processed

async def generate_response(messages: List[Dict], max_tokens: int = 2048, 
                          temperature: float = 0.7, top_p: float = 0.9) -> str:
    """Generate model response using the unified VL model"""
    global model, processor
    
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async with model_lock:  # Prevent concurrent inference issues
        try:
            # Process messages for multimodal input
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
            
            # Generate response using the same model for all modalities
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            
            # Extract newly generated tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode response
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
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
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_available": True,
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
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
        **gpu_info
    }

@app.get("/metrics")
async def metrics(api_key: str = Depends(verify_api_key)):
    """Monitoring metrics endpoint"""
    return {
        "active_requests": len(request_counts),
        "model_loaded": model is not None,
        "model_type": "unified_vision_language",
        "gpu_available": torch.cuda.is_available(),
        "memory_usage": {
            "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        },
        "capabilities": ["text", "image", "multimodal_chat"]
    }

if __name__ == "__main__":
    print("Starting Qwen2.5-VL-72B API Server...")
    print(f"API Key: {config.API_KEY}")
    print("Visit http://localhost:8000/docs for API documentation")
    print("This unified endpoint handles both text and vision seamlessly!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=True,
        log_level="info"
    )