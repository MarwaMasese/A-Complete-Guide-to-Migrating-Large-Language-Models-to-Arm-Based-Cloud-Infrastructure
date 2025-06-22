import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psutil
import time

from model_loader import ModelLoader
from inference import InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
model_loader: Optional[ModelLoader] = None
inference_engine: Optional[InferenceEngine] = None

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for generation")
    max_length: int = Field(default=512, ge=1, le=2048, description="Maximum length of generated text")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p (nucleus) sampling")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.1, ge=0.5, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")

class GenerateResponse(BaseModel):
    response: str
    prompt: str
    generation_time: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    system_info: Dict[str, Any]
    model_info: Optional[Dict[str, Any]] = None

class BatchGenerateRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of prompts for batch generation")
    max_length: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting LLM Inference API...")
    await load_model_on_startup()
    yield
    # Shutdown
    logger.info("Shutting down LLM Inference API...")

app = FastAPI(
    title="LLM ARM Inference API",
    description="Production-grade LLM inference API optimized for ARM-based cloud infrastructure",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def load_model_on_startup():
    """Load model during application startup"""
    global model_loader, inference_engine
    
    try:
        model_name = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
        logger.info(f"Loading model: {model_name}")
        
        model_loader = ModelLoader(model_name)
        
        # Try to load llama.cpp model if GGML_MODEL_PATH is provided
        ggml_path = os.getenv("GGML_MODEL_PATH")
        if ggml_path and os.path.exists(ggml_path):
            logger.info(f"Loading GGML model from: {ggml_path}")
            success = model_loader.load_llama_cpp_model(ggml_path)
        else:
            success = model_loader.load_model()
        
        if success:
            inference_engine = InferenceEngine(model_loader)
            logger.info("Model loaded successfully")
        else:
            logger.error("Failed to load model")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform
    
    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "cpu_usage_percent": psutil.cpu_percent(interval=1)
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    system_info = get_system_info()
    model_loaded = inference_engine is not None
    
    response = HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        system_info=system_info,
        model_info=model_loader.get_model_info() if model_loader else None
    )
    
    return response

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text from a prompt"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = inference_engine.generate_response(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            stop_sequences=request.stop_sequences
        )
        
        return GenerateResponse(**result)
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/batch_generate")
async def batch_generate_text(request: BatchGenerateRequest):
    """Generate text for multiple prompts"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.prompts) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 prompts per batch")
    
    try:
        results = inference_engine.batch_generate(
            prompts=request.prompts,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_loader.get_model_info()

@app.get("/model/stats")
async def model_stats():
    """Get model statistics"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return inference_engine.get_model_stats()

@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (useful for updates)"""
    background_tasks.add_task(load_model_on_startup)
    return {"message": "Model reload initiated"}

@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics endpoint"""
    system_info = get_system_info()
    
    metrics_text = f"""# HELP llm_memory_usage_bytes Memory usage in bytes
# TYPE llm_memory_usage_bytes gauge
llm_memory_usage_bytes {psutil.virtual_memory().used}

# HELP llm_cpu_usage_percent CPU usage percentage
# TYPE llm_cpu_usage_percent gauge
llm_cpu_usage_percent {system_info['cpu_usage_percent']}

# HELP llm_model_loaded Model loaded status
# TYPE llm_model_loaded gauge
llm_model_loaded {1 if inference_engine else 0}
"""
    
    return JSONResponse(content=metrics_text, media_type="text/plain")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LLM ARM Inference API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "architecture": get_system_info()["architecture"]
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    workers = int(os.getenv("WORKERS", 1))
    
    logger.info(f"Starting server on {host}:{port} with {workers} workers")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        access_log=True
    )
