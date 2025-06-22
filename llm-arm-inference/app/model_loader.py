import os
import platform
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, Dict, Any
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        self.is_arm = self._is_arm_architecture()
        self.use_llama_cpp = False
        
    def _is_arm_architecture(self) -> bool:
        """Detect if running on ARM architecture"""
        machine = platform.machine().lower()
        return any(arch in machine for arch in ['arm', 'aarch64'])
    
    def _get_device(self) -> str:
        """Get the appropriate device for inference"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _install_llama_cpp(self) -> bool:
        """Install appropriate llama-cpp-python version based on architecture"""
        try:
            if self.is_arm:
                # For ARM, install with NEON support
                logger.info("Installing llama-cpp-python with ARM NEON support...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "llama-cpp-python==0.2.20",
                    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"
                ])
            else:
                # For x86_64, install standard version
                logger.info("Installing llama-cpp-python for x86_64...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "llama-cpp-python==0.2.20"
                ])
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install llama-cpp-python: {e}")
            return False
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization config - only use 4-bit on ARM for better performance"""
        if not self.is_arm:
            return None
            
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model(self) -> bool:
        """Load the model with appropriate optimizations"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Architecture: {'ARM' if self.is_arm else 'x86_64'}")
            logger.info(f"Device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Get quantization config for ARM
            quantization_config = self._get_quantization_config()
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                "device_map": "auto" if self.device != "cpu" else None,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                logger.info("Using 4-bit quantization for ARM optimization")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_llama_cpp_model(self, model_path: str) -> bool:
        """Load model using llama.cpp for ARM optimization"""
        try:
            # Install llama-cpp-python if not already installed
            if not self._install_llama_cpp():
                return False
            
            from llama_cpp import Llama
            
            # Configure for ARM NEON if available
            n_threads = os.cpu_count() if self.is_arm else os.cpu_count() // 2
            
            self.model = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=n_threads,
                n_gpu_layers=0,  # Use CPU for ARM optimization
                use_mlock=True,
                verbose=False
            )
            
            self.use_llama_cpp = True
            logger.info(f"Llama.cpp model loaded with {n_threads} threads")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load llama.cpp model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "architecture": "ARM" if self.is_arm else "x86_64",
            "device": self.device,
            "quantized": self.is_arm and not self.use_llama_cpp,
            "llama_cpp": self.use_llama_cpp,
            "loaded": self.model is not None
        }
