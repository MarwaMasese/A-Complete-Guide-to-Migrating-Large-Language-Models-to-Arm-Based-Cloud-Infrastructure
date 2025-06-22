import torch
import logging
from typing import Dict, List, Optional
from transformers import StoppingCriteria, StoppingCriteriaList
import time

logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class InferenceEngine:
    def __init__(self, model_loader):
        self.model_loader = model_loader
        self.model = model_loader.model
        self.tokenizer = model_loader.tokenizer
        self.device = model_loader.device
        self.use_llama_cpp = model_loader.use_llama_cpp
        
    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict:
        """Generate response using the loaded model"""
        
        if self.model is None:
            raise ValueError("Model not loaded")
        
        start_time = time.time()
        
        try:
            if self.use_llama_cpp:
                response = self._generate_llama_cpp(
                    prompt, max_length, temperature, top_p, top_k
                )
            else:
                response = self._generate_transformers(
                    prompt, max_length, temperature, top_p, top_k,
                    repetition_penalty, do_sample, num_return_sequences, stop_sequences
                )
            
            generation_time = time.time() - start_time
            
            return {
                "response": response,
                "prompt": prompt,
                "generation_time": round(generation_time, 3),
                "model_info": self.model_loader.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _generate_llama_cpp(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> str:
        """Generate response using llama.cpp"""
        
        output = self.model(
            prompt,
            max_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            echo=False,
            stop=["</s>", "\n\n"]
        )
        
        return output['choices'][0]['text'].strip()
    
    def _generate_transformers(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        do_sample: bool,
        num_return_sequences: int,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Generate response using Transformers"""
        
        # Encode input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Prepare stopping criteria
        stopping_criteria = None
        if stop_sequences:
            stop_token_ids = []
            for seq in stop_sequences:
                tokens = self.tokenizer.encode(seq, add_special_tokens=False)
                stop_token_ids.extend(tokens)
            if stop_token_ids:
                stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])
        
        # Generate
        with torch.no_grad():
            generation_kwargs = {
                "input_ids": inputs,
                "max_length": min(len(inputs[0]) + max_length, 2048),
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            if stopping_criteria:
                generation_kwargs["stopping_criteria"] = stopping_criteria
            
            outputs = self.model.generate(**generation_kwargs)
        
        # Decode response
        input_length = len(inputs[0])
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[Dict]:
        """Generate responses for multiple prompts"""
        
        if self.use_llama_cpp:
            # llama.cpp doesn't support batch generation easily
            return [self.generate_response(prompt, max_length, temperature, top_p, **kwargs) 
                   for prompt in prompts]
        
        # For transformers, we can implement basic batching
        results = []
        for prompt in prompts:
            result = self.generate_response(prompt, max_length, temperature, top_p, **kwargs)
            results.append(result)
        
        return results
    
    def get_model_stats(self) -> Dict:
        """Get model statistics and performance metrics"""
        stats = {
            "model_info": self.model_loader.get_model_info(),
            "device": self.device,
            "backend": "llama.cpp" if self.use_llama_cpp else "transformers"
        }
        
        if hasattr(self.model, 'get_memory_footprint'):
            try:
                stats["memory_footprint"] = self.model.get_memory_footprint()
            except:
                pass
        
        return stats
