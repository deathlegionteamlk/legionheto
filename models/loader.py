import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import logging

from .registry import ModelRegistry
from ..utils.checkpointing import GradientCheckpointManager as MemoryOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_name: str, vram_gb: Optional[float] = None):
        self.model_name = model_name
        self.memory_optimizer = MemoryOptimizer(model_name, vram_gb)
    
    def load(
        self,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        max_seq_length: int = 2048,
        **kwargs
    ) -> tuple:
        arch_config = ModelRegistry.get_optimal_config(
            self.model_name,
            self.memory_optimizer.vram_gb
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        from transformers import BitsAndBytesConfig
        
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **kwargs
        )
        
        return model, tokenizer, arch_config
    
    def get_recommended_config(self) -> Dict[str, Any]:
        return self.memory_optimizer.get_memory_report(7)
