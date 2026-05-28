import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, List, Dict, Any
import logging

from .registry import ModelRegistry
from ..attention.flash_attn import has_flash_attn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegionHetoModel:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        trust_remote_code: bool = False,
        use_flash_attn: bool = True,
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self.use_flash_attn = use_flash_attn and has_flash_attn()
        
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.is_peft_model = False
        self.arch_config = None
        
        self._load_model()
    
    def _load_model(self):
        logger.info(f"Loading model: {self.model_name}")
        
        self.arch_config = ModelRegistry.get_optimal_config(self.model_name)
        
        if self.dtype is None:
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        quantization_config = None
        if self.load_in_4bit:
            logger.info("Loading model in 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.load_in_8bit:
            logger.info("Loading model in 8-bit quantization...")
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map=self.device_map,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            attn_implementation="flash_attention_2" if self.use_flash_attn else None,
        )
        
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded successfully!")
    
    def setup_lora(
        self,
        r: Optional[int] = None,
        alpha: Optional[int] = None,
        dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        use_rslora: bool = False,
    ) -> "LegionHetoModel":
        if self.model is None:
            raise RuntimeError("Model not loaded. Call __init__ first.")
        
        if target_modules is None and self.arch_config:
            target_modules = self.arch_config.get("target_modules")
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        
        r = r or (self.arch_config.get("lora_r") if self.arch_config else 16)
        alpha = alpha or (self.arch_config.get("lora_alpha") if self.arch_config else 32)
        
        logger.info(f"Setting up LoRA with r={r}, alpha={alpha}, target_modules={target_modules}")
        
        if self.load_in_4bit or self.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        self.peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias=bias,
            task_type="CAUSAL_LM",
            use_rslora=use_rslora,
        )
        
        self.model = get_peft_model(self.model, self.peft_config)
        self.is_peft_model = True
        
        logger.info("LoRA setup complete!")
        self.print_trainable_parameters()
        
        return self
    
    def get_trainable_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def print_trainable_parameters(self):
        trainable = self.get_trainable_parameters()
        total = self.get_total_parameters()
        pct = 100 * trainable / total if total > 0 else 0
        logger.info(f"Trainable params: {trainable:,} || Total params: {total:,} || Trainable%: {pct:.4f}%")
    
    def save_adapter(self, output_dir: str):
        if not self.is_peft_model:
            raise RuntimeError("Model is not a PEFT model. Call setup_lora() first.")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Adapter saved to {output_dir}")
    
    def load_adapter(self, adapter_path: str):
        from peft import PeftModel
        
        if self.model is None:
            raise RuntimeError("Base model not loaded.")
        
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.is_peft_model = True
        logger.info(f"Adapter loaded from {adapter_path}")
    
    def merge_and_unload(self):
        if not self.is_peft_model:
            logger.warning("Model is not a PEFT model, nothing to merge.")
            return self.model
        
        logger.info("Merging adapter weights with base model...")
        self.model = self.model.merge_and_unload()
        self.is_peft_model = False
        self.peft_config = None
        logger.info("Merge complete!")
        
        return self.model
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model or tokenizer not loaded.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def __repr__(self):
        return f"LegionHetoModel({self.model_name}, peft={self.is_peft_model})"
