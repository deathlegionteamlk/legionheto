import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import os
import json
from tqdm import tqdm


class BaseTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_grad_norm: float = 0.3,
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",
        seed: int = 42,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        
        self.seed = seed
        self.global_step = 0
        self.epoch = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        torch.manual_seed(seed)
        
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator if hasattr(self, 'data_collator') else None,
        )
    
    def get_eval_dataloader(self):
        if self.eval_dataset is None:
            return None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator if hasattr(self, 'data_collator') else None,
        )
    
    def compute_loss(self, model, inputs):
        raise NotImplementedError
    
    def training_step(self, model, inputs):
        model.train()
        loss = self.compute_loss(model, inputs)
        
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        loss.backward()
        
        return loss.item()
    
    def evaluate(self):
        if self.eval_dataset is None:
            return {}
        
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader()
        total_loss = 0
        
        with torch.no_grad():
            for inputs in eval_dataloader:
                loss = self.compute_loss(self.model, inputs)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(eval_dataloader)
        return {"eval_loss": avg_loss}
    
    def save_model(self, output_dir: Optional[str] = None):
        save_dir = output_dir or self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        config_path = os.path.join(save_dir, "trainer_config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "num_train_epochs": self.num_train_epochs,
                "learning_rate": self.learning_rate,
                "global_step": self.global_step,
            }, f, indent=2)
    
    def train(self):
        raise NotImplementedError
