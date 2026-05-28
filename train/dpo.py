import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, Any
from tqdm import tqdm
import os
from .base import BaseTrainer


class DPOTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-6,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100,
        max_grad_norm: float = 0.3,
        weight_decay: float = 0.01,
        lr_scheduler_type: str = "cosine",
        seed: int = 42,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
        )
        self.ref_model = ref_model
        self.beta = beta
        self.label_smoothing = label_smoothing
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def _get_batch_logps(self, logits, labels):
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)
        
        loss_mask = (labels != -100).float()
        
        return (per_token_logps * loss_mask).sum(-1)
    
    def compute_loss(self, model, inputs):
        chosen_input_ids = inputs["chosen_input_ids"]
        chosen_attention_mask = inputs["chosen_attention_mask"]
        rejected_input_ids = inputs["rejected_input_ids"]
        rejected_attention_mask = inputs["rejected_attention_mask"]
        
        chosen_outputs = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
        rejected_outputs = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
        
        with torch.no_grad():
            ref_chosen_outputs = self.ref_model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
            ref_rejected_outputs = self.ref_model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
        
        policy_chosen_logps = self._get_batch_logps(chosen_outputs.logits, chosen_input_ids)
        policy_rejected_logps = self._get_batch_logps(rejected_outputs.logits, rejected_input_ids)
        reference_chosen_logps = self._get_batch_logps(ref_chosen_outputs.logits, chosen_input_ids)
        reference_rejected_logps = self._get_batch_logps(ref_rejected_outputs.logits, rejected_input_ids)
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios
        losses = -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        
        return losses.mean()
    
    def get_lr_scheduler(self, optimizer, num_training_steps):
        if self.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(optimizer, T_max=num_training_steps)
        elif self.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
        else:
            return None
    
    def train(self):
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.gradient_accumulation_steps
        num_update_steps = num_update_steps_per_epoch * self.num_train_epochs
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = self.get_lr_scheduler(optimizer, num_update_steps)
        
        total_loss = 0
        self.global_step = 0
        
        for epoch in range(self.num_train_epochs):
            self.epoch = epoch
            self.model.train()
            self.ref_model.eval()
            
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_train_epochs}")
            
            for step, inputs in enumerate(epoch_iterator):
                inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
                
                loss = self.training_step(self.model, inputs)
                total_loss += loss
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
                    
                    self.global_step += 1
                    
                    if self.global_step % self.logging_steps == 0:
                        avg_loss = total_loss / self.logging_steps
                        epoch_iterator.set_postfix({"loss": f"{avg_loss:.4f}"})
                        total_loss = 0
                    
                    if self.global_step % self.save_steps == 0:
                        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.global_step}")
                        self.save_model(checkpoint_dir)
        
        self.save_model()
        return {"train_loss": total_loss}
