import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, Any
from tqdm import tqdm
import os
from .base import BaseTrainer


class RLOOTrainer(BaseTrainer):
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
        num_samples: int = 4,
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
        self.num_samples = num_samples

        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _compute_rewards(self, prompts, completions, device):
        with torch.no_grad():
            full_texts = [p + c for p, c in zip(prompts, completions)]
            inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).to(device)
            
            outputs = self.ref_model(**inputs)
            logits = outputs.logits[:, :-1, :]
            labels = inputs.input_ids[:, 1:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            mask = (labels != self.tokenizer.pad_token_id).float()
            sequence_log_probs = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)
            
        return sequence_log_probs

    def compute_loss(self, model, inputs):
        prompts = inputs["prompt"]
        chosen = inputs["chosen"]
        rejected = inputs["rejected"]
        
        device = next(model.parameters()).device
        
        chosen_rewards = self._compute_rewards(prompts, chosen, device)
        rejected_rewards = self._compute_rewards(prompts, rejected, device)
        
        chosen_full = [p + c for p, c in zip(prompts, chosen)]
        chosen_inputs = self.tokenizer(
            chosen_full,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(device)
        
        outputs = model(**chosen_inputs)
        logits = outputs.logits[:, :-1, :]
        labels = chosen_inputs.input_ids[:, 1:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        mask = (labels != self.tokenizer.pad_token_id).float()
        policy_log_probs = (token_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1)
        
        advantages = chosen_rewards - rejected_rewards
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        loss = -(policy_log_probs * advantages).mean()
        
        return loss + self.beta * (policy_log_probs ** 2).mean()

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
