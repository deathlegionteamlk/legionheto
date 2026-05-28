import torch
from torch.optim import Optimizer
from typing import Optional, Tuple, List
import math


class AdamW8bit(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        block_size: int = 2048,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size,
        )
        super().__init__(params, defaults)
        
        self.block_size = block_size
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["step"] = 0
                    
                    numel = p.numel()
                    num_blocks = (numel + block_size - 1) // block_size
                    
                    state["exp_avg_q"] = torch.zeros(num_blocks, dtype=torch.uint8, device=p.device)
                    state["exp_avg_norm"] = torch.zeros(num_blocks, dtype=torch.float32, device=p.device)
                    state["exp_avg_block_min"] = torch.zeros(num_blocks, dtype=torch.float32, device=p.device)
                    state["exp_avg_block_max"] = torch.zeros(num_blocks, dtype=torch.float32, device=p.device)
                    
                    state["exp_avg_sq_q"] = torch.zeros(num_blocks, dtype=torch.uint8, device=p.device)
                    state["exp_avg_sq_norm"] = torch.zeros(num_blocks, dtype=torch.float32, device=p.device)
                    state["exp_avg_sq_block_min"] = torch.zeros(num_blocks, dtype=torch.float32, device=p.device)
                    state["exp_avg_sq_block_max"] = torch.zeros(num_blocks, dtype=torch.float32, device=p.device)
    
    def _quantize_blockwise(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        numel = tensor.numel()
        num_blocks = (numel + self.block_size - 1) // self.block_size
        
        flat = tensor.view(-1)
        
        quantized_blocks = []
        block_mins = []
        block_maxs = []
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, numel)
            block = flat[start:end]
            
            block_min = block.min()
            block_max = block.max()
            
            if block_max - block_min > 1e-8:
                scaled = (block - block_min) / (block_max - block_min)
                quantized = (scaled * 255).to(torch.uint8)
            else:
                quantized = torch.zeros_like(block, dtype=torch.uint8)
            
            quantized_blocks.append(quantized)
            block_mins.append(block_min)
            block_maxs.append(block_max)
        
        quantized = torch.cat(quantized_blocks)
        block_mins = torch.stack(block_mins)
        block_maxs = torch.stack(block_maxs)
        
        norm = torch.norm(tensor)
        
        return quantized, norm, block_mins, block_maxs
    
    def _dequantize_blockwise(
        self,
        quantized: torch.Tensor,
        norm: torch.Tensor,
        block_mins: torch.Tensor,
        block_maxs: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        numel = shape.numel()
        num_blocks = (numel + self.block_size - 1) // self.block_size
        
        dequantized_blocks = []
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, numel)
            block = quantized[start:end].float()
            
            block_min = block_mins[i]
            block_max = block_maxs[i]
            
            dequantized = block / 255.0 * (block_max - block_min) + block_min
            dequantized_blocks.append(dequantized)
        
        dequantized = torch.cat(dequantized_blocks).view(shape)
        
        current_norm = torch.norm(dequantized)
        if current_norm > 1e-8:
            dequantized = dequantized * (norm / current_norm)
        
        return dequantized
    
    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW8bit does not support sparse gradients")
                
                state = self.state[p]
                state["step"] += 1
                step = state["step"]
                
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                
                exp_avg_q = state["exp_avg_q"]
                exp_avg_norm = state["exp_avg_norm"]
                exp_avg_block_min = state["exp_avg_block_min"]
                exp_avg_block_max = state["exp_avg_block_max"]
                
                exp_avg_sq_q = state["exp_avg_sq_q"]
                exp_avg_sq_norm = state["exp_avg_sq_norm"]
                exp_avg_sq_block_min = state["exp_avg_sq_block_min"]
                exp_avg_sq_block_max = state["exp_avg_sq_block_max"]
                
                if step == 1:
                    exp_avg_q_new, exp_avg_norm_new, exp_avg_min_new, exp_avg_max_new = self._quantize_blockwise(grad)
                    exp_avg_sq_q_new, exp_avg_sq_norm_new, exp_avg_sq_min_new, exp_avg_sq_max_new = self._quantize_blockwise(grad * grad)
                else:
                    exp_avg = self._dequantize_blockwise(
                        exp_avg_q, exp_avg_norm, exp_avg_block_min, exp_avg_block_max, grad.shape
                    )
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_q_new, exp_avg_norm_new, exp_avg_min_new, exp_avg_max_new = self._quantize_blockwise(exp_avg)
                    
                    exp_avg_sq = self._dequantize_blockwise(
                        exp_avg_sq_q, exp_avg_sq_norm, exp_avg_sq_block_min, exp_avg_sq_block_max, grad.shape
                    )
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    exp_avg_sq_q_new, exp_avg_sq_norm_new, exp_avg_sq_min_new, exp_avg_sq_max_new = self._quantize_blockwise(exp_avg_sq)
                
                state["exp_avg_q"] = exp_avg_q_new
                state["exp_avg_norm"] = exp_avg_norm_new
                state["exp_avg_block_min"] = exp_avg_min_new
                state["exp_avg_block_max"] = exp_avg_max_new
                
                state["exp_avg_sq_q"] = exp_avg_sq_q_new
                state["exp_avg_sq_norm"] = exp_avg_sq_norm_new
                state["exp_avg_sq_block_min"] = exp_avg_sq_min_new
                state["exp_avg_sq_block_max"] = exp_avg_sq_max_new
                
                exp_avg = self._dequantize_blockwise(
                    exp_avg_q_new, exp_avg_norm_new, exp_avg_min_new, exp_avg_max_new, grad.shape
                )
                exp_avg_sq = self._dequantize_blockwise(
                    exp_avg_sq_q_new, exp_avg_sq_norm_new, exp_avg_sq_min_new, exp_avg_sq_max_new, grad.shape
                )
                
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                step_size = lr / bias_correction1
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
