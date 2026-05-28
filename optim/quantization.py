import torch
from typing import Tuple


def quantize_blockwise(
    tensor: torch.Tensor,
    block_size: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = tensor.numel()
    num_blocks = (numel + block_size - 1) // block_size
    
    flat = tensor.view(-1)
    
    quantized_blocks = []
    block_mins = []
    block_maxs = []
    
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, numel)
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


def dequantize_blockwise(
    quantized: torch.Tensor,
    norm: torch.Tensor,
    block_mins: torch.Tensor,
    block_maxs: torch.Tensor,
    shape: torch.Size,
    block_size: int = 2048,
) -> torch.Tensor:
    numel = shape.numel()
    num_blocks = (numel + block_size - 1) // block_size
    
    dequantized_blocks = []
    
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, numel)
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
