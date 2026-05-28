import torch
from typing import Dict, Any


def get_gpu_memory() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"available": 0.0, "total": 0.0, "used": 0.0}
    
    return {
        "available": torch.cuda.mem_get_info()[0] / (1024**3),
        "total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "used": torch.cuda.memory_allocated() / (1024**3),
    }


def print_model_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.4f}%")
