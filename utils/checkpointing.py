import torch
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class GradientCheckpointManager:
    def __init__(self, model):
        self.model = model
        self.enabled = False
        self.compression_ratio = 0.5
    
    def enable(self, selective: bool = True, layers_to_checkpoint: Optional[List[int]] = None):
        if not hasattr(self.model, "gradient_checkpointing_enable"):
            logger.warning("Model does not support gradient checkpointing")
            return
        
        if selective and layers_to_checkpoint:
            self._enable_selective(layers_to_checkpoint)
        else:
            self.model.gradient_checkpointing_enable()
        
        self.enabled = True
        logger.info("Gradient checkpointing enabled")
    
    def _enable_selective(self, layers_to_checkpoint: List[int]):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for i, layer in enumerate(self.model.model.layers):
                if i in layers_to_checkpoint:
                    layer.gradient_checkpointing = True
    
    def disable(self):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        self.enabled = False
        logger.info("Gradient checkpointing disabled")
