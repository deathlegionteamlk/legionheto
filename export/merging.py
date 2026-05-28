import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
import copy


class ModelMerger:
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def merge_slerp(self, t: float = 0.5) -> nn.Module:
        return slerp_merge(self.models[0], self.models[1], t)
    
    def merge_ties(
        self,
        density: float = 0.6,
        weight_mask_rate: float = 0.2,
    ) -> nn.Module:
        return ties_merge(self.models, self.weights, density, weight_mask_rate)
    
    def merge_dare(
        self,
        drop_rate: float = 0.5,
        rescale: bool = True,
    ) -> nn.Module:
        return dare_merge(self.models, self.weights, drop_rate, rescale)


def slerp_merge(
    model1: nn.Module,
    model2: nn.Module,
    t: float = 0.5,
) -> nn.Module:
    merged_model = copy.deepcopy(model1)
    
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    merged_state = {}
    
    for key in state_dict1:
        if key not in state_dict2:
            merged_state[key] = state_dict1[key]
            continue
        
        param1 = state_dict1[key].float()
        param2 = state_dict2[key].float()
        
        omega = torch.acos(torch.clamp(
            torch.sum(param1 * param2) / (torch.norm(param1) * torch.norm(param2) + 1e-8),
            -1.0, 1.0
        ))
        
        sin_omega = torch.sin(omega)
        if sin_omega < 1e-8:
            merged_state[key] = (1 - t) * param1 + t * param2
        else:
            merged_state[key] = (
                torch.sin((1 - t) * omega) / sin_omega * param1 +
                torch.sin(t * omega) / sin_omega * param2
            )
        
        merged_state[key] = merged_state[key].to(state_dict1[key].dtype)
    
    merged_model.load_state_dict(merged_state, strict=False)
    return merged_model


def ties_merge(
    models: List[nn.Module],
    weights: List[float],
    density: float = 0.6,
    weight_mask_rate: float = 0.2,
) -> nn.Module:
    if len(models) < 2:
        return copy.deepcopy(models[0])
    
    base_model = models[0]
    merged_model = copy.deepcopy(base_model)
    
    base_state = base_model.state_dict()
    task_vectors = []
    
    for model in models[1:]:
        state = model.state_dict()
        task_vector = {}
        for key in base_state:
            if key in state:
                task_vector[key] = state[key] - base_state[key]
        task_vectors.append(task_vector)
    
    trimmed_vectors = []
    for tv in task_vectors:
        trimmed = {}
        for key, param in tv.items():
            flat = param.view(-1)
            k = int(density * flat.numel())
            if k > 0:
                topk = torch.topk(flat.abs(), k)
                mask = torch.zeros_like(flat)
                mask[topk.indices] = 1
                trimmed[key] = param * mask.view(param.shape)
            else:
                trimmed[key] = torch.zeros_like(param)
        trimmed_vectors.append(trimmed)
    
    merged_delta = {}
    for key in base_state:
        deltas = [tv.get(key, torch.zeros_like(base_state[key])) for tv in trimmed_vectors]
        if not deltas:
            continue
        
        stacked = torch.stack(deltas)
        signs = torch.sign(stacked)
        sign_agreement = signs.sum(dim=0).abs()
        
        majority_sign = torch.sign(signs.sum(dim=0))
        
        masked = []
        for delta in deltas:
            mask = (torch.sign(delta) == majority_sign).float()
            masked.append(delta * mask)
        
        avg_delta = sum(masked) / len(masked)
        
        flat = avg_delta.view(-1)
        k = int((1 - weight_mask_rate) * flat.numel())
        if k > 0:
            topk = torch.topk(flat.abs(), k)
            mask = torch.zeros_like(flat)
            mask[topk.indices] = 1
            avg_delta = avg_delta * mask.view(avg_delta.shape)
        
        merged_delta[key] = avg_delta
    
    merged_state = {}
    for key in base_state:
        if key in merged_delta:
            merged_state[key] = base_state[key] + merged_delta[key]
        else:
            merged_state[key] = base_state[key]
    
    merged_model.load_state_dict(merged_state, strict=False)
    return merged_model


def dare_merge(
    models: List[nn.Module],
    weights: List[float],
    drop_rate: float = 0.5,
    rescale: bool = True,
) -> nn.Module:
    if len(models) < 2:
        return copy.deepcopy(models[0])
    
    base_model = models[0]
    merged_model = copy.deepcopy(base_model)
    
    base_state = base_model.state_dict()
    task_vectors = []
    
    for model in models[1:]:
        state = model.state_dict()
        task_vector = {}
        for key in base_state:
            if key in state:
                diff = state[key] - base_state[key]
                mask = (torch.rand_like(diff) > drop_rate).float()
                if rescale:
                    mask = mask / (1 - drop_rate)
                task_vector[key] = diff * mask
        task_vectors.append(task_vector)
    
    merged_state = {}
    for key in base_state:
        deltas = [tv.get(key, torch.zeros_like(base_state[key])) for tv in task_vectors]
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            merged_state[key] = base_state[key] + avg_delta
        else:
            merged_state[key] = base_state[key]
    
    merged_model.load_state_dict(merged_state, strict=False)
    return merged_model
