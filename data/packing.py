import torch
from typing import List, Dict, Any


def pack_sequences(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int = 2048,
    pad_to_multiple_of: int = 8,
) -> Dict[str, torch.Tensor]:
    packed_input_ids = []
    packed_attention_mask = []
    packed_labels = []
    
    current_input_ids = []
    current_labels = []
    current_length = 0
    
    for example in examples:
        input_ids = example["input_ids"]
        labels = example.get("labels", input_ids.copy())
        
        if current_length + len(input_ids) > max_seq_length:
            if current_input_ids:
                padding_length = max_seq_length - len(current_input_ids)
                current_input_ids.extend([tokenizer.pad_token_id] * padding_length)
                current_labels.extend([-100] * padding_length)
                
                packed_input_ids.append(current_input_ids)
                packed_attention_mask.append([1 if i < current_length else 0 for i in range(max_seq_length)])
                packed_labels.append(current_labels)
            
            current_input_ids = input_ids.copy()
            current_labels = labels.copy()
            current_length = len(input_ids)
        else:
            current_input_ids.extend(input_ids)
            current_labels.extend(labels)
            current_length += len(input_ids)
    
    if current_input_ids:
        padding_length = max_seq_length - len(current_input_ids)
        current_input_ids.extend([tokenizer.pad_token_id] * padding_length)
        current_labels.extend([-100] * padding_length)
        
        packed_input_ids.append(current_input_ids)
        packed_attention_mask.append([1 if i < current_length else 0 for i in range(max_seq_length)])
        packed_labels.append(current_labels)
    
    return {
        "input_ids": torch.tensor(packed_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(packed_attention_mask, dtype=torch.long),
        "labels": torch.tensor(packed_labels, dtype=torch.long),
    }


def pack_sequences_efficient(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int = 2048,
) -> List[Dict[str, torch.Tensor]]:
    sequences = []
    current_tokens = []
    current_labels = []
    
    for example in examples:
        input_ids = example["input_ids"]
        labels = example.get("labels", [-100] * len(input_ids))
        
        if len(current_tokens) + len(input_ids) <= max_seq_length:
            current_tokens.extend(input_ids)
            current_labels.extend(labels)
        else:
            if current_tokens:
                sequences.append({
                    "input_ids": current_tokens[:max_seq_length],
                    "labels": current_labels[:max_seq_length],
                })
            
            current_tokens = input_ids[:max_seq_length]
            current_labels = labels[:max_seq_length]
    
    if current_tokens:
        sequences.append({
            "input_ids": current_tokens[:max_seq_length],
            "labels": current_labels[:max_seq_length],
        })
    
    return sequences
