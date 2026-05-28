import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class MLAConfig:
    embed_dim: int = 4096
    num_heads: int = 32
    num_key_value_heads: int = 8
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    dropout: float = 0.0


class DeepSeekMLA(nn.Module):
    def __init__(self, config: MLAConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.dropout = config.dropout
        
        self.q_head_dim = config.qk_rope_head_dim + self.v_head_dim
        
        self.q_down_proj = nn.Linear(config.embed_dim, config.q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        
        self.kv_down_proj = nn.Linear(config.embed_dim, config.kv_lora_rank, bias=False)
        self.kv_up_proj = nn.Linear(config.kv_lora_rank, self.num_heads * self.q_head_dim, bias=False)
        
        self.k_rope_proj = nn.Linear(config.embed_dim, self.qk_rope_head_dim, bias=False)
        
        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, config.embed_dim, bias=False)
        
        self.q_norm = nn.LayerNorm(config.q_lora_rank)
        self.kv_norm = nn.LayerNorm(config.kv_lora_rank)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        q_compressed = self.q_norm(self.q_down_proj(hidden_states))
        q = self.q_up_proj(q_compressed)
        q = q.view(batch_size, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        
        q_nope = q[..., :self.v_head_dim]
        q_pe = q[..., self.v_head_dim:]
        
        kv_compressed = self.kv_norm(self.kv_down_proj(hidden_states))
        kv = self.kv_up_proj(kv_compressed)
        kv = kv.view(batch_size, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        
        k_nope = kv[..., :self.v_head_dim]
        k_pe = self.k_rope_proj(hidden_states)
        k_pe = k_pe.view(batch_size, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)
        
        v = kv[..., :self.v_head_dim]
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k_nope = torch.cat([past_k, k_nope], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        past_key_value = (k_nope, v) if use_cache else None
        
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.q_head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, past_key_value
    
    def get_cache_size(self, seq_len: int) -> int:
        return self.num_heads * seq_len * self.v_head_dim * 2
