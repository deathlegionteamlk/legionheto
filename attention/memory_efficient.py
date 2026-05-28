import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MemoryEfficientAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        
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
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if self.num_key_value_heads != self.num_heads:
            key_states = key_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
            value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        kv_seq_len = key_states.shape[2]
        
        if seq_len <= 1024 or kv_seq_len <= 2048:
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, value_states)
        else:
            attn_output = self._memory_efficient_attention(
                query_states, key_states, value_states, attention_mask
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value
    
    def _memory_efficient_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        batch_size, num_heads, q_len, head_dim = query.shape
        _, _, kv_len, _ = key.shape
        
        num_chunks = (q_len + chunk_size - 1) // chunk_size
        outputs = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, q_len)
            q_chunk = query[:, :, start_idx:end_idx, :]
            
            attn_weights = torch.matmul(q_chunk, key.transpose(-2, -1)) / math.sqrt(head_dim)
            
            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, start_idx:end_idx, :]
                attn_weights = attn_weights + mask_chunk
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            out_chunk = torch.matmul(attn_weights, value)
            outputs.append(out_chunk)
        
        return torch.cat(outputs, dim=2)
