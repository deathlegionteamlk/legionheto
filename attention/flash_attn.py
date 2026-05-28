import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def has_flash_attn() -> bool:
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def get_attention_backend() -> str:
    if has_flash_attn():
        return "flash_attn"
    elif hasattr(F, "scaled_dot_product_attention"):
        return "sdpa"
    return "eager"


class FlashAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        use_flash: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        self.use_flash = use_flash and has_flash_attn()
        self.backend = get_attention_backend()
        
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)
        
        self.rotary_emb = None
        
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
        
        kv_seq_len = key_states.shape[2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[2]
        
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        if self.use_flash and has_flash_attn():
            from flash_attn import flash_attn_func
            attn_output = flash_attn_func(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
            )
            attn_output = attn_output.transpose(1, 2)
        elif self.backend == "sdpa":
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=attention_mask is None,
            )
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        return q, k


class MemoryEfficientAttention(FlashAttention):
    pass
