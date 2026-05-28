import torch
import struct
import os
from typing import Dict, Any, Optional, List
import numpy as np


GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15


class GGUFWriter:
    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, np.ndarray] = {}
        self.tensor_types: Dict[str, int] = {}
    
    def add_metadata(self, key: str, value: Any):
        self.metadata[key] = value
    
    def add_tensor(self, name: str, tensor: torch.Tensor, tensor_type: int = GGML_TYPE_F16):
        if tensor.dtype == torch.float32:
            np_tensor = tensor.detach().cpu().numpy().astype(np.float32)
        elif tensor.dtype == torch.float16:
            np_tensor = tensor.detach().cpu().numpy().astype(np.float16)
        else:
            np_tensor = tensor.detach().cpu().float().numpy()
        
        self.tensors[name] = np_tensor
        self.tensor_types[name] = tensor_type
    
    def write(self, filepath: str):
        with open(filepath, 'wb') as f:
            f.write(GGUF_MAGIC)
            f.write(struct.pack('<I', GGUF_VERSION))
            
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.metadata)))
            
            for key, value in self.metadata.items():
                self._write_string(f, key)
                self._write_value(f, value)
            
            tensor_info_offset = f.tell()
            tensor_data_offset = tensor_info_offset + self._calculate_tensor_info_size()
            tensor_data_offset = (tensor_data_offset + 31) & ~31
            
            current_offset = tensor_data_offset
            for name, tensor in self.tensors.items():
                self._write_string(f, name)
                f.write(struct.pack('<I', self.tensor_types[name]))
                f.write(struct.pack('<Q', len(tensor.shape)))
                for dim in tensor.shape:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<Q', current_offset))
                current_offset += tensor.nbytes
            
            f.seek(tensor_data_offset)
            for name, tensor in self.tensors.items():
                f.write(tensor.tobytes())
    
    def _write_string(self, f, s: str):
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
    
    def _write_value(self, f, value: Any):
        if isinstance(value, str):
            f.write(struct.pack('<I', 0))
            self._write_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', 4))
            f.write(struct.pack('<q', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 5))
            f.write(struct.pack('<d', value))
        elif isinstance(value, bool):
            f.write(struct.pack('<I', 6))
            f.write(struct.pack('<?', value))
        elif isinstance(value, list):
            f.write(struct.pack('<I', 7))
            f.write(struct.pack('<Q', len(value)))
            for item in value:
                self._write_value(f, item)
    
    def _calculate_tensor_info_size(self) -> int:
        size = 0
        for name in self.tensors:
            size += 8 + len(name.encode('utf-8'))
            size += 4
            size += 8
            tensor = self.tensors[name]
            size += 8 * len(tensor.shape)
            size += 8
        return size


class GGUFExporter:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.writer = GGUFWriter()
    
    def export(
        self,
        output_path: str,
        quantization: str = "Q4_K_M",
    ):
        self._add_metadata()
        self._add_tensors(quantization)
        self.writer.write(output_path)
    
    def _add_metadata(self):
        config = self.model.config if hasattr(self.model, 'config') else {}
        
        self.writer.add_metadata("general.architecture", "llama")
        self.writer.add_metadata("general.name", getattr(config, '_name_or_path', 'model'))
        self.writer.add_metadata("general.quantization_version", GGUF_VERSION)
        
        self.writer.add_metadata("llama.context_length", getattr(config, 'max_position_embeddings', 4096))
        self.writer.add_metadata("llama.embedding_length", getattr(config, 'hidden_size', 4096))
        self.writer.add_metadata("llama.block_count", getattr(config, 'num_hidden_layers', 32))
        self.writer.add_metadata("llama.feed_forward_length", getattr(config, 'intermediate_size', 11008))
        self.writer.add_metadata("llama.attention.head_count", getattr(config, 'num_attention_heads', 32))
        self.writer.add_metadata("llama.attention.head_count_kv", getattr(config, 'num_key_value_heads', 32))
        self.writer.add_metadata("llama.attention.layer_norm_rms_epsilon", getattr(config, 'rms_norm_eps', 1e-6))
        self.writer.add_metadata("llama.rope.dimension_count", getattr(config, 'hidden_size', 4096) // getattr(config, 'num_attention_heads', 32))
        
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}
        self.writer.add_metadata("tokenizer.ggml.model", "llama")
        self.writer.add_metadata("tokenizer.ggml.tokens", list(vocab.keys()))
    
    def _add_tensors(self, quantization: str):
        tensor_type_map = {
            "Q4_0": GGML_TYPE_Q4_0,
            "Q4_1": GGML_TYPE_Q4_1,
            "Q5_0": GGML_TYPE_Q5_0,
            "Q5_1": GGML_TYPE_Q5_1,
            "Q8_0": GGML_TYPE_Q8_0,
            "Q2_K": GGML_TYPE_Q2_K,
            "Q3_K": GGML_TYPE_Q3_K,
            "Q4_K": GGML_TYPE_Q4_K,
            "Q4_K_M": GGML_TYPE_Q4_K,
            "Q5_K": GGML_TYPE_Q5_K,
            "Q5_K_M": GGML_TYPE_Q5_K,
            "Q6_K": GGML_TYPE_Q6_K,
            "Q8_K": GGML_TYPE_Q8_K,
            "F16": GGML_TYPE_F16,
            "F32": GGML_TYPE_F32,
        }
        
        tensor_type = tensor_type_map.get(quantization, GGML_TYPE_Q4_K)
        
        state_dict = self.model.state_dict() if hasattr(self.model, 'state_dict') else {}
        
        for name, tensor in state_dict.items():
            gguf_name = self._map_tensor_name(name)
            self.writer.add_tensor(gguf_name, tensor, tensor_type)
    
    def _map_tensor_name(self, name: str) -> str:
        name_map = {
            "model.embed_tokens.weight": "token_embd.weight",
            "model.norm.weight": "output_norm.weight",
            "lm_head.weight": "output.weight",
        }
        
        if name in name_map:
            return name_map[name]
        
        if "layers." in name:
            name = name.replace("model.layers.", "blk.")
            name = name.replace(".self_attn.q_proj", ".attn_q")
            name = name.replace(".self_attn.k_proj", ".attn_k")
            name = name.replace(".self_attn.v_proj", ".attn_v")
            name = name.replace(".self_attn.o_proj", ".attn_output")
            name = name.replace(".mlp.gate_proj", ".ffn_gate")
            name = name.replace(".mlp.up_proj", ".ffn_up")
            name = name.replace(".mlp.down_proj", ".ffn_down")
            name = name.replace(".input_layernorm", ".attn_norm")
            name = name.replace(".post_attention_layernorm", ".ffn_norm")
        
        return name


def export_to_gguf(
    model,
    tokenizer,
    output_path: str,
    quantization: str = "Q4_K_M",
):
    exporter = GGUFExporter(model, tokenizer)
    exporter.export(output_path, quantization)
