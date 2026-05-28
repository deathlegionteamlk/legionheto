from typing import Dict, List, Optional, Type, Any
import re


class ModelRegistry:
    ARCHITECTURE_MAP = {
        "llama": {
            "config_class": "LlamaConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "mistral": {
            "config_class": "MistralConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "mixtral": {
            "config_class": "MixtralConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "qwen": {
            "config_class": "QwenConfig",
            "target_modules": ["c_attn", "c_proj", "w1", "w2"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "qwen2": {
            "config_class": "Qwen2Config",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "deepseek": {
            "config_class": "DeepseekConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
            "supports_mla": True,
        },
        "phi": {
            "config_class": "PhiConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "phi3": {
            "config_class": "Phi3Config",
            "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "gemma": {
            "config_class": "GemmaConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "gemma2": {
            "config_class": "Gemma2Config",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "command": {
            "config_class": "CohereConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "falcon": {
            "config_class": "FalconConfig",
            "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "falcon_mamba": {
            "config_class": "FalconMambaConfig",
            "target_modules": ["in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"],
            "model_type": "mamba",
            "supports_flash_attn": False,
        },
        "yi": {
            "config_class": "YiConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "stablelm": {
            "config_class": "StableLmConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "codellama": {
            "config_class": "LlamaConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "starcoder": {
            "config_class": "GPTBigCodeConfig",
            "target_modules": ["c_attn", "c_proj", "c_fc"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "starcoder2": {
            "config_class": "Starcoder2Config",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "dbrx": {
            "config_class": "DbrxConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "v1", "v2"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "gpt2": {
            "config_class": "GPT2Config",
            "target_modules": ["c_attn", "c_proj", "c_fc"],
            "model_type": "causal_lm",
            "supports_flash_attn": False,
        },
        "gpt_neox": {
            "config_class": "GPTNeoXConfig",
            "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "gptj": {
            "config_class": "GPTJConfig",
            "target_modules": ["q_proj", "v_proj", "out_proj", "fc_in", "fc_out"],
            "model_type": "causal_lm",
            "supports_flash_attn": False,
        },
        "bloom": {
            "config_class": "BloomConfig",
            "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "mamba": {
            "config_class": "MambaConfig",
            "target_modules": ["in_proj", "conv1d", "x_proj", "dt_proj", "out_proj"],
            "model_type": "mamba",
            "supports_flash_attn": False,
        },
        "granite": {
            "config_class": "GraniteConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "nemotron": {
            "config_class": "NemotronConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
        "openelm": {
            "config_class": "OpenELMConfig",
            "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "ffn_proj", "ffn_down"],
            "model_type": "causal_lm",
            "supports_flash_attn": True,
        },
    }

    NAME_PATTERNS = {
        r"llama[-_]?3": "llama",
        r"llama[-_]?2": "llama",
        r"llama": "llama",
        r"mistral": "mistral",
        r"mixtral": "mixtral",
        r"qwen2": "qwen2",
        r"qwen": "qwen",
        r"deepseek": "deepseek",
        r"phi[-_]?3": "phi3",
        r"phi": "phi",
        r"gemma[-_]?2": "gemma2",
        r"gemma": "gemma",
        r"command": "command",
        r"cohere": "command",
        r"falcon[-_]?mamba": "falcon_mamba",
        r"falcon": "falcon",
        r"yi": "yi",
        r"stablelm": "stablelm",
        r"code[-_]?llama": "codellama",
        r"starcoder2": "starcoder2",
        r"starcoder": "starcoder",
        r"dbrx": "dbrx",
        r"gpt[-_]?neox": "gpt_neox",
        r"gpt[-_]?j": "gptj",
        r"gpt2": "gpt2",
        r"bloom": "bloom",
        r"mamba": "mamba",
        r"granite": "granite",
        r"nemotron": "nemotron",
        r"openelm": "openelm",
    }

    @classmethod
    def detect_architecture(cls, model_name_or_path: str) -> Optional[str]:
        model_lower = model_name_or_path.lower()
        
        for pattern, arch in cls.NAME_PATTERNS.items():
            if re.search(pattern, model_lower):
                return arch
        
        return None

    @classmethod
    def get_config(cls, arch: str) -> Dict[str, Any]:
        return cls.ARCHITECTURE_MAP.get(arch, {})

    @classmethod
    def get_target_modules(cls, arch: str) -> List[str]:
        config = cls.get_config(arch)
        return config.get("target_modules", ["q_proj", "v_proj"])

    @classmethod
    def supports_flash_attn(cls, arch: str) -> bool:
        config = cls.get_config(arch)
        return config.get("supports_flash_attn", False)

    @classmethod
    def supports_mla(cls, arch: str) -> bool:
        config = cls.get_config(arch)
        return config.get("supports_mla", False)

    @classmethod
    def list_supported_architectures(cls) -> List[str]:
        return list(cls.ARCHITECTURE_MAP.keys())


class ArchitectureDetector:
    @staticmethod
    def from_model_name(model_name: str) -> Optional[str]:
        return ModelRegistry.detect_architecture(model_name)
    
    @staticmethod
    def from_config(config) -> Optional[str]:
        model_type = getattr(config, "model_type", None)
        
        type_map = {
            "llama": "llama",
            "mistral": "mistral",
            "mixtral": "mixtral",
            "qwen": "qwen",
            "qwen2": "qwen2",
            "deepseek": "deepseek",
            "phi": "phi",
            "phi3": "phi3",
            "gemma": "gemma",
            "gemma2": "gemma2",
            "cohere": "command",
            "falcon": "falcon",
            "falcon_mamba": "falcon_mamba",
            "yi": "yi",
            "stablelm": "stablelm",
            "gpt_bigcode": "starcoder",
            "starcoder2": "starcoder2",
            "dbrx": "dbrx",
            "gpt2": "gpt2",
            "gpt_neox": "gpt_neox",
            "gptj": "gptj",
            "bloom": "bloom",
            "mamba": "mamba",
            "granite": "granite",
            "nemotron": "nemotron",
            "openelm": "openelm",
        }
        
        return type_map.get(model_type)
