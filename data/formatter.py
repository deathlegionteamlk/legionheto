from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class DataFormatter(ABC):
    @abstractmethod
    def format(self, item: Dict[str, Any]) -> str:
        pass


class AlpacaFormatter(DataFormatter):
    def __init__(
        self,
        instruction_col: str = "instruction",
        input_col: str = "input",
        output_col: str = "output",
    ):
        self.instruction_col = instruction_col
        self.input_col = input_col
        self.output_col = output_col
    
    def format(self, item: Dict[str, Any]) -> str:
        instruction = item.get(self.instruction_col, "")
        input_text = item.get(self.input_col, "")
        output = item.get(self.output_col, "")
        
        if input_text:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


class ChatFormatter(DataFormatter):
    def __init__(
        self,
        messages_col: str = "messages",
        system_message: Optional[str] = None,
    ):
        self.messages_col = messages_col
        self.system_message = system_message
    
    def format(self, item: Dict[str, Any]) -> str:
        messages = item.get(self.messages_col, [])
        
        formatted = ""
        if self.system_message:
            formatted += f"System: {self.system_message}\n\n"
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role.capitalize()}: {content}\n"
        
        return formatted.strip()


class ShareGPTFormatter(DataFormatter):
    def __init__(self, conversations_col: str = "conversations"):
        self.conversations_col = conversations_col
    
    def format(self, item: Dict[str, Any]) -> str:
        conversations = item.get(self.conversations_col, [])
        
        formatted = ""
        for turn in conversations:
            from_role = turn.get("from", "human")
            value = turn.get("value", "")
            
            if from_role == "human":
                formatted += f"User: {value}\n"
            elif from_role == "gpt":
                formatted += f"Assistant: {value}\n"
        
        return formatted.strip()


class CompletionFormatter(DataFormatter):
    def __init__(
        self,
        text_col: str = "text",
        prefix: str = "",
        suffix: str = "",
    ):
        self.text_col = text_col
        self.prefix = prefix
        self.suffix = suffix
    
    def format(self, item: Dict[str, Any]) -> str:
        text = item.get(self.text_col, "")
        return f"{self.prefix}{text}{self.suffix}"


class DPOFormatter:
    def __init__(
        self,
        prompt_col: str = "prompt",
        chosen_col: str = "chosen",
        rejected_col: str = "rejected",
    ):
        self.prompt_col = prompt_col
        self.chosen_col = chosen_col
        self.rejected_col = rejected_col
    
    def format(self, item: Dict[str, Any]) -> Dict[str, str]:
        return {
            "prompt": item.get(self.prompt_col, ""),
            "chosen": item.get(self.chosen_col, ""),
            "rejected": item.get(self.rejected_col, ""),
        }


class ORPOFormatter(DPOFormatter):
    pass
