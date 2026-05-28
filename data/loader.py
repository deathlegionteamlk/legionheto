import json
import os
from typing import Optional, Iterator, Dict, Any, List, Callable
from datasets import load_dataset, Dataset
import torch
from torch.utils.data import IterableDataset


class StreamingDataset(IterableDataset):
    def __init__(
        self,
        data_source: str,
        tokenizer,
        max_seq_length: int = 2048,
        format_fn: Optional[Callable] = None,
        batch_size: int = 1000,
    ):
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.format_fn = format_fn
        self.batch_size = batch_size
        
    def _load_stream(self):
        if self.data_source.endswith(".jsonl"):
            with open(self.data_source, 'r') as f:
                for line in f:
                    yield json.loads(line)
        elif self.data_source.endswith(".json"):
            with open(self.data_source, 'r') as f:
                data = json.load(f)
                for item in data:
                    yield item
        elif os.path.isdir(self.data_source):
            for filename in os.listdir(self.data_source):
                if filename.endswith(".jsonl"):
                    filepath = os.path.join(self.data_source, filename)
                    with open(filepath, 'r') as f:
                        for line in f:
                            yield json.loads(line)
        else:
            dataset = load_dataset(self.data_source, streaming=True)
            for item in dataset["train"]:
                yield item
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        
        for item in self._load_stream():
            if self.format_fn:
                text = self.format_fn(item)
            else:
                text = item.get("text", "")
            
            if not text:
                continue
            
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            yield {
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }
    
    def __len__(self):
        return 1000000


class DataLoader:
    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 2048,
        format_type: str = "alpaca",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.format_type = format_type
    
    def load(
        self,
        source: str,
        split: str = "train",
        streaming: bool = False,
        **kwargs,
    ):
        if streaming:
            return StreamingDataset(
                source,
                self.tokenizer,
                self.max_seq_length,
                format_fn=self._get_format_fn(),
            )
        
        if source.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=source, split=split)
        elif source.endswith(".json"):
            dataset = load_dataset("json", data_files=source, split=split)
        elif source.endswith(".csv"):
            dataset = load_dataset("csv", data_files=source, split=split)
        elif source.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=source, split=split)
        else:
            dataset = load_dataset(source, split=split, **kwargs)
        
        return dataset
    
    def _get_format_fn(self):
        if self.format_type == "alpaca":
            return self._format_alpaca
        elif self.format_type == "chat":
            return self._format_chat
        return None
    
    def _format_alpaca(self, item: Dict[str, Any]) -> str:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if input_text:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    def _format_chat(self, item: Dict[str, Any]) -> str:
        messages = item.get("messages", [])
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role.capitalize()}: {content}\n"
        return formatted.strip()
    
    def tokenize(
        self,
        dataset: Dataset,
        text_column: str = "text",
        add_eos_token: bool = True,
    ) -> Dataset:
        def tokenize_function(examples):
            texts = examples[text_column]
            
            if add_eos_token and self.tokenizer.eos_token:
                texts = [text + self.tokenizer.eos_token for text in texts]
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        return dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
