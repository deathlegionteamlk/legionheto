import json
import random
import re
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SeedTask:
    instruction: str
    input_text: str
    output_text: str
    category: str = "general"


class SelfInstructGenerator:
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_few_shot: int = 3,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.num_few_shot = num_few_shot
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        self.seed_tasks: List[SeedTask] = []
        
    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
    def add_seed_tasks(self, tasks: List[Dict]):
        for task in tasks:
            self.seed_tasks.append(SeedTask(
                instruction=task.get("instruction", ""),
                input_text=task.get("input", ""),
                output_text=task.get("output", ""),
                category=task.get("category", "general")
            ))
            
    def load_seed_tasks_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                self.add_seed_tasks(data)
            elif isinstance(data, dict) and "tasks" in data:
                self.add_seed_tasks(data["tasks"])
                
    def _create_generation_prompt(self) -> str:
        if len(self.seed_tasks) < self.num_few_shot:
            selected_tasks = self.seed_tasks
        else:
            selected_tasks = random.sample(self.seed_tasks, self.num_few_shot)
            
        examples = []
        for task in selected_tasks:
            examples.append(f"Instruction: {task.instruction}")
            if task.input_text:
                examples.append(f"Input: {task.input_text}")
            examples.append(f"Output: {task.output_text}")
            examples.append("")
            
        prompt = "You are a helpful assistant. Generate a new instruction-response pair similar to these examples:\n\n"
        prompt += "\n".join(examples)
        prompt += "\nInstruction:"
        
        return prompt
        
    def _parse_generated_text(self, text: str) -> Optional[Dict]:
        lines = text.strip().split('\n')
        
        instruction = ""
        input_text = ""
        output_text = ""
        
        current_field = None
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith("instruction:"):
                instruction = line.split(":", 1)[1].strip()
                current_field = "instruction"
            elif line.lower().startswith("input:"):
                input_text = line.split(":", 1)[1].strip()
                current_field = "input"
            elif line.lower().startswith("output:"):
                output_text = line.split(":", 1)[1].strip()
                current_field = "output"
            elif current_field == "instruction" and instruction and not input_text and not output_text:
                instruction += " " + line
            elif current_field == "output" and output_text:
                output_text += " " + line
                
        if instruction and output_text:
            return {
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
            }
        return None
        
    def _calculate_diversity(self, samples: List[Dict]) -> float:
        if len(samples) < 2:
            return 1.0
            
        instructions = [s["instruction"].lower() for s in samples]
        total_similarity = 0
        count = 0
        
        for i in range(len(instructions)):
            for j in range(i + 1, len(instructions)):
                words_i = set(instructions[i].split())
                words_j = set(instructions[j].split())
                
                if words_i and words_j:
                    intersection = len(words_i & words_j)
                    union = len(words_i | words_j)
                    similarity = intersection / union if union > 0 else 0
                    total_similarity += similarity
                    count += 1
                    
        avg_similarity = total_similarity / count if count > 0 else 0
        diversity = 1 - avg_similarity
        
        return diversity
        
    def generate(
        self,
        num_samples: int = 100,
        filter_quality: bool = True,
        diversity_threshold: float = 0.7,
        max_retries: int = 3,
    ) -> List[Dict]:
        self.load_model()
        
        generated_samples = []
        retries = 0
        
        while len(generated_samples) < num_samples and retries < max_retries * num_samples:
            prompt = self._create_generation_prompt()
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = generated_text[len(prompt):].strip()
            
            parsed = self._parse_generated_text(generated_text)
            
            if parsed:
                if filter_quality:
                    if self._is_quality_sample(parsed):
                        generated_samples.append(parsed)
                else:
                    generated_samples.append(parsed)
                    
            retries += 1
            
        diversity = self._calculate_diversity(generated_samples)
        
        if diversity < diversity_threshold and len(generated_samples) > 10:
            random.shuffle(generated_samples)
            
        return generated_samples
        
    def _is_quality_sample(self, sample: Dict) -> bool:
        instruction = sample.get("instruction", "")
        output = sample.get("output", "")
        
        if len(instruction) < 10 or len(output) < 10:
            return False
            
        if len(instruction) > 500 or len(output) > 2000:
            return False
            
        low_quality_patterns = [
            r"^\d+\.",
            r"^\*\*",
            r"^\[",
            r"^\{",
        ]
        
        for pattern in low_quality_patterns:
            if re.match(pattern, instruction):
                return False
                
        return True
        
    def export_to_json(self, samples: List[Dict], filepath: str):
        with open(filepath, 'w') as f:
            json.dump(samples, f, indent=2)
            
    def export_to_alpaca_format(self, samples: List[Dict], filepath: str):
        alpaca_data = []
        for sample in samples:
            alpaca_data.append({
                "instruction": sample["instruction"],
                "input": sample.get("input", ""),
                "output": sample["output"],
            })
            
        with open(filepath, 'w') as f:
            json.dump(alpaca_data, f, indent=2)


class EvolInstructGenerator:
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
    ):
        self.model_name = model_name
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        
        self.evol_prompts = [
            "Make this instruction more complex by adding specific constraints:",
            "Rewrite this instruction to require deeper reasoning:",
            "Expand this instruction with additional context and requirements:",
            "Make this instruction more specific and detailed:",
            "Add a scenario to this instruction to make it more practical:",
        ]
        
    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
    def evolve_instruction(
        self,
        instruction: str,
        num_evolutions: int = 3,
        temperature: float = 0.8,
    ) -> List[str]:
        self.load_model()
        
        evolved_instructions = []
        
        for _ in range(num_evolutions):
            evol_prompt = random.choice(self.evol_prompts)
            full_prompt = f"{evol_prompt}\n\nOriginal: {instruction}\n\nEvolved:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
            evolved = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            evolved = evolved[len(full_prompt):].strip().split('\n')[0]
            
            if evolved and len(evolved) > 20:
                evolved_instructions.append(evolved)
                
        return evolved_instructions
        
    def generate_responses(
        self,
        instructions: List[str],
        temperature: float = 0.7,
    ) -> List[Dict]:
        self.load_model()
        
        results = []
        
        for instruction in instructions:
            prompt = f"Instruction: {instruction}\n\nResponse:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            results.append({
                "instruction": instruction,
                "input": "",
                "output": response,
            })
            
        return results
