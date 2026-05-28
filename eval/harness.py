import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvaluationConfig:
    tasks: List[str]
    batch_size: int = 1
    device: str = "auto"
    num_fewshot: int = 0
    limit: Optional[int] = None
    output_path: str = "./eval_results"


class LegionHetoHarness:
    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[EvaluationConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig(tasks=[])
        
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
            
        self.results = {}
        
    def evaluate(self, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        if tasks is None:
            tasks = self.config.tasks
            
        for task in tasks:
            print(f"Evaluating on {task}...")
            
            if task == "hellaswag":
                self.results[task] = self._eval_hellaswag()
            elif task == "arc_easy":
                self.results[task] = self._eval_arc(difficulty="easy")
            elif task == "arc_challenge":
                self.results[task] = self._eval_arc(difficulty="challenge")
            elif task == "winogrande":
                self.results[task] = self._eval_winogrande()
            elif task == "truthfulqa":
                self.results[task] = self._eval_truthfulqa()
            elif task == "boolq":
                self.results[task] = self._eval_boolq()
            elif task == "piqa":
                self.results[task] = self._eval_piqa()
            elif task == "siqa":
                self.results[task] = self._eval_siqa()
            elif task == "openbookqa":
                self.results[task] = self._eval_openbookqa()
            elif task == "copa":
                self.results[task] = self._eval_copa()
            else:
                print(f"Task {task} not implemented, skipping...")
                
        return self.results
        
    def _eval_hellaswag(self) -> Dict[str, float]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)
            
            if self.config.limit:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
                
            correct = 0
            total = 0
            
            for example in dataset:
                context = example["ctx"]
                endings = example["endings"]
                label = int(example["label"])
                
                scores = []
                for ending in endings:
                    prompt = f"{context} {ending}"
                    score = self._score_text(prompt)
                    scores.append(score)
                    
                predicted = scores.index(max(scores))
                if predicted == label:
                    correct += 1
                total += 1
                
            accuracy = correct / total if total > 0 else 0
            return {"accuracy": accuracy, "correct": correct, "total": total}
            
        except Exception as e:
            print(f"Error in hellaswag evaluation: {e}")
            return {"accuracy": 0.0, "error": str(e)}
            
    def _eval_arc(self, difficulty: str = "easy") -> Dict[str, float]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("ai2_arc", f"ARC-{difficulty.capitalize()}", split="test", trust_remote_code=True)
            
            if self.config.limit:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
                
            correct = 0
            total = 0
            
            for example in dataset:
                question = example["question"]
                choices = example["choices"]
                choice_texts = choices["text"]
                choice_labels = choices["label"]
                answer_key = example["answerKey"]
                
                scores = []
                for choice_text in choice_texts:
                    prompt = f"Question: {question}\nAnswer: {choice_text}"
                    score = self._score_text(prompt)
                    scores.append(score)
                    
                predicted_idx = scores.index(max(scores))
                predicted_label = choice_labels[predicted_idx]
                
                if predicted_label == answer_key:
                    correct += 1
                total += 1
                
            accuracy = correct / total if total > 0 else 0
            return {"accuracy": accuracy, "correct": correct, "total": total}
            
        except Exception as e:
            print(f"Error in arc evaluation: {e}")
            return {"accuracy": 0.0, "error": str(e)}
            
    def _eval_winogrande(self) -> Dict[str, float]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
            
            if self.config.limit:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
                
            correct = 0
            total = 0
            
            for example in dataset:
                sentence = example["sentence"]
                option1 = example["option1"]
                option2 = example["option2"]
                answer = example["answer"]
                
                scores = []
                for option in [option1, option2]:
                    filled = sentence.replace("_", option)
                    score = self._score_text(filled)
                    scores.append(score)
                    
                predicted = scores.index(max(scores)) + 1
                if str(predicted) == str(answer):
                    correct += 1
                total += 1
                
            accuracy = correct / total if total > 0 else 0
            return {"accuracy": accuracy, "correct": correct, "total": total}
            
        except Exception as e:
            print(f"Error in winogrande evaluation: {e}")
            return {"accuracy": 0.0, "error": str(e)}
            
    def _eval_truthfulqa(self) -> Dict[str, float]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True)
            
            if self.config.limit:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
                
            results = []
            
            for example in dataset:
                question = example["question"]
                best_answer = example["best_answer"]
                
                generated = self._generate_text(f"Q: {question}\nA:", max_new_tokens=100)
                results.append({
                    "question": question,
                    "generated": generated,
                    "reference": best_answer,
                })
                
            return {"num_samples": len(results), "samples": results[:10]}
            
        except Exception as e:
            print(f"Error in truthfulqa evaluation: {e}")
            return {"error": str(e)}
            
    def _eval_boolq(self) -> Dict[str, float]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("boolq", split="validation", trust_remote_code=True)
            
            if self.config.limit:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
                
            correct = 0
            total = 0
            
            for example in dataset:
                passage = example["passage"]
                question = example["question"]
                answer = example["answer"]
                
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
                generated = self._generate_text(prompt, max_new_tokens=10).lower()
                
                predicted = "true" in generated or "yes" in generated
                if predicted == answer:
                    correct += 1
                total += 1
                
            accuracy = correct / total if total > 0 else 0
            return {"accuracy": accuracy, "correct": correct, "total": total}
            
        except Exception as e:
            print(f"Error in boolq evaluation: {e}")
            return {"accuracy": 0.0, "error": str(e)}
            
    def _eval_piqa(self) -> Dict[str, float]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
            
            if self.config.limit:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
                
            correct = 0
            total = 0
            
            for example in dataset:
                goal = example["goal"]
                sol1 = example["sol1"]
                sol2 = example["sol2"]
                label = example["label"]
                
                scores = []
                for sol in [sol1, sol2]:
                    prompt = f"Goal: {goal}\nSolution: {sol}"
                    score = self._score_text(prompt)
                    scores.append(score)
                    
                predicted = scores.index(max(scores))
                if predicted == label:
                    correct += 1
                total += 1
                
            accuracy = correct / total if total > 0 else 0
            return {"accuracy": accuracy, "correct": correct, "total": total}
            
        except Exception as e:
            print(f"Error in piqa evaluation: {e}")
            return {"accuracy": 0.0, "error": str(e)}
            
    def _eval_siqa(self) -> Dict[str, float]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("social_i_qa", split="train", trust_remote_code=True)
            
            if self.config.limit:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
                
            correct = 0
            total = 0
            
            for example in dataset:
                context = example["context"]
                question = example["question"]
                answer_a = example["answerA"]
                answer_b = example["answerB"]
                answer_c = example["answerC"]
                label = int(example["label"]) - 1
                
                scores = []
                for answer in [answer_a, answer_b, answer_c]:
                    prompt = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
                    score = self._score_text(prompt)
                    scores.append(score)
                    
                predicted = scores.index(max(scores))
                if predicted == label:
                    correct += 1
                total += 1
                
            accuracy = correct / total if total > 0 else 0
            return {"accuracy": accuracy, "correct": correct, "total": total}
            
        except Exception as e:
            print(f"Error in siqa evaluation: {e}")
            return {"accuracy": 0.0, "error": str(e)}
            
    def _eval_openbookqa(self) -> Dict[str, float]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("openbookqa", "main", split="test", trust_remote_code=True)
            
            if self.config.limit:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
                
            correct = 0
            total = 0
            
            for example in dataset:
                question_stem = example["question_stem"]
                choices = example["choices"]
                answer_key = example["answerKey"]
                
                scores = []
                for choice_text in choices["text"]:
                    prompt = f"Question: {question_stem}\nAnswer: {choice_text}"
                    score = self._score_text(prompt)
                    scores.append(score)
                    
                predicted_idx = scores.index(max(scores))
                predicted_label = choices["label"][predicted_idx]
                
                if predicted_label == answer_key:
                    correct += 1
                total += 1
                
            accuracy = correct / total if total > 0 else 0
            return {"accuracy": accuracy, "correct": correct, "total": total}
            
        except Exception as e:
            print(f"Error in openbookqa evaluation: {e}")
            return {"accuracy": 0.0, "error": str(e)}
            
    def _eval_copa(self) -> Dict[str, float]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("super_glue", "copa", split="validation", trust_remote_code=True)
            
            if self.config.limit:
                dataset = dataset.select(range(min(self.config.limit, len(dataset))))
                
            correct = 0
            total = 0
            
            for example in dataset:
                premise = example["premise"]
                choice1 = example["choice1"]
                choice2 = example["choice2"]
                question = example["question"]
                label = example["label"]
                
                scores = []
                for choice in [choice1, choice2]:
                    if question == "cause":
                        prompt = f"{choice} because {premise}"
                    else:
                        prompt = f"{premise} so {choice}"
                    score = self._score_text(prompt)
                    scores.append(score)
                    
                predicted = scores.index(max(scores))
                if predicted == label:
                    correct += 1
                total += 1
                
            accuracy = correct / total if total > 0 else 0
            return {"accuracy": accuracy, "correct": correct, "total": total}
            
        except Exception as e:
            print(f"Error in copa evaluation: {e}")
            return {"accuracy": 0.0, "error": str(e)}
            
    def _score_text(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss.item()
            
        return -loss
        
    def _generate_text(self, prompt: str, max_new_tokens: int = 100) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()
            
        return generated
        
    def save_results(self, filepath: Optional[str] = None):
        if filepath is None:
            os.makedirs(self.config.output_path, exist_ok=True)
            filepath = os.path.join(self.config.output_path, "results.json")
            
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        print(f"Results saved to {filepath}")
        
    def print_summary(self):
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for task, result in self.results.items():
            if "accuracy" in result:
                print(f"{task}: {result['accuracy']:.4f}")
            else:
                print(f"{task}: {result}")
                
        print("="*50)
