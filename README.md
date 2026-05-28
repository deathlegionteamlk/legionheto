<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d0d0d,50:1a1a2e,100:16213e&height=200&section=header&text=LEGIONHETO&fontSize=80&fontColor=00ff88&animation=fadeIn&fontAlignY=38&desc=LLM%20Fine-Tuning%20Framework&descAlignY=60&descColor=888888" width="100%"/>

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=22&pause=1000&color=00FF88&center=true&vCenter=true&multiline=true&repeat=true&width=700&height=80&lines=Fine-tune+any+LLM.+Any+architecture.;SFT+%E2%80%A2+DPO+%E2%80%A2+ORPO+%E2%80%A2+RLOO+%E2%80%A2+KTO+%E2%80%A2+SimPO;Flash+Attention+2+%E2%80%A2+8-bit+AdamW+%E2%80%A2+GGUF+Export)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Version](https://img.shields.io/badge/version-0.2.0-00ff88?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-success?style=for-the-badge)

</div>

---

<div align="center">

## ⚡ What's this?

</div>

legionheto is a fine-tuning framework for large language models. Not a wrapper around another wrapper. Not "batteries included" in the way that means you can't touch anything. It's a proper toolkit — you pick what you need and run.

Seven trainers. Three attention backends. An 8-bit optimizer. GGUF and TensorRT-LLM export. Synthetic data generation. Built-in eval. Memory profiling. All in one repo, all in Python.

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDd4bnV6dGJ4OWF0dXhiY3lxd2tyZm1iczUwOHBzanJlOGFuMnB6eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/QpVUMRUJGokfqXyfa1/giphy.gif" width="600"/>

### 🚀 Quick Start

</div>

```python
from legionheto import LegionHetoModel, SFTTrainer

# Load any HuggingFace model
model = LegionHetoModel("meta-llama/Llama-2-7b-hf")

# Attach LoRA — r and alpha are yours to tune
model.setup_lora(r=16, alpha=32)

# Pick a trainer, hand it your dataset, go
trainer = SFTTrainer(model, dataset)
trainer.train()
```

That's it for the basics. Everything else is opt-in.

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYXJzbXdwOWZhMmdsMXY4M2x2dXB5NzB5NzFyMnBxMzlhYmdxbzI5MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o7abKhOpu0NwenH3O/giphy.gif" width="500"/>

## 🏗️ Project Structure

</div>

```
legionheto/
├── 🧠 attention/     Flash Attention 2 · DeepSeek MLA · Memory-Efficient Attention
├── 💻 cli/           Command-line interface
├── 📦 data/          Dataset loading · Streaming · Synthetic data generation
├── 📊 eval/          Evaluation harness (HellaSwag, MMLU, and more)
├── 📤 export/        GGUF · TensorRT-LLM · Model merging
├── 🔧 models/        Model loading with automatic architecture detection
├── ⚙️  optim/         8-bit AdamW optimizer
├── 🏋️  train/         SFT · DPO · ORPO · RLOO · SimPO · IPO · KTO
└── 🛠️  utils/         Logging · Checkpointing · Memory profiler · Dashboards
```

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNzJ3NXQ3Z3ExanU5N25qMng5NmV6d2ExeWl1dWJ0b2VxbjRldWJqNiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/du3J3cXyzhj75IOgvA/giphy.gif" width="480"/>

## 🎯 Trainers

</div>

Seven training methods. You probably don't need all seven — but it's good to have options when a paper drops and you want to test it immediately.

| Trainer | What it does |
|---|---|
| `SFTTrainer` | Supervised fine-tuning — the baseline everyone starts with |
| `DPOTrainer` | Direct Preference Optimization — teach the model what you prefer |
| `ORPOTrainer` | Odds Ratio Preference Optimization — no reference model needed |
| `RLOOTrainer` | REINFORCE Leave-One-Out — leaner RL-style training |
| `SimPOTrainer` | Simple Preference Optimization — no reference, no reward model |
| `IPOTrainer` | Identity Preference Optimization — fixes DPO's overfitting edge cases |
| `KTOTrainer` | Kahneman-Tversky Optimization — works with unpaired preference data |

```python
from legionheto import DPOTrainer, ORPOTrainer

# DPO — classic alignment training
trainer = DPOTrainer(model, dataset)
trainer.train()

# ORPO — skip the reference model entirely
trainer = ORPOTrainer(model, dataset)
trainer.train()
```

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExZzlqNGdqMXZ4cjF5dWZsdWI5MW1iNWhvcnk2dHY5M3ZlaG1kODNiNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT9IgzoKnwFNmISR8I/giphy.gif" width="450"/>

## ⚡ Attention Backends

</div>

Three options. FA2 if your hardware supports it. DeepSeek's MLA if you're working with that architecture. Memory-efficient attention as the fallback.

```python
from legionheto import FlashAttention, DeepSeekMLA, MemoryEfficientAttention

# Flash Attention 2 — faster on long sequences, needs CUDA
model.set_attention(FlashAttention())

# DeepSeek Multi-Head Latent Attention
model.set_attention(DeepSeekMLA())

# No FA2? This works everywhere
model.set_attention(MemoryEfficientAttention())
```

FA2 makes a real difference on long contexts — if you're training on sequences over 2k tokens and not using it, you're leaving speed on the table.

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNm9xc2Q5eTl1d3ZzcnVncmRyNHN1amhsajFpODB1MWVjNW0wcWZmeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/077i6AULCXc0FKTj9s/giphy.gif" width="420"/>

## 🔋 Optimizer

</div>

```python
from legionheto import AdamW8bit

optimizer = AdamW8bit(model.parameters(), lr=2e-4)
```

8-bit AdamW stores optimizer states in 8-bit instead of 32-bit. On a 7B model that's roughly 14GB of optimizer state down to ~3.5GB. If you're fighting VRAM limits, this is one of the first switches worth flipping.

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOHhqamJtOXE2eXd1NTRyYm92N2d1cnFsaG1kMXgxenh4OWIxeG12ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/26tn33aiTi1jkl6H6/giphy.gif" width="440"/>

## 📤 Export

</div>

Train in Python, deploy anywhere.

```python
from legionheto import export_to_gguf, export_to_tensorrt_llm, ModelMerger

# GGUF — runs in llama.cpp, Ollama, LM Studio
export_to_gguf(model, output_path="model.gguf")

# TensorRT-LLM — for production inference on NVIDIA hardware
export_to_tensorrt_llm(model, output_dir="./trt_engine")

# Bake LoRA adapters into the base weights permanently
merged = ModelMerger(model).merge()
merged.save("./merged_model")
```

The GGUF path is the most common one. Fine-tune in legionheto, export to GGUF, run locally in Ollama. Works well.

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaGNzaHlhanFldG5va3B1eGV3bTlnNGpiOHd6dDQxNnM4OWhmMTkyMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPEqDGUULpEU0aQ/giphy.gif" width="400"/>

## 🧬 Synthetic Data

</div>

No dataset? Generate one. Two methods — Self-Instruct for building from seed tasks, Evol-Instruct for making existing data harder.

```python
from legionheto import SelfInstructGenerator, EvolInstructGenerator

# Build instruction data from scratch
gen = SelfInstructGenerator(model)
dataset = gen.generate(seed_tasks, n=1000)

# Take existing data and push it harder
evol = EvolInstructGenerator(model)
evolved = evol.evolve(dataset)
```

Evol-Instruct is worth trying even on good datasets. The harder variants often surface weaknesses the original data misses.

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbWVnanVtOGpsMnI3NnBqaTR5Z2g0bnF3M283cG1peTR0a3BuMTVtZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/l46Cy1rHbQ92uuLXa/giphy.gif" width="440"/>

## 📊 Evaluation

</div>

```python
from legionheto import LegionHetoHarness, EvaluationConfig

config = EvaluationConfig(
    tasks=["hellaswag", "mmlu"],
    num_fewshot=5
)

harness = LegionHetoHarness(model, config)
results = harness.run()
```

Run standard benchmarks without leaving the framework. Useful for checking whether a training run actually helped.

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdXJvOXZleTd0NGt3OXg1eGhleGZkbjN4NGMzanhrOXQxNG9penl6NyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oKIPnAiaMCws8nOsE/giphy.gif" width="420"/>

## 🖥️ Monitoring

</div>

```python
from legionheto import TrainingMemoryProfiler, create_dashboard

# See exactly where your memory is going
with TrainingMemoryProfiler() as profiler:
    trainer.train()

profiler.report()

# WandB or console — your call
dashboard = create_dashboard("wandb", project="my-finetune")
# or
dashboard = create_dashboard("console")
```

The memory profiler is genuinely useful. GPU OOM errors are almost always avoidable once you can see what's actually allocating.

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNXZmeWI2cmVna2ZkNWpxaGVwcGp0ZW82ZGp6Y3dxNHA0NmxhNWJ1eiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ZVik7pIojeNzVDqUqK/giphy.gif" width="420"/>

## 📦 Installation

</div>

```bash
git clone https://github.com/deathlegionteamlk/legionheto.git
cd legionheto
pip install -e .
```

Editable install so local changes take effect immediately. No reinstall needed during development.

**Requirements:**

- Python 3.8+
- PyTorch
- CUDA GPU — needed for Flash Attention 2 and 8-bit optimizer. CPU works for testing but not for real training runs.

---

<div align="center">

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=16&pause=1500&color=00FF88&center=true&vCenter=true&width=600&lines=Built+by+Death+Legion+Team+LK+%F0%9F%87%B1%F0%9F%87%B0;Issues+%26+PRs+welcome;If+something%27s+broken%2C+open+an+issue)](https://git.io/typing-svg)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:16213e,50:1a1a2e,100:0d0d0d&height=120&section=footer&animation=fadeIn" width="100%"/>

**legionheto** · v0.2.0 · [deathlegionteamlk](https://github.com/deathlegionteamlk) · Python · MIT

</div>
