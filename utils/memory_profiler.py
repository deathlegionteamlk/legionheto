import time
import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import torch
import psutil


@dataclass
class MemorySnapshot:
    timestamp: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_peak_mb: float
    cpu_used_mb: float
    cpu_percent: float
    step: Optional[int] = None
    

class MemoryProfiler:
    def __init__(self, log_interval: int = 10, output_dir: str = "./memory_logs"):
        self.log_interval = log_interval
        self.output_dir = output_dir
        self.snapshots: List[MemorySnapshot] = []
        self.step = 0
        self.start_time = None
        
        os.makedirs(output_dir, exist_ok=True)
        
    def _get_gpu_memory(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "peak": 0}
            
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "peak": peak,
        }
        
    def _get_cpu_memory(self) -> Dict[str, float]:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        return {
            "used": memory_info.rss / 1024**2,
            "percent": cpu_percent,
        }
        
    def snapshot(self, step: Optional[int] = None) -> MemorySnapshot:
        gpu_mem = self._get_gpu_memory()
        cpu_mem = self._get_cpu_memory()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            gpu_allocated_mb=gpu_mem["allocated"],
            gpu_reserved_mb=gpu_mem["reserved"],
            gpu_peak_mb=gpu_mem["peak"],
            cpu_used_mb=cpu_mem["used"],
            cpu_percent=cpu_mem["percent"],
            step=step if step is not None else self.step,
        )
        
        self.snapshots.append(snapshot)
        self.step += 1
        
        return snapshot
        
    def log(self, message: str = ""):
        snapshot = self.snapshot()
        
        if self.step % self.log_interval == 0:
            print(f"[Step {snapshot.step}] GPU: {snapshot.gpu_allocated_mb:.1f}MB | "
                  f"CPU: {snapshot.cpu_used_mb:.1f}MB {message}")
                  
    @contextmanager
    def profile_block(self, name: str = "block"):
        start_snapshot = self.snapshot()
        print(f"[Profile] Starting {name}...")
        
        try:
            yield self
        finally:
            end_snapshot = self.snapshot()
            gpu_delta = end_snapshot.gpu_allocated_mb - start_snapshot.gpu_allocated_mb
            print(f"[Profile] {name} complete. GPU delta: {gpu_delta:.1f}MB")
            
    def get_peak_memory(self) -> Dict[str, float]:
        if not self.snapshots:
            return {"gpu": 0, "cpu": 0}
            
        peak_gpu = max(s.gpu_peak_mb for s in self.snapshots)
        peak_cpu = max(s.cpu_used_mb for s in self.snapshots)
        
        return {"gpu": peak_gpu, "cpu": peak_cpu}
        
    def get_timeline(self) -> List[Dict[str, Any]]:
        return [asdict(s) for s in self.snapshots]
        
    def save_timeline(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = os.path.join(self.output_dir, "memory_timeline.json")
            
        timeline = self.get_timeline()
        
        with open(filepath, 'w') as f:
            json.dump(timeline, f, indent=2)
            
        print(f"Memory timeline saved to {filepath}")
        
    def generate_report(self) -> str:
        if not self.snapshots:
            return "No memory snapshots recorded."
            
        peak = self.get_peak_memory()
        avg_gpu = sum(s.gpu_allocated_mb for s in self.snapshots) / len(self.snapshots)
        avg_cpu = sum(s.cpu_used_mb for s in self.snapshots) / len(self.snapshots)
        
        report = f"""
Memory Profile Report
====================
Total Steps: {len(self.snapshots)}

Peak Memory Usage:
  GPU: {peak['gpu']:.1f} MB
  CPU: {peak['cpu']:.1f} MB

Average Memory Usage:
  GPU: {avg_gpu:.1f} MB
  CPU: {avg_cpu:.1f} MB

Current Memory:
  GPU Allocated: {self.snapshots[-1].gpu_allocated_mb:.1f} MB
  GPU Reserved: {self.snapshots[-1].gpu_reserved_mb:.1f} MB
  CPU Used: {self.snapshots[-1].cpu_used_mb:.1f} MB
"""
        return report
        
    def save_report(self, filepath: Optional[str] = None):
        if filepath is None:
            filepath = os.path.join(self.output_dir, "memory_report.txt")
            
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            f.write(report)
            
        print(f"Memory report saved to {filepath}")
        
    def reset(self):
        self.snapshots = []
        self.step = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
    def __enter__(self):
        self.reset()
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"\nProfiling complete. Duration: {elapsed:.1f}s")
        print(self.generate_report())
        self.save_timeline()
        self.save_report()


class TrainingMemoryProfiler:
    def __init__(self, trainer, log_interval: int = 10):
        self.trainer = trainer
        self.profiler = MemoryProfiler(log_interval=log_interval)
        
    def on_step_end(self, step: int):
        self.profiler.snapshot(step)
        
    def on_epoch_end(self, epoch: int):
        peak = self.profiler.get_peak_memory()
        print(f"Epoch {epoch} - Peak GPU: {peak['gpu']:.1f}MB, Peak CPU: {peak['cpu']:.1f}MB")
        
    def on_train_end(self):
        self.profiler.save_timeline()
        self.profiler.save_report()
