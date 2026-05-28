import os
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import time


@dataclass
class DashboardConfig:
    project_name: str = "legionheto"
    run_name: Optional[str] = None
    offline: bool = False
    log_model: bool = False
    log_freq: int = 10


class WandbDashboard:
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        **kwargs
    ):
        self.config = config or DashboardConfig()
        self.run = None
        self.step = 0
        self.history = []
        
        try:
            import wandb
            self.wandb = wandb
            self.available = True
        except ImportError:
            print("wandb not installed. Running in offline mode.")
            self.available = False
            self.config.offline = True
            
        self._init_run(**kwargs)
        
    def _init_run(self, **kwargs):
        if not self.available or self.config.offline:
            self.run = None
            return
            
        self.run = self.wandb.init(
            project=self.config.project_name,
            name=self.config.run_name,
            config=kwargs,
            reinit=True,
        )
        
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        current_step = step if step is not None else self.step
        
        metrics_with_step = {**metrics, "step": current_step}
        self.history.append(metrics_with_step)
        
        if self.run:
            self.run.log(metrics, step=current_step)
            
        self.step = current_step + 1
        
    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        epoch: Optional[int] = None,
    ):
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
        }
        
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm
            
        if epoch is not None:
            metrics["train/epoch"] = epoch
            
        self.log(metrics)
        
    def log_evaluation(self, metrics: Dict[str, float]):
        eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
        self.log(eval_metrics)
        
    def log_sample_generations(
        self,
        prompts: List[str],
        completions: List[str],
        num_samples: int = 5,
    ):
        samples = []
        for prompt, completion in zip(prompts[:num_samples], completions[:num_samples]):
            samples.append({"prompt": prompt, "completion": completion})
            
        if self.run:
            self.run.log({"samples": self.wandb.Table(
                columns=["prompt", "completion"],
                data=[[s["prompt"], s["completion"]] for s in samples]
            )})
            
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        if self.run:
            self.run.config.update(hparams)
            
    def log_model_checkpoint(self, checkpoint_path: str, aliases: Optional[List[str]] = None):
        if self.run and self.config.log_model:
            artifact = self.wandb.Artifact(
                name=f"model-{self.run.id}",
                type="model",
            )
            artifact.add_dir(checkpoint_path)
            self.run.log_artifact(artifact, aliases=aliases or ["latest"])
            
    def finish(self):
        if self.run:
            self.run.finish()
            
    def save_offline_logs(self, output_dir: str = "./wandb_offline"):
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, f"run_{int(time.time())}.json")
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
            
        print(f"Offline logs saved to {filepath}")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.config.offline:
            self.save_offline_logs()
        self.finish()


class ConsoleDashboard:
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.step = 0
        self.metrics_history = []
        
    def log(self, metrics: Dict[str, Any]):
        self.metrics_history.append({**metrics, "step": self.step})
        
        if self.step % self.log_interval == 0:
            metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                      for k, v in metrics.items()])
            print(f"[Step {self.step}] {metrics_str}")
            
        self.step += 1
        
    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        grad_norm: Optional[float] = None,
        epoch: Optional[int] = None,
    ):
        metrics = {"loss": loss, "lr": learning_rate}
        if grad_norm:
            metrics["grad_norm"] = grad_norm
        if epoch:
            metrics["epoch"] = epoch
        self.log(metrics)
        
    def plot_loss_curve(self, save_path: Optional[str] = None):
        try:
            import matplotlib.pyplot as plt
            
            steps = [m["step"] for m in self.metrics_history if "loss" in m or "train/loss" in m]
            losses = [m.get("loss", m.get("train/loss", 0)) for m in self.metrics_history 
                      if "loss" in m or "train/loss" in m]
            
            if not steps or not losses:
                print("No loss data to plot")
                return
                
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
                print(f"Loss curve saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not installed. Cannot plot loss curve.")
            
    def generate_summary(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {}
            
        losses = [m.get("loss", m.get("train/loss", 0)) for m in self.metrics_history 
                  if "loss" in m or "train/loss" in m]
        
        if not losses:
            return {}
            
        return {
            "total_steps": self.step,
            "final_loss": losses[-1],
            "min_loss": min(losses),
            "max_loss": max(losses),
            "avg_loss": sum(losses) / len(losses),
        }


def create_dashboard(
    use_wandb: bool = True,
    offline: bool = False,
    **kwargs
) -> Any:
    if use_wandb and not offline:
        try:
            return WandbDashboard(**kwargs)
        except Exception as e:
            print(f"Failed to initialize wandb: {e}. Falling back to console dashboard.")
            
    return ConsoleDashboard()
