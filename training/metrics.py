import math
import json
from pathlib import Path
from typing import Dict, List, Optional


class MetricsTracker:
    """BUG-011 fix: Metrics tracker with optional WandB/TensorBoard backends."""

    def __init__(self, backend: str = "json", wandb_project: Optional[str] = None,
                 wandb_entity: Optional[str] = None, experiment_name: str = "emind_exp",
                 config: Optional[Dict] = None):
        self.history: List[Dict[str, float]] = []
        self.current_epoch: Dict[str, float] = {}
        self._step_losses: List[float] = []
        self.backend = backend
        self._wandb = None
        self._tb_writer = None

        if backend == "wandb":
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    project=wandb_project or "emind",
                    entity=wandb_entity,
                    name=experiment_name,
                    config=config or {},
                )
            except ImportError:
                print("[WARN] wandb not installed, falling back to json logging")
                self.backend = "json"
        elif backend == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
            except ImportError:
                print("[WARN] tensorboard not installed, falling back to json logging")
                self.backend = "json"

    def log_step(self, loss: float, lr: float, step: int):
        self._step_losses.append(loss)
        self.current_epoch["lr"] = lr
        self.current_epoch["step"] = step
        if self._wandb is not None:
            self._wandb.log({"train/loss": loss, "train/lr": lr}, step=step)
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("train/loss", loss, step)
            self._tb_writer.add_scalar("train/lr", lr, step)

    def log_eval(self, eval_loss: float, perplexity: Optional[float] = None):
        self.current_epoch["eval_loss"] = eval_loss
        if perplexity is not None:
            self.current_epoch["perplexity"] = perplexity
        elif eval_loss > 0:
            self.current_epoch["perplexity"] = math.exp(eval_loss)
        step = self.current_epoch.get("step", 0)
        if self._wandb is not None:
            self._wandb.log({
                "eval/loss": eval_loss,
                "eval/perplexity": self.current_epoch.get("perplexity", 0),
            }, step=step)
        if self._tb_writer is not None:
            self._tb_writer.add_scalar("eval/loss", eval_loss, step)
            if "perplexity" in self.current_epoch:
                self._tb_writer.add_scalar("eval/perplexity", self.current_epoch["perplexity"], step)

    def end_epoch(self, epoch: int, train_loss: Optional[float] = None):
        if train_loss is None and self._step_losses:
            train_loss = sum(self._step_losses) / len(self._step_losses)
        self.current_epoch["epoch"] = epoch
        self.current_epoch["train_loss"] = train_loss
        if train_loss is not None and train_loss > 0:
            self.current_epoch["train_perplexity"] = math.exp(train_loss)
        self.history.append(dict(self.current_epoch))
        self._step_losses = []
        self.current_epoch = {}

    def best_metric(self, metric: str = "eval_loss", minimize: bool = True) -> Optional[float]:
        if not self.history or metric not in self.history[0]:
            return None
        values = [h.get(metric, float('inf')) for h in self.history if metric in h]
        if not values:
            return None
        return min(values) if minimize else max(values)

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.history, f, indent=2)
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()

    def load(self, path: str):
        with open(path) as f:
            self.history = json.load(f)

    def summary(self) -> str:
        if not self.history:
            return "No training data"
        best_eval = self.best_metric("eval_loss")
        last = self.history[-1]
        return (
            f"Epochs: {len(self.history)}, "
            f"Best eval loss: {f'{best_eval:.4f}' if best_eval is not None else 'N/A'}, "
            f"Last train loss: {last.get('train_loss', 'N/A')}, "
            f"Last eval loss: {last.get('eval_loss', 'N/A')}"
        )
