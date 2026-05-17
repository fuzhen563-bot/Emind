import math
import json
from pathlib import Path
from typing import Dict, List, Optional


class MetricsTracker:
    def __init__(self):
        self.history: List[Dict[str, float]] = []
        self.current_epoch: Dict[str, float] = {}
        self._step_losses: List[float] = []

    def log_step(self, loss: float, lr: float, step: int):
        self._step_losses.append(loss)
        self.current_epoch["lr"] = lr
        self.current_epoch["step"] = step

    def log_eval(self, eval_loss: float, perplexity: Optional[float] = None):
        self.current_epoch["eval_loss"] = eval_loss
        if perplexity is not None:
            self.current_epoch["perplexity"] = perplexity
        elif eval_loss > 0:
            self.current_epoch["perplexity"] = math.exp(eval_loss)

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
