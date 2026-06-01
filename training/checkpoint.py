import os
import json
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from collections import OrderedDict


class CheckpointManager:
    def __init__(self, output_dir: str, save_total_limit: int = 3, experiment_name: str = "emind"):
        self.output_dir = Path(output_dir) / experiment_name
        self.save_total_limit = save_total_limit
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, step: int) -> Path:
        return self.output_dir / f"step_{step}"

    def _best_path(self) -> Path:
        return self.output_dir / "best"

    def _latest_path(self) -> Path:
        return self.output_dir / "latest"

    def save(
        self,
        step: int,
        model_state: Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        ckpt_path = self._checkpoint_path(step)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "step": step,
            "model_state_dict": model_state,
            "model_config": model_config or {},
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "metrics": metrics or {},
        }
        torch.save(save_dict, ckpt_path / "model.pt")
        if metrics:
            with open(ckpt_path / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        latest = self._latest_path()
        latest.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, latest / "model.pt")

        if is_best:
            best = self._best_path()
            best.mkdir(parents=True, exist_ok=True)
            torch.save(save_dict, best / "model.pt")

        self._enforce_limit()

    def _enforce_limit(self):
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        while len(checkpoints) > self.save_total_limit:
            import shutil
            shutil.rmtree(checkpoints[0])
            checkpoints = checkpoints[1:]

    def load_latest(self, model: torch.nn.Module, optimizer=None, scheduler=None):
        ckpt = self._latest_path() / "model.pt"
        if not ckpt.exists():
            return 0, {}
        return self._load(ckpt, model, optimizer, scheduler)

    def load_best(self, model: torch.nn.Module, optimizer=None, scheduler=None):
        ckpt = self._best_path() / "model.pt"
        if not ckpt.exists():
            return 0, {}
        return self._load(ckpt, model, optimizer, scheduler)

    def load(self, step: int, model: torch.nn.Module, optimizer=None, scheduler=None):
        ckpt = self._checkpoint_path(step) / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint step_{step} not found")
        return self._load(ckpt, model, optimizer, scheduler)

    def _load(self, ckpt_path: Path, model, optimizer=None, scheduler=None):
        device = next(model.parameters()).device
        save_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = save_dict["model_state_dict"]
        # 处理 DDP/FSDP 的 module. 前缀
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        if optimizer and save_dict.get("optimizer_state_dict"):
            optimizer.load_state_dict(save_dict["optimizer_state_dict"])
        if scheduler and save_dict.get("scheduler_state_dict"):
            scheduler.load_state_dict(save_dict["scheduler_state_dict"])
        return save_dict.get("step", 0), save_dict.get("metrics", {})

    def get_history(self) -> list:
        history = []
        for d in sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        ):
            metrics_file = d / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    history.append(json.load(f))
        return history
