import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any, Callable, Union
from pathlib import Path

from model import EmindLM, EmindConfig, create_model
from training.config import TrainingConfig
from training.checkpoint import CheckpointManager
from training.metrics import MetricsTracker


class TrainerBase:
    def __init__(
        self,
        model: EmindLM,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = torch.device(config.device)

        self.model.to(self.device)
        if config.use_fsdp:
            self._wrap_fsdp()

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.checkpoint = CheckpointManager(config.output_dir, config.save_total_limit, config.experiment_name)
        self.metrics = MetricsTracker()
        self.global_step = 0
        self.epoch = 0

        self.scaler = torch.cuda.amp.GradScaler() if config.use_fp16 and self.device.type == "cuda" else None

    def _wrap_fsdp(self):
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from model import TransformerBlock

            wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls={TransformerBlock})
            self.model = FSDP(
                self.model,
                auto_wrap_policy=wrap_policy,
                mixed_precision=None,
                device_id=self.device if self.device.type == "cuda" else None,
            )
        except ImportError:
            print("FSDP not available, falling back to DDP")

    def _create_optimizer(self) -> AdamW:
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    def _create_scheduler(self) -> LambdaLR:
        effective_steps = len(self.train_dataset) // (self.config.batch_size * max(1, self.config.gradient_accumulation_steps))
        total_steps = max(1, effective_steps * self.config.epochs)

        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return float(step) / max(1, self.config.warmup_steps)
            progress = float(step - self.config.warmup_steps) / max(1, total_steps - self.config.warmup_steps)
            if self.config.lr_scheduler == "cosine":
                return max(self.config.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
            elif self.config.lr_scheduler == "linear":
                return max(self.config.min_lr_ratio, 1.0 - progress)
            return max(self.config.min_lr_ratio, 1.0 - progress)

        return LambdaLR(self.optimizer, lr_lambda)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
        )

    def eval_dataloader(self) -> Optional[DataLoader]:
        if self.eval_dataset is None:
            return None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
        )

    def compute_loss(self, batch: Any) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: Any) -> torch.Tensor:
        loss = self.compute_loss(batch)
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss.detach()

    def optimizer_step(self):
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def evaluate(self) -> Dict[str, float]:
        if not self.eval_dataset:
            return {}
        loader = self.eval_dataloader()
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in loader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        self.model.train()
        avg_loss = total_loss / max(1, num_batches)
        return {"eval_loss": avg_loss, "perplexity": math.exp(avg_loss) if avg_loss > 0 else float('inf')}

    def train(self, resume: bool = False):
        if resume:
            start_step, _ = self.checkpoint.load_latest(self.model, self.optimizer, self.scheduler)
            self.global_step = start_step

        train_loader = self.train_dataloader()
        total_batches = len(train_loader)
        best_eval_loss = float('inf')
        early_stop_counter = 0
        self.model.train()

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_start = time.time()

            for batch_idx, batch in enumerate(train_loader):
                step_loss = self.training_step(batch)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.metrics.log_step(step_loss.item() * self.config.gradient_accumulation_steps, current_lr, self.global_step)

                    if self.global_step % self.config.logging_steps == 0:
                        print(f"Epoch {epoch+1}/{self.config.epochs} | Step {self.global_step} | Loss: {step_loss.item() * self.config.gradient_accumulation_steps:.4f} | LR: {current_lr:.2e}")

                    if self.global_step % self.config.eval_steps == 0 and self.eval_dataset is not None:
                        eval_metrics = self.evaluate()
                        self.metrics.log_eval(eval_metrics.get("eval_loss", 0), eval_metrics.get("perplexity"))
                        print(f"  Eval loss: {eval_metrics.get('eval_loss', 0):.4f} | Perplexity: {eval_metrics.get('perplexity', 0):.2f}")

                        if eval_metrics.get("eval_loss", float('inf')) < best_eval_loss:
                            best_eval_loss = eval_metrics["eval_loss"]
                            early_stop_counter = 0
                            self._save(is_best=True)
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= self.config.early_stop_patience:
                                print(f"Early stopping triggered at step {self.global_step}")
                                return

                    if self.global_step % self.config.save_steps == 0:
                        self._save()

                epoch_loss += step_loss.item() * self.config.gradient_accumulation_steps

            avg_epoch_loss = epoch_loss / total_batches
            self.metrics.end_epoch(epoch, avg_epoch_loss)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s | Avg loss: {avg_epoch_loss:.4f}")

        self.metrics.save(str(Path(self.config.output_dir) / self.config.experiment_name / "metrics.json"))
        print(f"Training complete. {self.metrics.summary()}")

    def _save(self, is_best: bool = False):
        model_state = self.model.state_dict()
        if self.config.use_fsdp:
            model_state = self.model.state_dict()
        self.checkpoint.save(
            step=self.global_step,
            model_state=model_state,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            metrics={"train_loss": self.metrics.current_epoch.get("train_loss", 0)},
            is_best=is_best,
        )
