import os
import math
import time
import torch
from contextlib import nullcontext
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
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

        self.rank = 0
        self.world_size = 1
        if config.use_fsdp:
            from training.distributed import init_distributed
            result = init_distributed()
            if result[0] is not None:
                self.rank, self.world_size = result

        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if config.activation_checkpointing:
            self.model.config.activation_checkpointing = True
        if config.use_fsdp:
            self._wrap_fsdp()
        if config.compile_model and hasattr(torch, 'compile') and not config.use_fsdp:
            self.model = torch.compile(self.model, mode=config.compile_mode)
        self.model.neftune_alpha = config.neftune_noise_alpha

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.checkpoint = CheckpointManager(config.output_dir, config.save_total_limit, config.experiment_name)
        # BUG-011 fix: pass logging backend config to MetricsTracker
        self.metrics = MetricsTracker(
            backend=config.log_backend,
            wandb_project=config.wandb_project,
            wandb_entity=config.wandb_entity,
            experiment_name=config.experiment_name,
        )
        self.global_step = 0
        self.epoch = 0

        use_grad_scaler = config.use_fp16 and self.device.type == "cuda" and not config.use_fsdp
        self.scaler = torch.cuda.amp.GradScaler() if use_grad_scaler else None

    def _wrap_fsdp(self):
        try:
            from functools import partial
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from torch.distributed.fsdp.api import MixedPrecision
            from model import TransformerBlock

            wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})

            mp_policy = None
            if self.config.use_bf16:
                mp_policy = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )

            self.model = FSDP(
                self.model,
                auto_wrap_policy=wrap_policy,
                mixed_precision=mp_policy,
                device_id=torch.cuda.current_device(),
            )
        except ImportError as e:
            print(f"FSDP not available ({e}), falling back to DDP")

    def _create_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optimizer == "adafactor":
            from torch.optim import Adafactor
            return Adafactor(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay,
                             scale_parameter=False, relative_step=False, warmup_init=False)
        fused = self.config.use_fused_adam and torch.cuda.is_available()
        return AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            fused=fused,
        )

    def _create_scheduler(self) -> LambdaLR:
        if hasattr(self.train_dataset, '_data'):
            total_samples = len(self.train_dataset._data)
        else:
            total_samples = len(self.train_dataset)
        effective_steps = total_samples // (self.config.batch_size * self.world_size * max(1, self.config.gradient_accumulation_steps))
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
        sampler = None
        shuffle = True
        if self.config.use_fsdp and self.world_size > 1:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
        )

    def eval_dataloader(self) -> Optional[DataLoader]:
        if self.eval_dataset is None:
            return None
        sampler = None
        if self.config.use_fsdp and self.world_size > 1:
            sampler = DistributedSampler(self.eval_dataset, shuffle=False)
        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory,
        )

    def compute_loss(self, batch: Any) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: Any) -> torch.Tensor:
        # BUG-007 fix: BF16 autocast for non-FSDP mode
        if self.config.use_bf16 and not self.config.use_fsdp and self.device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = self.compute_loss(batch)
        else:
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
            # BUG-010 fix: FSDP requires model.clip_grad_norm_() instead of torch.nn.utils
            if self.config.use_fsdp and hasattr(self.model, 'clip_grad_norm_'):
                self.model.clip_grad_norm_(self.config.max_grad_norm)
            else:
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
        # BUG-007 fix: BF16 autocast for non-FSDP eval
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) \
            if self.config.use_bf16 and not self.config.use_fsdp and self.device.type == "cuda" \
            else nullcontext()
        with torch.no_grad(), ctx:
            for batch in loader:
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        # BUG-014 fix: aggregate eval loss across distributed ranks
        if self.config.use_fsdp and self.world_size > 1:
            import torch.distributed as dist
            loss_tensor = torch.tensor([total_loss, num_batches], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = loss_tensor[0].item()
            num_batches = int(loss_tensor[1].item())
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
        is_main = self.rank == 0

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_start = time.time()

            if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(train_loader):
                # BUG-004 fix: NEFTune noise only on first gradient accumulation step
                if self.config.neftune_noise_alpha > 0:
                    if batch_idx % self.config.gradient_accumulation_steps == 0:
                        self.model.neftune_alpha = self.config.neftune_noise_alpha
                    else:
                        self.model.neftune_alpha = 0.0

                step_loss = self.training_step(batch)

                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    self.global_step += 1
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.metrics.log_step(step_loss.item() * self.config.gradient_accumulation_steps, current_lr, self.global_step)

                    if is_main and self.global_step % self.config.logging_steps == 0:
                        print(f"Epoch {epoch+1}/{self.config.epochs} | Step {self.global_step} | Loss: {step_loss.item() * self.config.gradient_accumulation_steps:.4f} | LR: {current_lr:.2e}")

                    if is_main and self.global_step % self.config.eval_steps == 0 and self.eval_dataset is not None:
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

                    if is_main and self.global_step % self.config.save_steps == 0:
                        self._save()

                epoch_loss += step_loss.item() * self.config.gradient_accumulation_steps

            if is_main:
                avg_epoch_loss = epoch_loss / total_batches
                self.metrics.end_epoch(epoch, avg_epoch_loss)
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s | Avg loss: {avg_epoch_loss:.4f}")

            if hasattr(self.train_dataset, 'advance_stage') and (epoch + 1) % max(1, self.config.epochs // self.config.curriculum_stages) == 0:
                if self.train_dataset.advance_stage():
                    if is_main:
                        start, end = self.train_dataset.get_stage_data()
                        print(f"  >>> Curriculum: stage {self.train_dataset.current_stage}/{self.train_dataset.num_stages} (samples {start}-{end})")
                    train_loader = self.train_dataloader()
                    total_batches = len(train_loader)

        if is_main:
            self.metrics.save(str(Path(self.config.output_dir) / self.config.experiment_name / "metrics.json"))
            self._save()
            print(f"Training complete. {self.metrics.summary()}")

    def _save(self, is_best: bool = False):
        model_state = None
        optimizer_state = None
        if self.config.use_fsdp:
            # BUG-013 fix: proper FSDP checkpoint saving with full state dict
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType, FullOptimStateDictConfig
            # Save full model state dict (rank0_only, offload to CPU to save GPU memory)
            model_cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
            optim_cfg = FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, model_cfg, optim_cfg):
                model_state = self.model.state_dict()
                optimizer_state = self.optimizer.state_dict()
            if self.rank != 0:
                return
        else:
            model_state = self.model.state_dict()
            optimizer_state = self.optimizer.state_dict()

        import dataclasses
        model_cfg = dataclasses.asdict(self.model.config) if hasattr(self.model, 'config') else {}
        self.checkpoint.save(
            step=self.global_step,
            model_state=model_state,
            model_config=model_cfg,
            optimizer_state=optimizer_state,
            scheduler_state=self.scheduler.state_dict(),
            metrics={"train_loss": self.metrics.current_epoch.get("train_loss", 0)},
            is_best=is_best,
        )
