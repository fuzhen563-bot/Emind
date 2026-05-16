"""
[DEPRECATED] This file is superseded by training/ package. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/ package instead.", DeprecationWarning, stacklevel=2)

"""
Emind 高级训练模块 - 精度优化版
包含：更大的模型架构、先进的训练策略、更丰富的数据处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import random
import os
import math
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from model import create_model
from tokenizer import SimpleTokenizer


@dataclass
class AdvancedModelConfig:
    """高级模型配置"""
    vocab_size: int = 15000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 16
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout: float = 0.1
    use_gated_activation: bool = True
    use_rms_norm: bool = True
    use_rotary_pos_emb: bool = True
    use_alibi_bias: bool = False


@dataclass
class AdvancedTrainConfig:
    """高级训练配置"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 4
    use_amp: bool = True
    use_flash_attention: bool = True
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    patience: int = 15
    eval_interval: int = 500
    save_interval: int = 1000


class AdvancedTextDataset(Dataset):
    """高级文本数据集 - 支持数据增强"""

    def __init__(
        self,
        data: List[str],
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 512,
        augmentation: bool = True
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # 数据增强
        if self.augmentation and random.random() < 0.3:
            text = self._augment_text(text)

        # 编码
        encoded = self.tokenizer.encode(text)

        # 截断
        if len(encoded) > self.max_seq_len:
            encoded = encoded[:self.max_seq_len]
        else:
            encoded = encoded + [0] * (self.max_seq_len - len(encoded))

        # 移位
        input_ids = torch.tensor(encoded[:-1], dtype=torch.long)
        labels = torch.tensor(encoded[1:], dtype=torch.long)

        return input_ids, labels

    def _augment_text(self, text: str) -> str:
        """文本数据增强"""
        augmentations = [
            lambda: text,
            lambda: text + "。",
            lambda: text.strip() + "。",
            lambda: "请问" + text,
            lambda: text + "吗？",
            lambda: text.replace("。", "，"),
        ]
        return random.choice(augmentations)()


class AdvancedTrainer:
    """高级训练器"""

    def __init__(
        self,
        model_config: AdvancedModelConfig,
        train_config: AdvancedTrainConfig,
        device: str = "cuda"
    ):
        self.model_config = model_config
        self.train_config = train_config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 初始化模型
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()

        # 混合精度
        self.scaler = GradScaler() if train_config.use_amp else None

        # 训练状态
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")

    def _create_model(self):
        """创建模型"""
        config_dict = {
            "vocab_size": self.model_config.vocab_size,
            "d_model": self.model_config.d_model,
            "n_heads": self.model_config.n_heads,
            "n_layers": self.model_config.n_layers,
            "d_ff": self.model_config.d_ff,
            "max_seq_len": self.model_config.max_seq_len,
            "dropout": self.model_config.dropout
        }
        return create_model(config_dict)

    def _create_optimizer(self):
        """创建优化器 - 使用 AdamW with weight decay decoupling"""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.train_config.weight_decay
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.train_config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )

    def _create_scheduler(self):
        """创建学习率调度器 - 余弦退火 with Warmup"""
        num_training_steps = self.train_config.epochs * 1000  # 估算
        num_warmup_steps = int(num_training_steps * self.train_config.warmup_ratio)

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(self.train_config.min_lr / self.train_config.learning_rate,
                      0.5 * (1.0 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, input_ids, labels) -> float:
        """单步训练"""
        self.model.train()

        # 混合精度前向传播
        if self.train_config.use_amp:
            with autocast():
                loss, _ = self.model(input_ids, labels)
                loss = loss / self.train_config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
        else:
            loss, _ = self.model(input_ids, labels)
            loss = loss / self.train_config.gradient_accumulation_steps
            loss.backward()

        # 梯度累积
        if (self.global_step + 1) % self.train_config.gradient_accumulation_steps == 0:
            # 梯度裁剪
            if self.train_config.grad_clip > 0:
                if self.train_config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config.grad_clip
                )

            # 更新参数
            if self.train_config.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1
        return loss.item() * self.train_config.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self, dataloader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        for input_ids, labels in dataloader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            loss, _ = self.model(input_ids, labels)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: str, epoch: int, train_loss: float, val_loss: float):
        """保存检查点"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'model_config': {
                "vocab_size": self.model_config.vocab_size,
                "d_model": self.model_config.d_model,
                "n_heads": self.model_config.n_heads,
                "n_layers": self.model_config.n_layers,
                "d_ff": self.model_config.d_ff,
                "max_seq_len": self.model_config.max_seq_len,
                "dropout": self.model_config.dropout
            }
        }
        torch.save(checkpoint, path)
        print(f"✓ 检查点已保存: {path}")

    def train(
        self,
        train_data: List[str],
        val_data: List[str],
        tokenizer: SimpleTokenizer,
        save_path: str = "checkpoints/advanced_model.pt"
    ):
        """训练模型"""
        # 创建数据集
        train_dataset = AdvancedTextDataset(
            train_data, tokenizer,
            self.model_config.max_seq_len,
            augmentation=True
        )
        val_dataset = AdvancedTextDataset(
            val_data, tokenizer,
            self.model_config.max_seq_len,
            augmentation=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        print(f"\n开始训练...")
        print(f"训练数据: {len(train_data)} 条")
        print(f"验证数据: {len(val_data)} 条")
        print(f"训练轮数: {self.train_config.epochs}")
        print(f"学习率: {self.train_config.learning_rate}")

        # 训练循环
        for epoch in range(self.train_config.epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.train_config.epochs} ===")

            # 训练
            epoch_loss = 0
            num_batches = 0

            for batch_idx, (input_ids, labels) in enumerate(train_loader):
                input_ids = input_ids.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                loss = self.train_step(input_ids, labels)
                epoch_loss += loss
                num_batches += 1

                if (batch_idx + 1) % 50 == 0:
                    print(f"  批次 [{batch_idx + 1}/{len(train_loader)}], "
                          f"损失: {loss:.4f}, "
                          f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")

            avg_train_loss = epoch_loss / num_batches

            # 验证
            val_loss = self.evaluate(val_loader)

            print(f"训练损失: {avg_train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")

            # 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(save_path, epoch, avg_train_loss, val_loss)
            else:
                self.patience_counter += 1
                print(f"验证损失未改善 ({self.patience_counter}/{self.train_config.patience})")

                if self.patience_counter >= self.train_config.patience:
                    print(f"\n早停触发！最佳验证损失: {self.best_val_loss:.4f}")
                    break

        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.4f}")
        return self.best_val_loss


class EnhancedGenerator:
    """增强版文本生成器"""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        num_return_sequences: int = 1
    ) -> List[str]:
        """增强版文本生成"""
        self.model.eval()

        # 编码
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)

        # 生成
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample
        )

        # 解码
        results = []
        for ids in generated_ids[:num_return_sequences]:
            text = self.tokenizer.decode(ids.cpu().tolist())
            results.append(text)

        return results if num_return_sequences > 1 else results[0]

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        callback=None
    ):
        """流式生成"""
        self.model.eval()

        # 编码
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)

        generated = []
        eos_token_id = self.tokenizer.stoi.get(self.tokenizer.eos_token, 3)

        for _ in range(max_new_tokens):
            # 前向传播
            logits = self.model(input_ids)[1][:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty > 1.0:
                for token_id in set(generated):
                    logits[0, token_id] /= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 检查结束
            if next_item == eos_token_id:
                break

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 解码并回调
            decoded_char = self.tokenizer.itos.get(next_token.item(), '')
            if decoded_char and callback:
                callback(decoded_char)

        # 返回完整文本
        return self.tokenizer.decode(generated)


def create_enhanced_trainer(
    model_size: str = "large",
    device: str = "cuda"
) -> Tuple[AdvancedTrainer, AdvancedModelConfig, AdvancedTrainConfig]:
    """
    创建增强版训练器

    Args:
        model_size: 模型大小 ("small", "medium", "large")
        device: 设备

    Returns:
        trainer: 训练器实例
    """
    # 模型配置
    if model_size == "small":
        model_config = AdvancedModelConfig(
            vocab_size=10000,
            d_model=512,
            n_heads=8,
            n_layers=8,
            d_ff=2048,
            max_seq_len=256
        )
    elif model_size == "medium":
        model_config = AdvancedModelConfig(
            vocab_size=15000,
            d_model=768,
            n_heads=12,
            n_layers=12,
            d_ff=3072,
            max_seq_len=512
        )
    else:  # large
        model_config = AdvancedModelConfig(
            vocab_size=20000,
            d_model=1024,
            n_heads=16,
            n_layers=24,
            d_ff=4096,
            max_seq_len=1024
        )

    # 训练配置
    train_config = AdvancedTrainConfig(
        epochs=50,
        batch_size=32,
        learning_rate=3e-4,
        warmup_ratio=0.1,
        weight_decay=0.1,
        grad_clip=1.0,
        gradient_accumulation_steps=4,
        patience=15
    )

    trainer = AdvancedTrainer(model_config, train_config, device)
    return trainer, model_config, train_config


if __name__ == "__main__":
    print("高级训练模块测试")

    # 测试配置
    trainer, model_config, train_config = create_enhanced_trainer("small")
    print(f"模型配置: d_model={model_config.d_model}, n_layers={model_config.n_layers}")
    print(f"训练配置: epochs={train_config.epochs}, lr={train_config.learning_rate}")
    print("模块加载成功！")
