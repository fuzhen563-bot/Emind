#!/usr/bin/env python3
"""
[DEPRECATED] This file is superseded by training/ package. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/ package instead.", DeprecationWarning, stacklevel=2)

"""
Emind 持续学习脚本 — 增量训练 + 数据增强
在已有模型基础上进行增量学习和持续优化

用法:
    python 06_post_learning.py --model-path checkpoints/best/model.pt --data data/new_data.json
"""
import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import EmindConfig, create_model
from tokenizer import EmindTokenizer
from training import SFTTrainer, TrainingConfig, SFTDataset


def augment_data(data: list, times: int = 3) -> list:
    templates = [
        "请解释：{text}",
        "关于{text}，你怎么看？",
        "详细说明{text}的原理",
        "用通俗的语言解释{text}",
    ]
    augmented = list(data)
    for item in data:
        text = item if isinstance(item, str) else item.get("prompt", item.get("text", ""))
        for _ in range(times):
            template = random.choice(templates)
            augmented.append(template.format(text=text))
    return augmented


def main():
    parser = argparse.ArgumentParser(description="Emind 持续学习 (Post-Learning)")
    parser.add_argument("--model-path", type=str, required=True, help="已有模型路径")
    parser.add_argument("--data", type=str, default="data/sft.json")
    parser.add_argument("--augment", action="store_true", help="启用数据增强")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    cfg = EmindConfig.from_dict(ckpt.get("model_config", {}))
    model = create_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded model {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    tokenizer = EmindTokenizer(vocab_size=cfg.vocab_size)
    data = []
    if os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            data = json.load(f) if args.data.endswith(".json") else [line.strip() for line in f if line.strip()]

    if args.augment:
        data = augment_data(data)
        print(f"Data augmented to {len(data)} samples")

    dataset = SFTDataset(data, tokenizer, max_seq_len=cfg.max_seq_len)

    train_cfg = TrainingConfig(
        mode="post_learning", epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, output_dir=args.output_dir, warmup_steps=50,
        max_seq_len=cfg.max_seq_len, use_bf16=True, device=str(device),
    )
    trainer = SFTTrainer(model, train_cfg, dataset)
    trainer.train()


if __name__ == "__main__":
    main()
