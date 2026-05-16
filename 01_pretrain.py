#!/usr/bin/env python3
"""
[DEPRECATED] This file is superseded by training/ package. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/ package instead.", DeprecationWarning, stacklevel=2)

"""
Emind 预训练脚本 — 使用 unified TrainerBase
在大规模无标注数据上进行语言模型预训练

用法:
    python 01_pretrain.py --data data/train.txt --epochs 5 --batch-size 8
    torchrun --nproc_per_node=4 01_pretrain.py --use-fsdp
"""
import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import EmindConfig, create_model
from tokenizer import EmindTokenizer
from training import SFTTrainer, TrainingConfig, SFTDataset


def main():
    parser = argparse.ArgumentParser(description="Emind 预训练")
    parser.add_argument("--data", type=str, default="data/train.txt", help="训练数据路径")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    tokenizer = EmindTokenizer(vocab_size=args.vocab_size)

    data = []
    if os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            data = [line.strip() for line in f if line.strip()]
    if not data:
        data = ["深度学习是人工智能的核心技术。神经网络模型在自然语言处理中应用广泛。",
                "Transformer 架构改变了序列建模的方式。注意力机制让模型关注重要信息。"]

    dataset = SFTDataset(data, tokenizer, max_seq_len=args.max_seq_len)

    cfg = EmindConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=0.0,
    )
    model = create_model(cfg)

    train_cfg = TrainingConfig(
        mode="pretrain",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        use_bf16=True,
        use_fsdp=args.use_fsdp,
        device=args.device,
    )
    trainer = SFTTrainer(model, train_cfg, dataset)
    trainer.train()


if __name__ == "__main__":
    main()
