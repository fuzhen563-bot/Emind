#!/usr/bin/env python3
"""
[DEPRECATED] This file is superseded by training/ package. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/ package instead.", DeprecationWarning, stacklevel=2)

"""
Emind LoRA 微调脚本 — 高效参数微调
使用 LoRA 低秩适配，大幅减少可训练参数量

用法:
    python 04_lora.py --data data/sft.json --rank 8
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EmindConfig, create_model
from tokenizer import EmindTokenizer
from training import SFTTrainer, TrainingConfig, SFTDataset, apply_lora


def main():
    parser = argparse.ArgumentParser(description="Emind LoRA 微调")
    parser.add_argument("--data", type=str, default="data/sft.json")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    tokenizer = EmindTokenizer(vocab_size=args.vocab_size)
    data = []
    if os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            data = json.load(f) if args.data.endswith(".json") else [line.strip() for line in f if line.strip()]
    dataset = SFTDataset(data, tokenizer, max_seq_len=args.max_seq_len)

    cfg = EmindConfig(vocab_size=args.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                      n_kv_heads=args.n_heads // 4, n_layers=args.n_layers, d_ff=args.d_model * 4,
                      max_seq_len=args.max_seq_len)
    model = create_model(cfg)

    if args.model_path and os.path.exists(args.model_path):
        import torch
        model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=False))

    apply_lora(model, rank=args.rank, alpha=args.alpha)

    train_cfg = TrainingConfig(
        mode="lora", epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        output_dir=args.output_dir, max_seq_len=args.max_seq_len, use_bf16=True, device=args.device,
    )
    trainer = SFTTrainer(model, train_cfg, dataset)
    trainer.train()

    from training import merge_lora
    merge_lora(model)
    print("LoRA weights merged into base model.")


if __name__ == "__main__":
    main()
