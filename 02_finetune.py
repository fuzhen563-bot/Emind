#!/usr/bin/env python3
"""
[DEPRECATED] This file is superseded by training/ package. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/ package instead.", DeprecationWarning, stacklevel=2)

"""
Emind 监督微调脚本 — 使用 SFTTrainer
在指令数据上进行监督微调

用法:
    python 02_finetune.py --data data/sft.json --epochs 3 --batch-size 4
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EmindConfig, create_model
from tokenizer import EmindTokenizer
from training import SFTTrainer, TrainingConfig, SFTDataset


def load_data(path: str):
    if not os.path.exists(path):
        return [{"prompt": "你好", "response": "你好！有什么可以帮助你的吗？"}] * 50
    with open(path, encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Emind 监督微调")
    parser.add_argument("--data", type=str, default="data/sft.json")
    parser.add_argument("--model-path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    tokenizer = EmindTokenizer(vocab_size=args.vocab_size)
    data = load_data(args.data)
    dataset = SFTDataset(data, tokenizer, max_seq_len=args.max_seq_len)

    if args.model_path and os.path.exists(args.model_path):
        import torch
        cfg = EmindConfig.from_dict(torch.load(args.model_path, map_location="cpu", weights_only=False).get("model_config", {}))
        model = create_model(cfg)
        model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=False)["model_state_dict"])
    else:
        cfg = EmindConfig(vocab_size=args.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
                          n_kv_heads=args.n_kv_heads, n_layers=args.n_layers, d_ff=args.d_ff * 4,
                          max_seq_len=args.max_seq_len)
        model = create_model(cfg)

    train_cfg = TrainingConfig(
        mode="sft", epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        output_dir=args.output_dir, max_seq_len=args.max_seq_len,
        use_bf16=True, use_fsdp=args.use_fsdp, device=args.device,
    )
    trainer = SFTTrainer(model, train_cfg, dataset)
    trainer.train()


if __name__ == "__main__":
    main()
