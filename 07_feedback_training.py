#!/usr/bin/env python3
"""
[DEPRECATED] This file is superseded by training/ package. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/ package instead.", DeprecationWarning, stacklevel=2)

"""
Emind 反馈训练脚本 — 基于人类偏好对齐 (DPO)
使用 DPO 算法优化模型以对齐人类偏好

用法:
    python 07_feedback_training.py --model-path checkpoints/best/model.pt --data data/dpo.json
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import EmindConfig, create_model
from tokenizer import EmindTokenizer
from training import DPOTrainer, TrainingConfig, DPODataset


def main():
    parser = argparse.ArgumentParser(description="Emind DPO 偏好对齐")
    parser.add_argument("--model-path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--ref-model-path", type=str, default=None, help="参考模型路径 (默认使用基础模型)")
    parser.add_argument("--data", type=str, default="data/dpo.json")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta 参数")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    cfg = EmindConfig.from_dict(ckpt.get("model_config", {}))
    model = create_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    ref_model = None
    if args.ref_model_path and os.path.exists(args.ref_model_path):
        ref_ckpt = torch.load(args.ref_model_path, map_location=device, weights_only=False)
        ref_model = create_model(cfg)
        ref_model.load_state_dict(ref_ckpt["model_state_dict"], strict=False)
        print("Reference model loaded")

    tokenizer = EmindTokenizer(vocab_size=cfg.vocab_size)

    data = []
    if os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = [{"prompt": "写一首诗", "chosen": "春风又绿江南岸", "rejected": "不知道"}]

    dataset = DPODataset(data, tokenizer, max_seq_len=cfg.max_seq_len)

    train_cfg = TrainingConfig(
        mode="dpo", epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, output_dir=args.output_dir, max_seq_len=cfg.max_seq_len,
        use_bf16=True, device=str(device),
    )
    trainer = DPOTrainer(model, ref_model, train_cfg, dataset, beta=args.beta)
    trainer.train()


if __name__ == "__main__":
    main()
