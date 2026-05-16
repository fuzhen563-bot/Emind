#!/usr/bin/env python3
"""
[DEPRECATED] This file is superseded by training/ package. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/ package instead.", DeprecationWarning, stacklevel=2)

"""
Emind 知识蒸馏脚本 — 使用 DistillationTrainer
从教师模型蒸馏知识到学生模型

用法:
    python 03_distillation.py --teacher checkpoints/teacher.pt --data data/train.txt
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EmindConfig, create_model
from tokenizer import EmindTokenizer
from training import DistillationTrainer, TrainingConfig, DistillationDataset


def main():
    parser = argparse.ArgumentParser(description="Emind 知识蒸馏")
    parser.add_argument("--teacher", type=str, required=True, help="教师模型路径")
    parser.add_argument("--data", type=str, default="data/train.txt")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--student-d-model", type=int, default=512)
    parser.add_argument("--student-n-heads", type=int, default=8)
    parser.add_argument("--student-n-kv-heads", type=int, default=4)
    parser.add_argument("--student-n-layers", type=int, default=6)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha-ce", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--use-fsdp", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    import torch
    tokenizer = EmindTokenizer(vocab_size=args.vocab_size)

    data = []
    if os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            data = [line.strip() for line in f if line.strip()]
    dataset = DistillationDataset(data, tokenizer, max_seq_len=args.max_seq_len)

    teacher_ckpt = torch.load(args.teacher, map_location="cpu", weights_only=False)
    teacher_cfg = EmindConfig.from_dict(teacher_ckpt.get("model_config", {}))
    teacher_model = create_model(teacher_cfg)

    student_cfg = EmindConfig(vocab_size=args.vocab_size, d_model=args.student_d_model,
                              n_heads=args.student_n_heads, n_kv_heads=args.student_n_kv_heads,
                              n_layers=args.student_n_layers, d_ff=args.student_d_model * 4,
                              max_seq_len=args.max_seq_len)
    student_model = create_model(student_cfg)

    train_cfg = TrainingConfig(
        mode="distill", epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
        output_dir=args.output_dir, max_seq_len=args.max_seq_len,
        use_bf16=True, use_fsdp=args.use_fsdp, device=args.device,
    )
    trainer = DistillationTrainer(
        student_model, teacher_model, train_cfg, dataset,
        temperature=args.temperature, alpha_ce=args.alpha_ce,
    )
    trainer.train()


if __name__ == "__main__":
    main()
