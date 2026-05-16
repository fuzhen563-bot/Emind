#!/usr/bin/env python3
"""
[DEPRECATED] This file is superseded by distributed_utils.py. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use distributed_utils.py instead.", DeprecationWarning, stacklevel=2)

"""
Emind 分布式训练启动器
支持所有训练模式的分布式启动
"""

import torch
import os
import sys

from distributed_utils import (
    setup_distributed, cleanup_distributed, get_local_rank, get_world_size,
    is_main_process, distributed_log, count_gpus, auto_world_size
)


def get_train_script(train_type: str) -> str:
    """获取训练脚本路径"""
    scripts = {
        "pretrain": "01_pretrain.py",
        "finetune": "02_finetune.py",
        "distillation": "03_distillation.py",
        "lora": "04_lora.py",
        "inference": "05_inference.py",
        "post_learning": "06_post_learning.py",
        "feedback": "07_feedback_training.py"
    }
    return scripts.get(train_type, "01_pretrain.py")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Emind 分布式训练启动器")

    parser.add_argument("--train_type", type=str, default="pretrain",
                        choices=["pretrain", "finetune", "distillation", "lora",
                                 "inference", "post_learning", "feedback"],
                        help="训练类型")

    parser.add_argument("--world_size", type=int, default=None,
                        help="总进程数（默认: GPU数量）")

    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小（每张卡）")

    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="学习率")

    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="使用混合精度训练")

    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    parser.add_argument("--backend", type=str, default="nccl",
                        help="分布式后端 (nccl/gloo)")

    args = parser.parse_args()

    if args.world_size is None:
        args.world_size = count_gpus()

    rank = 0
    world_size = args.world_size

    if world_size > 1:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        setup_distributed(rank, world_size, args.backend)

    distributed_log(rank, "=" * 60)
    distributed_log(rank, "Emind 分布式训练启动器")
    distributed_log(rank, "=" * 60)
    distributed_log(rank, f"训练类型: {args.train_type}")
    distributed_log(rank, f"进程数: {world_size}")
    distributed_log(rank, f"设备: CUDA" if torch.cuda.is_available() else "设备: CPU")
    distributed_log(rank, "=" * 60)

    train_script = get_train_script(args.train_type)
    script_path = os.path.join(os.path.dirname(__file__), train_script)

    if not os.path.exists(script_path):
        distributed_log(rank, f"错误: 训练脚本不存在: {script_path}")
        sys.exit(1)

    env_vars = {
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": str(rank),
        "RANK": str(rank),
    }

    cmd = [
        sys.executable,
        script_path,
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--learning_rate={args.learning_rate}",
    ]

    if args.use_amp:
        cmd.append("--use_amp")

    if world_size > 1:
        cmd.extend(["--local_rank", str(rank), "--world_size", str(world_size)])

    distributed_log(rank, f"启动命令: {' '.join(cmd)}")

    os.execve(sys.executable, cmd + sys.argv[1:], env_vars)


if __name__ == "__main__":
    main()