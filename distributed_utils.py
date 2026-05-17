"""
Emind distributed training utilities — DDP + FSDP support.
"""
import os
import math
import random
import argparse
from typing import Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset

from model import EmindLM, TransformerBlock


def setup_distributed(rank: int, world_size: int, backend: str = "nccl"):
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_loss(loss: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(loss)
        return loss / dist.get_world_size()
    return loss


def set_seeds(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_distributed_model(
    model: EmindLM,
    device: torch.device,
    use_fsdp: bool = False,
    use_ddp: bool = False,
):
    if use_fsdp and torch.cuda.is_available():
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
                ShardingStrategy,
            )
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            bf16_available = torch.cuda.is_bf16_supported()
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16 if bf16_available else torch.float16,
                reduce_dtype=torch.bfloat16 if bf16_available else torch.float16,
                buffer_dtype=torch.bfloat16 if bf16_available else torch.float16,
            )

            model = FSDP(
                model,
                auto_wrap_policy=transformer_auto_wrap_policy(transformer_layer_cls={TransformerBlock}),
                mixed_precision=mp_policy,
                device_id=device,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                limit_all_gathers=True,
            )
            return model
        except ImportError:
            print("FSDP not available, falling back to DDP")
            use_ddp = True

    if use_ddp and torch.cuda.is_available():
        return DDP(model, device_ids=[device.index], find_unused_parameters=False)

    return model


def create_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def save_distributed_model(model: nn.Module, path: str):
    state = None
    if isinstance(model, (FSDP,)):
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig, StateDictType
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            if is_main_process():
                state = model.state_dict()
    elif hasattr(model, "module"):
        if is_main_process():
            state = model.module.state_dict()
    else:
        if is_main_process():
            state = model.state_dict()
    if state is not None:
        torch.save(state, path)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emind distributed training")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank")
    parser.add_argument("--use-fsdp", action="store_true", help="Use FSDP")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory")
    return parser
