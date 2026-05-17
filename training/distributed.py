import os
import torch
import torch.distributed as dist


def init_distributed():
    if not dist.is_available() or not dist.is_nccl_available():
        return None, 1
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
