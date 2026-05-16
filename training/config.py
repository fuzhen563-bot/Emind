from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainingConfig:
    mode: str = "sft"
    seed: int = 42
    device: str = "auto"
    output_dir: str = "checkpoints"
    experiment_name: str = "emind_exp"

    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_len: int = 2048
    dataloader_num_workers: int = 0
    pin_memory: bool = True

    learning_rate: float = 2e-5
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_steps: int = 200
    lr_scheduler: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    use_fp16: bool = False
    use_bf16: bool = True
    use_fsdp: bool = False
    fsdp_full_shard: bool = True
    activation_checkpointing: bool = True

    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 200
    eval_batch_size: Optional[int] = None
    logging_steps: int = 10
    early_stop_patience: int = 5

    resume_from: Optional[str] = None

    def __post_init__(self):
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size * 2

    @property
    def effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation_steps
