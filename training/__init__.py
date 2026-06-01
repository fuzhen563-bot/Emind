# Lazy-load torch-dependent modules on first use
from training.distillation_pipeline import DistillationPipeline, DistillationConfig, distill_and_train


def __getattr__(name):
    import importlib
    _LAZY = {
        "TrainingConfig": ("training.config", "TrainingConfig"),
        "MetricsTracker": ("training.metrics", "MetricsTracker"),
        "TrainerBase": ("training.trainer", "TrainerBase"),
        "SFTTrainer": ("training.sft", "SFTTrainer"),
        "SFTDataset": ("training.sft", "SFTDataset"),
        "PretrainTrainer": ("training.pretrain", "PretrainTrainer"),
        "PretrainDataset": ("training.pretrain", "PretrainDataset"),
        "DPOTrainer": ("training.dpo", "DPOTrainer"),
        "DPODataset": ("training.dpo", "DPODataset"),
        "DistillationTrainer": ("training.distill", "DistillationTrainer"),
        "DistillationDataset": ("training.distill", "DistillationDataset"),
        "CurriculumDataset": ("training.curriculum", "CurriculumDataset"),
        "StageScheduler": ("training.curriculum", "StageScheduler"),
        "apply_lora": ("training.lora", "apply_lora"),
        "merge_lora": ("training.lora", "merge_lora"),
        "lora_state_dict": ("training.lora", "lora_state_dict"),
        "LoRALayer": ("training.lora", "LoRALayer"),
        "CheckpointManager": ("training.checkpoint", "CheckpointManager"),
        "PPOConfig": ("training.rl", "PPOConfig"),
        "GRPOConfig": ("training.rl", "GRPOConfig"),
        "PPODataset": ("training.rl", "PPODataset"),
        "PPOTrainer": ("training.rl", "PPOTrainer"),
        "GRPOTrainer": ("training.rl", "GRPOTrainer"),
        "RewardModel": ("training.rl", "RewardModel"),
        "RewardModelTrainer": ("training.rl", "RewardModelTrainer"),
    }
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'training' has no attribute '{name}'")


__all__ = [
    "TrainingConfig", "CheckpointManager", "MetricsTracker",
    "TrainerBase", "SFTTrainer", "SFTDataset",
    "PretrainTrainer", "PretrainDataset",
    "DPOTrainer", "DPODataset",
    "DistillationTrainer", "DistillationDataset",
    "CurriculumDataset", "StageScheduler",
    "apply_lora", "merge_lora", "lora_state_dict", "LoRALayer",
    "PPOConfig", "GRPOConfig", "PPODataset",
    "PPOTrainer", "GRPOTrainer",
    "RewardModel", "RewardModelTrainer",
    "DistillationPipeline", "DistillationConfig", "distill_and_train",
]
