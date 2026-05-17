"""
Emind — 亦梓科技 AI 模型核心包 (v2.0)
统一重新导出
"""
import os
import sys
import warnings

# Ensure project root is on path for root-level modules (model.py, tokenizer.py, etc.)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Model
from model import EmindLM, EmindConfig, create_model

# Tokenizer
from tokenizer import EmindTokenizer

# Training
from training import (
    TrainerBase, SFTTrainer, DPOTrainer, DistillationTrainer,
    TrainingConfig, SFTDataset, DPODataset, DistillationDataset,
    CheckpointManager, MetricsTracker, apply_lora, merge_lora, LoRALayer,
    PPOConfig, GRPOConfig, PPODataset,
    PPOTrainer, GRPOTrainer,
    RewardModel, RewardModelTrainer,
    DistillationPipeline, DistillationConfig, distill_and_train,
)

# Unified Inference
from unified_inference import (
    UnifiedInferenceEngine, BackendConfig,
    create_inference_engine, get_cloud_api_models,
)

# Data Pipeline
try:
    from data_pipeline import (
        DataCollector, DataCleaner, DataSynthesizer,
        DataFormatter, DatasetManager,
    )
except ImportError:
    warnings.warn("data_pipeline module not available", ImportWarning)

# Evaluation
try:
    from eval import (
        EvaluatorBase, MMLUEvaluator, CEvalEvaluator,
        HumanEvalEvaluator, EvaluationRunner,
    )
except ImportError:
    warnings.warn("eval module not available", ImportWarning)

# Distributed
from distributed_utils import setup_distributed, cleanup_distributed, create_distributed_model, create_distributed_dataloader, is_main_process

__version__ = "2.0.0"

__all__ = [
    # Model
    "EmindLM", "EmindConfig", "create_model",
    # Tokenizer
    "EmindTokenizer",
    # Training
    "TrainerBase", "SFTTrainer", "DPOTrainer", "DistillationTrainer",
    "PPOTrainer", "GRPOTrainer",
    "TrainingConfig", "SFTDataset", "DPODataset", "DistillationDataset",
    "PPOConfig", "GRPOConfig", "PPODataset",
    "CheckpointManager", "MetricsTracker", "apply_lora", "merge_lora", "LoRALayer",
    "RewardModel", "RewardModelTrainer",
    "DistillationPipeline", "DistillationConfig", "distill_and_train",
    # Inference
    "UnifiedInferenceEngine", "BackendConfig", "create_inference_engine", "get_cloud_api_models",
    # Data Pipeline
    "DataCollector", "DataCleaner", "DataSynthesizer", "DataFormatter", "DatasetManager",
    # Evaluation
    "EvaluatorBase", "MMLUEvaluator", "CEvalEvaluator", "HumanEvalEvaluator", "EvaluationRunner",
    # Distributed
    "setup_distributed", "cleanup_distributed", "create_distributed_model", "create_distributed_dataloader", "is_main_process",
]
