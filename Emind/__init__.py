"""
Emind — 亦梓科技 AI 模型核心包 (v2.0)
统一重新导出
"""
import warnings

# Model
from model import EmindLM, EmindConfig, create_model

# Tokenizer
from tokenizer import EmindTokenizer

# Training
from training import (
    TrainerBase, SFTTrainer, DPOTrainer, DistillationTrainer,
    TrainingConfig, SFTDataset, DPODataset, DistillationDataset,
    CheckpointManager, MetricsTracker, apply_lora, merge_lora, LoRALayer,
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
from distributed_utils import setup_ddp, setup_fsdp

__version__ = "2.0.0"

__all__ = [
    # Model
    "EmindLM", "EmindConfig", "create_model",
    # Tokenizer
    "EmindTokenizer",
    # Training
    "TrainerBase", "SFTTrainer", "DPOTrainer", "DistillationTrainer",
    "TrainingConfig", "SFTDataset", "DPODataset", "DistillationDataset",
    "CheckpointManager", "MetricsTracker", "apply_lora", "merge_lora", "LoRALayer",
    # Inference
    "UnifiedInferenceEngine", "BackendConfig", "create_inference_engine", "get_cloud_api_models",
    # Data Pipeline
    "DataCollector", "DataCleaner", "DataSynthesizer", "DataFormatter", "DatasetManager",
    # Evaluation
    "EvaluatorBase", "MMLUEvaluator", "CEvalEvaluator", "HumanEvalEvaluator", "EvaluationRunner",
    # Distributed
    "setup_ddp", "setup_fsdp",
]
