"""Emind config wrapper — delegates to root-level model and training."""
from model import EmindConfig
from training import TrainingConfig
__all__ = ["EmindConfig", "TrainingConfig"]
