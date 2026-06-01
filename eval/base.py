"""
EvaluatorBase — 评测基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable


class EvaluatorBase(ABC):
    def __init__(self, name: str):
        self.name = name
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def load(self, path: Optional[str] = None):
        ...

    @abstractmethod
    def evaluate(self, model_fn: Callable, split: str = "test", **kwargs) -> Dict[str, Any]:
        ...

    def summary(self) -> str:
        lines = [f"=== {self.name} ==="]
        for key, val in sorted(self.results.items()):
            if isinstance(val, float):
                lines.append(f"  {key}: {val:.2%}")
            else:
                lines.append(f"  {key}: {val}")
        return "\n".join(lines)

    def accuracy(self, predictions: List[str], references: List[str]) -> float:
        correct = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
        return correct / max(len(references), 1)
