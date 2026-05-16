"""
MMLU Evaluator — 多任务语言理解评测
格式：{"question": ..., "choices": [..., ...], "answer": 0~3}
"""
import json
import random
from typing import Dict, Any, List, Optional, Callable
from eval.base import EvaluatorBase


class MMLUEvaluator(EvaluatorBase):
    def __init__(self):
        super().__init__("MMLU")
        self.data: List[Dict] = []
        self.subjects: Dict[str, List[Dict]] = {}

    def load(self, path: Optional[str] = None):
        if path is None:
            path = "data/eval/mmlu.jsonl"
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.data.append(item)
                    subj = item.get("subject", "unknown")
                    if subj not in self.subjects:
                        self.subjects[subj] = []
                    self.subjects[subj].append(item)
        print(f"MMLU: loaded {len(self.data)} questions across {len(self.subjects)} subjects")
        for subj, items in sorted(self.subjects.items()):
            print(f"  {subj}: {len(items)}")

    def evaluate(self, model_fn: Callable, split: str = "test", **kwargs) -> Dict[str, Any]:
        subset = kwargs.get("subset")
        sample_limit = kwargs.get("sample_limit")
        tasks = self.data
        if subset and subset in self.subjects:
            tasks = self.subjects[subset]
        if sample_limit:
            tasks = tasks[:sample_limit]
        correct = 0
        total = 0
        self.results = {}
        for subj, items in self._group_by_subject(tasks).items():
            subj_correct = 0
            for item in items:
                prompt = self._format_prompt(item)
                pred = model_fn(prompt)
                answer_idx = self._extract_answer(pred, item["choices"])
                if answer_idx == item["answer"]:
                    correct += 1
                    subj_correct += 1
                total += 1
            self.results[subj] = subj_correct / max(len(items), 1)
        self.results["overall"] = correct / max(total, 1)
        return self.results

    def _group_by_subject(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        groups = {}
        for item in items:
            subj = item.get("subject", "unknown")
            if subj not in groups:
                groups[subj] = []
            groups[subj].append(item)
        return groups

    def _format_prompt(self, item: Dict) -> str:
        question = item["question"]
        choices = item["choices"]
        labels = ["A", "B", "C", "D"]
        lines = [f"Question: {question}"]
        for i, choice in enumerate(choices):
            if i < len(labels):
                lines.append(f"{labels[i]}. {choice}")
        lines.append("\nAnswer:")
        return "\n".join(lines)

    def _extract_answer(self, text: str, choices: List[str]) -> int:
        text = text.strip().upper()
        labels = ["A", "B", "C", "D"]
        for i, label in enumerate(labels):
            if text.startswith(label) or f"({label})" in text or f"[{label}]" in text:
                return i
        for i, choice in enumerate(choices):
            if choice.strip().lower() in text.lower():
                return i
        return -1
