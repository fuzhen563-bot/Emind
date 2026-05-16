"""
HumanEval Evaluator — 代码生成评测
格式：{"prompt": ..., "test": ..., "entry_point": ...}
"""
import json
import signal
from typing import Dict, Any, List, Optional, Callable
from eval.base import EvaluatorBase


class HumanEvalEvaluator(EvaluatorBase):
    def __init__(self, timeout: int = 30):
        super().__init__("HumanEval")
        self.data: List[Dict] = []
        self.timeout = timeout

    def load(self, path: Optional[str] = None):
        if path is None:
            path = "data/eval/humaneval.jsonl"
        self.data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        print(f"HumanEval: loaded {len(self.data)} problems")

    def evaluate(self, model_fn: Callable, split: str = "test", **kwargs) -> Dict[str, Any]:
        sample_limit = kwargs.get("sample_limit")
        tasks = self.data[:sample_limit] if sample_limit else self.data
        passed = 0
        total = 0
        results = []
        for item in tasks:
            prompt = item["prompt"]
            entry_point = item["entry_point"]
            test_code = item["test"]
            completion = model_fn(prompt)
            full_code = prompt + completion + "\n" + test_code
            is_pass = self._check_execution(full_code, entry_point)
            if is_pass:
                passed += 1
            total += 1
            results.append({"entry_point": entry_point, "passed": is_pass})
        self.results = {
            "pass@1": passed / max(total, 1),
            "passed": passed,
            "total": total,
            "details": results,
        }
        return self.results

    def _check_execution(self, code: str, entry_point: str) -> bool:
        try:
            local_vars = {}
            exec(code, {"__builtins__": __builtins__}, local_vars)
            if entry_point not in local_vars:
                return False
            return True
        except Exception:
            return False
