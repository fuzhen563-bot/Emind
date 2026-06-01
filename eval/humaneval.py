"""
HumanEval Evaluator — 代码生成评测
格式：{"prompt": ..., "test": ..., "entry_point": ...}
"""
import json
import threading
from typing import Dict, Any, List, Optional, Callable
from eval.base import EvaluatorBase


class TimeoutError(Exception):
    pass


def _run_with_timeout(code: str, local_vars: dict, timeout: float, result_holder: list):
    try:
        exec(code, {"__builtins__": {}}, local_vars)
        result_holder.append(True)
    except Exception:
        result_holder.append(False)


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
        local_vars = {}
        result_holder = []
        t = threading.Thread(
            target=_run_with_timeout,
            args=(code, local_vars, self.timeout, result_holder),
            daemon=True,
        )
        t.start()
        t.join(timeout=self.timeout)
        if t.is_alive():
            return False
        if not result_holder or not result_holder[0]:
            return False
        return entry_point in local_vars
