"""
EvaluationRunner — 统一评测运行器
"""
import json
import time
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime

from eval.base import EvaluatorBase
from eval.mmlu import MMLUEvaluator
from eval.ceval import CEvalEvaluator
from eval.humaneval import HumanEvalEvaluator


class EvaluationRunner:
    def __init__(self, model_fn: Callable, eval_dir: str = "eval_results"):
        self.model_fn = model_fn
        self.eval_dir = Path(eval_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.evaluators = {
            "mmlu": MMLUEvaluator(),
            "ceval": CEvalEvaluator(),
            "humaneval": HumanEvalEvaluator(),
        }

    def register(self, name: str, evaluator: EvaluatorBase):
        self.evaluators[name] = evaluator

    def run(self, benchmarks: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        if benchmarks is None:
            benchmarks = list(self.evaluators.keys())
        results = {}
        for name in benchmarks:
            if name not in self.evaluators:
                print(f"Unknown benchmark: {name}, skipping")
                continue
            print(f"\n{'='*40}")
            print(f"Running {name}...")
            print(f"{'='*40}")
            evaluator = self.evaluators[name]
            try:
                evaluator.load(kwargs.pop("data_path", None))
            except FileNotFoundError:
                print(f"Data not found for {name}, skipping")
                continue
            t0 = time.time()
            eval_results = evaluator.evaluate(self.model_fn, **kwargs)
            elapsed = time.time() - t0
            results[name] = {
                "results": eval_results,
                "time_seconds": round(elapsed, 2),
                "summary": evaluator.summary(),
            }
            print(evaluator.summary())
            print(f"Time: {elapsed:.1f}s")
        self.results = results
        self._save(results)
        return results

    def _save(self, results: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.eval_dir / f"eval_{timestamp}.json"
        with open(str(path), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {path}")

    def print_leaderboard(self, results: Optional[Dict[str, Any]] = None):
        if results is None:
            results = getattr(self, "results", {})
        if not results:
            print("No results to display")
            return
        print("\n" + "=" * 50)
        print("                 Emind Leaderboard")
        print("=" * 50)
        for name, data in sorted(results.items()):
            r = data.get("results", {})
            overall = r.get("overall", r.get("pass@1", 0))
            if isinstance(overall, float):
                print(f"  {name:12s}: {overall:.2%}")
        print("=" * 50)
