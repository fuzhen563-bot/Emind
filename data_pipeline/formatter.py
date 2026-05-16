"""
Data Formatter — 数据格式化，支持多种训练格式
SFT: {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
DPO: {"prompt": ..., "chosen": ..., "rejected": ...}
Pretrain: {"text": ...}
"""
import json
from typing import List, Dict, Any, Optional


class DataFormatter:
    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt or "你是一个有用的人工智能助手。"

    def to_sft(self, items: List[Dict], prompt_key: str = "prompt",
               response_key: str = "response", use_system: bool = True) -> List[Dict]:
        results = []
        for item in items:
            prompt = item.get(prompt_key, item.get("instruction", item.get("text", "")))
            response = item.get(response_key, item.get("output", item.get("answer", "")))
            if not prompt or not response:
                continue
            messages = []
            if use_system:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": response})
            results.append({"messages": messages})
        print(f"Formatted {len(results)} SFT samples")
        return results

    def to_dpo(self, items: List[Dict], prompt_key: str = "prompt",
               chosen_key: str = "chosen", rejected_key: str = "rejected") -> List[Dict]:
        results = []
        for item in items:
            prompt = item.get(prompt_key, "")
            chosen = item.get(chosen_key, "")
            rejected = item.get(rejected_key, "")
            if not prompt or not chosen or not rejected:
                continue
            results.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
        print(f"Formatted {len(results)} DPO samples")
        return results

    def to_pretrain(self, items: List[Any], text_key: str = "text") -> List[Dict]:
        results = []
        for item in items:
            if isinstance(item, str):
                results.append({"text": item})
            elif isinstance(item, dict):
                text = item.get(text_key, item.get("prompt", item.get("content", json.dumps(item, ensure_ascii=False))))
                results.append({"text": text})
            else:
                results.append({"text": str(item)})
        print(f"Formatted {len(results)} pretrain samples")
        return results

    def to_alpaca(self, items: List[Dict]) -> List[Dict]:
        results = []
        for item in items:
            inst = item.get("instruction", item.get("prompt", ""))
            inp = item.get("input", "")
            out = item.get("output", item.get("response", ""))
            if not inst or not out:
                continue
            results.append({
                "instruction": inst,
                "input": inp,
                "output": out,
            })
        print(f"Formatted {len(results)} Alpaca-style samples")
        return results

    def to_sharegpt(self, items: List[Dict]) -> List[Dict]:
        results = []
        for item in items:
            conversations = []
            prompt = item.get("prompt", item.get("instruction", ""))
            response = item.get("response", item.get("output", ""))
            if prompt:
                conversations.append({"from": "human", "value": prompt})
            if response:
                conversations.append({"from": "gpt", "value": response})
            if conversations:
                results.append({"conversations": conversations})
        print(f"Formatted {len(results)} ShareGPT-style samples")
        return results

    def to_openai(self, items: List[Dict]) -> List[Dict]:
        return self.to_sft(items, use_system=False)

    def save_jsonl(self, items: List[Dict], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(items)} items to {filepath}")

    def load_jsonl(self, filepath: str) -> List[Dict]:
        items = []
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
