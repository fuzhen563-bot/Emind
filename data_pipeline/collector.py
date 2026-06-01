"""
Data Collector — 多源数据采集
支持：本地文件、JSON/JSONL、CSV、网页抓取、API 摄入
"""
import os
import json
import csv
import glob
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path


class DataCollector:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def from_text(self, file_path: str) -> List[str]:
        lines = []
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        print(f"Loaded {len(lines)} lines from {file_path}")
        return lines

    def from_json(self, file_path: str) -> List[Dict]:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"Loaded {len(data)} items from {file_path}")
            return data
        print(f"Loaded JSON object from {file_path}")
        return [data]

    def from_jsonl(self, file_path: str) -> List[Dict]:
        items = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        print(f"Loaded {len(items)} items from {file_path}")
        return items

    def from_csv(self, file_path: str, text_column: str = "text") -> List[str]:
        items = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column in row and row[text_column].strip():
                    items.append(row[text_column].strip())
        print(f"Loaded {len(items)} items from {file_path}")
        return items

    def from_directory(self, dir_path: str, pattern: str = "*.txt") -> List[str]:
        items = []
        for fpath in glob.glob(os.path.join(dir_path, pattern)):
            items.extend(self.from_text(fpath))
        print(f"Loaded {len(items)} items from {dir_path}/{pattern}")
        return items

    def from_conversations(self, file_path: str) -> List[Dict]:
        data = self.from_json(file_path) if file_path.endswith(".json") else self.from_jsonl(file_path)
        conversations = []
        for item in data:
            messages = item.get("messages", item.get("conversations", []))
            i = 0
            while i < len(messages) - 1:
                if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                    conversations.append({
                        "prompt": messages[i].get("content", messages[i].get("value", "")),
                        "response": messages[i + 1].get("content", messages[i + 1].get("value", "")),
                    })
                    i += 2
                else:
                    i += 1
        print(f"Extracted {len(conversations)} prompt-response pairs")
        return conversations

    def list_sources(self) -> List[str]:
        sources = []
        for ext in ("*.txt", "*.json", "*.jsonl", "*.csv"):
            for f in self.data_dir.glob(ext):
                sources.append(str(f))
            for f in self.data_dir.glob(f"**/{ext}"):
                if str(f) not in sources:
                    sources.append(str(f))
        return sorted(sources)

    def collect(self, source: str) -> List:
        if not os.path.exists(source):
            return []
        if source.endswith(".txt"):
            return self.from_text(source)
        elif source.endswith(".jsonl"):
            return self.from_jsonl(source)
        elif source.endswith(".json"):
            return self.from_json(source)
        elif source.endswith(".csv"):
            return self.from_csv(source)
        else:
            return self.from_directory(source)
