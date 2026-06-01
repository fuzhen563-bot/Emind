"""
Dataset Manager — 数据集版本管理与自动处理管线
"""
import os
import json
import shutil
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Union


class DatasetManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.cache_dir = self.base_dir / "cache"
        self.version_dir = self.base_dir / "versions"
        for d in [self.raw_dir, self.processed_dir, self.cache_dir, self.version_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def register_raw(self, source_path: str, name: Optional[str] = None) -> str:
        src = Path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")
        if name is None:
            name = src.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = self.raw_dir / f"{name}_{timestamp}{src.suffix}"
        shutil.copy2(str(src), str(dest))
        print(f"Registered raw data: {dest}")
        return str(dest)

    def process(self, raw_path: str, pipeline: List[Callable],
                output_name: Optional[str] = None) -> str:
        items = self._load_raw(raw_path)
        for fn in pipeline:
            items = fn(items)
        if output_name is None:
            output_name = Path(raw_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.processed_dir / f"{output_name}_{timestamp}.jsonl"
        with open(str(out_path), "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Processed -> {out_path} ({len(items)} items)")
        return str(out_path)

    def _load_raw(self, path: str) -> List:
        if path.endswith(".jsonl"):
            items = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            return items
        elif path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        elif path.endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                return [{"text": line.strip()} for line in f if line.strip()]
        else:
            raise ValueError(f"Unsupported format: {path}")

    def list_raw(self) -> List[Dict]:
        entries = []
        for f in sorted(self.raw_dir.iterdir()):
            if f.is_file():
                entries.append({"name": f.name, "path": str(f), "size": f.stat().st_size})
        return entries

    def list_processed(self) -> List[Dict]:
        entries = []
        for f in sorted(self.processed_dir.glob("*.jsonl")):
            entries.append({"name": f.name, "path": str(f), "size": f.stat().st_size})
        return entries

    def create_version(self, name: str, files: Optional[List[str]] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ver = f"{name}_{timestamp}"
        ver_path = self.version_dir / ver
        ver_path.mkdir(parents=True, exist_ok=True)
        manifest = {"version": ver, "created": timestamp, "files": []}
        if files:
            for f in files:
                src = Path(f)
                if src.exists():
                    dest = ver_path / src.name
                    shutil.copy2(str(src), str(dest))
                    manifest["files"].append({"name": src.name, "path": str(dest), "size": src.stat().st_size})
        with open(str(ver_path / "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"Created version: {ver}")
        return str(ver_path)

    def list_versions(self) -> List[Dict]:
        versions = []
        for d in sorted(self.version_dir.iterdir()):
            if d.is_dir():
                manifest_path = d / "manifest.json"
                if manifest_path.exists():
                    with open(str(manifest_path)) as f:
                        versions.append(json.load(f))
                else:
                    versions.append({"version": d.name, "created": "unknown"})
        return versions

    def get_version(self, version: str) -> Optional[str]:
        for d in self.version_dir.iterdir():
            if d.is_dir() and d.name == version:
                return str(d)
        for d in self.version_dir.iterdir():
            if d.is_dir() and d.name.startswith(version):
                return str(d)
        return None

    def export(self, source: str, output_path: str, format_fn: Callable,
               limit: Optional[int] = None) -> str:
        items = self._load_raw(source)
        if limit:
            items = items[:limit]
        formatted = format_fn(items)
        out_dir = Path(output_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if output_path.endswith(".jsonl"):
            with open(output_path, "w", encoding="utf-8") as f:
                for item in formatted:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(formatted, f, ensure_ascii=False, indent=2)
        print(f"Exported {len(formatted)} items to {output_path}")
        return output_path

    def stats(self, path: str) -> Dict:
        items = self._load_raw(path)
        total_chars = 0
        total_tokens_est = 0
        for item in items:
            text = json.dumps(item, ensure_ascii=False)
            total_chars += len(text)
            zh_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            en_count = len(text) - zh_count
            total_tokens_est += int(zh_count * 1.5 + en_count * 0.25) + 1
        return {
            "file": path,
            "samples": len(items),
            "total_chars": total_chars,
            "estimated_tokens": total_tokens_est,
            "avg_chars_per_sample": total_chars // max(len(items), 1),
        }
