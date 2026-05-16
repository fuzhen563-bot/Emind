"""
Data Cleaner — 数据清洗与质量过滤
支持：去重、质量评分、PII 脱敏、语言检测、长度过滤
"""
import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter


class DataCleaner:
    def __init__(self, min_length: int = 10, max_length: int = 8192):
        self.min_length = min_length
        self.max_length = max_length

    def deduplicate(self, items: List[Any], key: Optional[str] = None) -> List[Any]:
        seen: Set[str] = set()
        result = []
        for item in items:
            if isinstance(item, dict) and key:
                sig = str(item.get(key, ""))
            else:
                sig = str(item)
            if sig not in seen:
                seen.add(sig)
                result.append(item)
        removed = len(items) - len(result)
        if removed:
            print(f"Dedup removed {removed} items ({len(result)} remaining)")
        return result

    def filter_length(self, items: List[Any], text_field: Optional[str] = None) -> List[Any]:
        result = []
        for item in items:
            text = item if isinstance(item, str) else (item.get(text_field or "text", "") if isinstance(item, dict) else str(item))
            length = len(text)
            if self.min_length <= length <= self.max_length:
                result.append(item)
        removed = len(items) - len(result)
        if removed:
            print(f"Length filter removed {removed} items (min={self.min_length}, max={self.max_length})")
        return result

    def filter_quality(self, items: List[Any], text_field: Optional[str] = None, threshold: float = 0.1) -> List[Any]:
        result = []
        for item in items:
            text = item if isinstance(item, str) else (item.get(text_field or "text", "") if isinstance(item, dict) else str(item))
            score = self._quality_score(text)
            if score >= threshold:
                result.append(item)
        removed = len(items) - len(result)
        if removed:
            print(f"Quality filter removed {removed} items (threshold={threshold})")
        return result

    def _quality_score(self, text: str) -> float:
        if not text:
            return 0.0
        score = 1.0
        ascii_chars = sum(1 for c in text if 32 <= ord(c) < 127)
        total_chars = len(text)
        ascii_ratio = ascii_chars / total_chars if total_chars > 0 else 0
        if ascii_ratio > 0.9:
            score -= 0.3
        repeated = re.findall(r'(.)\1{4,}', text)
        if repeated:
            score -= 0.2 * min(len(repeated), 3)
        url_count = len(re.findall(r'https?://\S+', text))
        if url_count > 5:
            score -= 0.1
        gibberish = re.findall(r'[^\w\s\u4e00-\u9fff]', text)
        if len(gibberish) > total_chars * 0.3:
            score -= 0.3
        return max(0.0, score)

    def strip_pii(self, text: str, mask_token: str = "[REDACTED]") -> str:
        phone = re.sub(r'1[3-9]\d{9}', mask_token, text)
        id_card = re.sub(r'\d{17}[\dXx]', mask_token, phone)
        email = re.sub(r'[\w.-]+@[\w.-]+\.\w+', mask_token, id_card)
        ip = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', mask_token, email)
        bank = re.sub(r'\d{16,19}', mask_token, ip)
        return bank

    def detect_language(self, text: str) -> str:
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_chars = len(re.findall(r'[a-zA-Z]', text))
        total = max(cn_chars + en_chars, 1)
        if cn_chars / total > 0.3:
            return "zh"
        if en_chars / total > 0.3:
            return "en"
        if cn_chars > en_chars:
            return "zh"
        return "en"

    def filter_language(self, items: List[Any], target: str = "zh", text_field: Optional[str] = None) -> List[Any]:
        result = []
        for item in items:
            text = item if isinstance(item, str) else (item.get(text_field or "text", "") if isinstance(item, dict) else str(item))
            if self.detect_language(text) == target:
                result.append(item)
        removed = len(items) - len(result)
        if removed:
            print(f"Language filter removed {removed} items (target={target})")
        return result

    def clean(self, items: List[Any], dedup_key: Optional[str] = None, text_field: Optional[str] = None,
              target_lang: Optional[str] = None, quality_threshold: float = 0.0) -> List[Any]:
        items = self.deduplicate(items, key=dedup_key)
        items = self.filter_length(items, text_field=text_field)
        if quality_threshold > 0:
            items = self.filter_quality(items, text_field=text_field, threshold=quality_threshold)
        if target_lang:
            items = self.filter_language(items, target=target_lang, text_field=text_field)
        return items

    def strip_pii_from_dataset(self, items: List[Any], text_fields: Optional[List[str]] = None) -> List[Any]:
        if text_fields is None:
            text_fields = ["text", "prompt", "response", "content"]
        result = []
        for item in items:
            if isinstance(item, str):
                result.append(self.strip_pii(item))
            elif isinstance(item, dict):
                copied = dict(item)
                for field in text_fields:
                    if field in copied and isinstance(copied[field], str):
                        copied[field] = self.strip_pii(copied[field])
                result.append(copied)
            else:
                result.append(item)
        return result
