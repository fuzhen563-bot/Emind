"""
蒸馏引擎 — 连接 Teacher 模型 → 生成 SFT 数据 → 输出 JSONL
"""
import json
import os
import sys
import time

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Optional

from zhengliu.config import DistillConfig, parse_args
from zhengliu.seeds import generate_prompts


class DistillEngine:
    def __init__(self, config: DistillConfig):
        self.config = config
        self.teacher = None
        self._use_requests = False
        if not config.dry_run:
            self._init_teacher()

    def _init_teacher(self):
        backend_type = self.config.backend_type
        try:
            from unified_inference import UnifiedInferenceEngine, BackendConfig
        except ImportError:
            print("  unified_inference 不可用，使用 HTTP 直连模式")
            self._use_requests = True
            return

        cfg = BackendConfig(
            backend_type=backend_type,
            model_name=self.config.model_name,
            base_url=self.config.resolved_base_url,
            api_key=self.config.resolved_api_key,
        )
        self.teacher = UnifiedInferenceEngine(cfg)
        if self.teacher.is_available():
            print(f"  Teacher 已就绪: {backend_type} / {cfg.model_name}")
            self._use_requests = False
        else:
            print(f"  Teacher 后端 {backend_type} 不可用，回退 HTTP 直连")
            if cfg.resolved_api_key:
                masked = cfg.resolved_api_key[:8] + "..." + cfg.resolved_api_key[-4:]
                print(f"    使用 API Key: {masked}")
            print(f"    Endpoint: {cfg.base_url or cfg.resolved_base_url}/v1/chat/completions")
            print(f"    Model: {cfg.model_name}")
            self._use_requests = True

    def _generate_via_requests(self, prompt: str) -> str:
        import requests

        cfg = self.config
        headers = {"Content-Type": "application/json"}
        ak = cfg.resolved_api_key
        if ak:
            headers["Authorization"] = f"Bearer {ak}"

        payload = {
            "model": cfg.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }
        url = f"{cfg.resolved_base_url.rstrip('/')}/v1/chat/completions"

        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                body = resp.text[:300]
                print(f"  HTTP {resp.status_code} (尝试 {attempt+1}/3): {body}")
                if attempt < 2:
                    time.sleep(2)
            except requests.exceptions.ConnectionError:
                print(f"  ⚠ 连接失败 (尝试 {attempt+1}/3): 无法连接到 {url}")
                print(f"    请检查 --base-url 是否正确、网络是否可达")
                time.sleep(3)
            except requests.exceptions.Timeout:
                print(f"  ⚠ 超时 (尝试 {attempt+1}/3): 请求超时 ({120}s)")
                time.sleep(3)
            except Exception as e:
                print(f"  ⚠ 请求异常 (尝试 {attempt+1}/3): {e}")
                time.sleep(2)
        return ""

    def generate_one(self, item: dict, use_cot: bool) -> Optional[dict]:
        prompt = item["prompt"]
        data_type = item.get("type", "code")
        identity = self.config.identity

        if use_cot:
            full_prompt = f"{identity}\n\n{prompt}\n\n请先逐步思考，再给出最终答案。"
        else:
            full_prompt = f"{identity}\n\n{prompt}"

        try:
            if self._use_requests:
                response = self._generate_via_requests(full_prompt)
            else:
                response = self.teacher.generate(
                    full_prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            return None

        if not response or len(response.strip()) < 10:
            reason = "空响应" if not response else f"过短 ({len(response.strip())} 字符)"
            if response and len(response.strip()) < 10:
                print(f"  ✗ 跳过: {reason} — 响应内容: {response.strip()[:60]}")
            return None

        strategy = "cot" if use_cot else "direct"
        result = {
            "response": response.strip(),
            "strategy": strategy,
            "source": "zhengliu",
        }
        result.update(item)
        return result

    def run(self) -> list:
        cfg = self.config

        all_items = []
        for dtype, count in cfg.type_counts.items():
            prompts = generate_prompts(dtype, count)
            for p in prompts:
                if isinstance(p, str):
                    all_items.append({"prompt": p, "type": dtype})
                else:
                    item = {"type": dtype}
                    item.update(p)
                    all_items.append(item)

        print(f"共 {len(all_items)} 条种子 prompt")
        for dtype, count in cfg.type_counts.items():
            print(f"  {dtype}: {count} 条")

        # 预览模式：只打印不调 API
        if cfg.dry_run:
            print("\n=== 预览模式 (--dry-run) ===")
            for i, item in enumerate(all_items, 1):
                print(f"\n--- [{item.get('type','?')}] Prompt {i} ---")
                print(item["prompt"][:500])
                if len(item["prompt"]) > 500:
                    print(f"... (共 {len(item['prompt'])} 字符)")
            print(f"\n共 {len(all_items)} 条 prompt，未调用 API")
            return []

        print("\n开始蒸馏...")

        results = []
        done = 0

        def worker(item):
            use_cot = not cfg.no_cot and item.get("type") in ("reasoning", "deep_reasoning")
            return self.generate_one(item, use_cot=use_cot)

        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = [pool.submit(worker, item) for item in all_items]
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)
                done += 1
                if done % 10 == 0 or done == len(all_items):
                    pct = done / len(all_items) * 100
                    bar_len = 20
                    filled = int(bar_len * done / len(all_items))
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"  [{bar}] {done}/{len(all_items)} ({pct:.0f}%) 成功: {len(results)}")

        # 去重
        seen = set()
        deduped = []
        for r in results:
            h = hashlib.md5((r["prompt"] + r["response"]).encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                deduped.append(r)

        return deduped


def main():
    cfg = parse_args()
    engine = DistillEngine(cfg)
    data = engine.run()

    if cfg.dry_run:
        return

    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(cfg.output_dir, f"{ts}_distilled_sft.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_attempted = sum(cfg.type_counts.values())
    by_type = Counter(r["type"] for r in data)
    print(f"\n完成！{len(data)} 条有效数据（共尝试 {total_attempted} 条，失败 {total_attempted - len(data)} 条）")
    if data:
        print(f"输出: {out_path}")
        print(f"类型分布: {dict(by_type)}")
    else:
        print("建议:")
        print("  1. python3 -m zhengliu.distill --dry-run --code 5  先预览 prompt")
        print("  2. 检查 --api-key / --base-url / --model 是否正确")
        print("  3. 确认 Teacher API 服务可用")


if __name__ == "__main__":
    main()
