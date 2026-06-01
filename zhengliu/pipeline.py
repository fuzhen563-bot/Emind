"""
Pipeline — 一键自动蒸馏
  自动发现模型 → 轮询切换 → 无限/固定轮次 → 累积输出
  支持 checkpoint/resume (V2.0)
"""
import hashlib
import json
import os
import sys
import time
import signal
from typing import List, Optional

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import requests

from zhengliu.config import DistillConfig
from zhengliu.distill import DistillEngine

_NON_CHAT = ["embedding", "whisper", "tts", "moderation", "davinci", "instruct", "dall-e"]


class Pipeline:
    """多模型自动蒸馏流水线"""

    def __init__(self, config: DistillConfig):
        self.config = config
        self.model_pool: List[dict] = []
        self.pointer = 0
        self.total_success = 0
        self._stop = False
        self._dead_models = set()  # 已确认不可用的模型
        self._checkpoint_path: Optional[str] = None  # V2.0: checkpoint 路径

    def discover_models(self, verify: bool = True) -> list:
        """发现 + 验证 API 上所有可用模型，只保留通过预检的"""
        cfg = self.config
        url = f"{cfg.resolved_base_url.rstrip('/')}/v1/models"
        headers = {"Authorization": f"Bearer {cfg.resolved_api_key}"}
        print(f"  发现模型: GET {url}")
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                print(f"  失败 HTTP {resp.status_code}")
                return []
        except Exception as e:
            print(f"  连接失败: {e}")
            return []

        raw = resp.json().get("data", [])
        exclude = set(m.strip().lower() for m in cfg.exclude_models.split(",") if m.strip())
        candidates = []
        for m in raw:
            mid = m.get("id", "")
            if not mid: continue
            if any(kw in mid.lower() for kw in _NON_CHAT): continue
            if mid.lower() in exclude: continue
            candidates.append(m)
        candidates.sort(key=lambda x: x["id"])

        if not verify:
            print(f"  发现 {len(raw)} 个, 过滤后 {len(candidates)} 个 (未验证):")
            for m in candidates: print(f"    - {m['id']}")
            return candidates

        # 逐一验证可用性 (限制并发避免 rate limit)
        available = []
        dead = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=3) as pool:  # 降到 3 避免限流
            futures = {pool.submit(self._quick_health_check, m["id"]): m for m in candidates}
            for f in as_completed(futures):
                m = futures[f]
                ok = f.result()
                if ok:
                    available.append(m)
                else:
                    dead.append(m)

        available.sort(key=lambda x: x["id"])
        print(f"  发现 {len(raw)} 个, 过滤后 {len(candidates)} 个候选, 验证通过 {len(available)} 个:")
        for m in available:
            print(f"    ✓ {m['id']}")
        if dead:
            print(f"  不可用 ({len(dead)} 个):")
            for m in dead:
                print(f"    ✗ {m['id']}")
            self._dead_models.update(m["id"] for m in dead)
        return available

    def stop(self):
        self._stop = True

    def _quick_health_check(self, model_name: str) -> bool:
        """快速预检：发送最短 prompt 确认模型可用"""
        cfg = self.config
        url = f"{cfg.resolved_base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if cfg.resolved_api_key:
            headers["Authorization"] = f"Bearer {cfg.resolved_api_key}"
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "1+1=?"}],
            "max_tokens": 10,
            "temperature": 0.0,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            if resp.status_code == 400:
                body = resp.text[:200]
                print(f"    400 详情: {body}")
                # 检查是否是参数错误 vs 模型不存在
                if "model" in body.lower() or "not found" in body.lower():
                    return False
                return False
            return False
        except Exception:
            return False

    # ── checkpoint 支持 ──────────────────────────────────────────────

    def _load_checkpoint(self, path: str) -> List[str]:
        """加载已有 checkpoint，返回已完成 prompt 的哈希集合 (跳过)"""
        seen = set()
        if not os.path.isfile(path):
            print(f"  checkpoint 不存在: {path}")
            return []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                h = hashlib.md5(line.encode()).hexdigest()
                seen.add(h)
        print(f"  加载 checkpoint: {path} ({len(seen)} 条已完成)")
        return seen

    def _save_checkpoint(self, path: str, results: list) -> None:
        """增量保存蒸馏结果到 checkpoint"""
        mode = "a" if os.path.exists(path) else "w"
        with open(path, mode, encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._checkpoint_path = path

    # ── 主循环 ───────────────────────────────────────────────────────

    def auto_run(self, resume_from: str = "") -> list:
        cfg = self.config
        signal.signal(signal.SIGINT, lambda s, f: self.stop())

        if cfg.dry_run:
            print("\n=== 预览模式 ===")
            return []

        # 构建模型池
        if cfg.auto_models:
            self.model_pool = [{"id": m.strip()} for m in cfg.auto_models.split(",") if m.strip()]
        elif cfg.auto_discover:
            self.model_pool = self.discover_models()
        if not self.model_pool:
            self.model_pool = [{"id": cfg.model_name or "gpt-4o-mini"}]

        print(f"模型池: {[m['id'] for m in self.model_pool]}")
        max_runs = cfg.auto_runs
        infinite = max_runs <= 0
        print(f"模式: {'♾ 无限' if infinite else f'{max_runs} 轮'}")
        if infinite: print("按 Ctrl+C 停止")

        cumulative = []
        out_dir = cfg.output_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cumu_path = os.path.join(out_dir, f"{ts}_auto_distill.jsonl")
        log_path = os.path.join(out_dir, f"{ts}_auto_run.log")

        # V2.0: checkpoint resume
        seen_prompts = set()
        if resume_from:
            seen_prompts = set(self._load_checkpoint(resume_from))
            self._checkpoint_path = resume_from
        elif cfg.resume:
            # 自动查找最新 checkpoint
            candidates = sorted(
                [f for f in os.listdir(out_dir) if f.endswith("_auto_distill.jsonl")],
                reverse=True,
            )
            if candidates:
                ckpt = os.path.join(out_dir, candidates[0])
                seen_prompts = set(self._load_checkpoint(ckpt))
                self._checkpoint_path = ckpt
                resume_from = ckpt

        self._log(f"一键蒸馏启动 | 模型池:{len(self.model_pool)} | 轮次:{'无限' if infinite else max_runs}", log_path)
        if resume_from:
            self._log(f"续跑: {resume_from} (跳过 {len(seen_prompts)} 条)", log_path)

        run_idx = 0
        consecutive_fail = 0

        while infinite or run_idx < max_runs:
            if self._stop:
                self._log("停止信号", log_path)
                break

            run_idx += 1
            now = time.strftime("%H:%M:%S")
            self._log(f"\n=== 第 {run_idx} 轮 ({now}) ===", log_path)
            print(f"\n── 第 {run_idx} 轮 ──")

            ok = False
            engine = None

            for attempt in range(len(self.model_pool)):
                if self._stop: break
                mi = self.model_pool[(self.pointer + attempt) % len(self.model_pool)]
                mn = mi["id"]

                # 跳过已知不可用模型 (蒸馏后标记的)
                if mn in self._dead_models:
                    continue

                print(f"  ▶ [{attempt+1}/{len(self.model_pool)}] {mn}")
                self._log(f"尝试: {mn}", log_path)

                if engine is None:
                    engine = DistillEngine(cfg)
                engine.config.model = mn

                # V2.0: 传入 resume_from 让 DistillEngine 内部跳过已完成 prompt
                data = engine.run(resume_from=resume_from or None, seen_prompts=seen_prompts)
                if data:
                    ok = True
                    self.total_success += len(data)
                    cumulative.extend(data)
                    # 增量保存 checkpoint
                    if self._checkpoint_path:
                        self._save_checkpoint(self._checkpoint_path, data)
                    else:
                        with open(cumu_path, "a", encoding="utf-8") as f:
                            for r in data:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    self._log(f"✓ {mn}: {len(data)} 条", log_path)
                    consecutive_fail = 0
                    # 清除 dead list (模型可能恢复)
                    self._dead_models.discard(mn)
                    break
                else:
                    self._dead_models.add(mn)
                    self._log(f"✗ {mn}: 返回空, 标记不可用", log_path)

            if not ok:
                consecutive_fail += 1
                if len(self._dead_models) >= len(self.model_pool):
                    self._dead_models.clear()  # 全部不可用，重置尝试
                wait = min(60 * consecutive_fail, 300)
                msg = f"全部失败 (连续 {consecutive_fail} 轮), 等待 {wait}s..."
                print(f"  ⚠ {msg}"); self._log(msg, log_path)
                time.sleep(wait)
            else:
                self.pointer = (self.pointer + 1) % len(self.model_pool)

            print(f"  累计成功: {self.total_success} 条")

        self._log(f"\n蒸馏结束 | 累计: {self.total_success} 条", log_path)
        print(f"\n完成! 累计: {self.total_success} 条\n输出: {cumu_path}")
        return cumulative

    def _log(self, msg, path):
        print(msg)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


__all__ = ["Pipeline"]
