"""
<<<<<<< HEAD
蒸馏引擎 — 基础蒸馏 + 进阶蒸馏 + 质量控制 + 多轮修正
=======
蒸馏引擎 — 连接 Teacher 模型 → 生成 SFT 数据 → 输出 JSONL
>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329
"""
import json
import os
import sys
import time
<<<<<<< HEAD
import hashlib
import math
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Optional
=======
>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

<<<<<<< HEAD
from zhengliu.config import DistillConfig
from zhengliu.seeds import generate_prompts, inject_reasoning_trace, generate_reasoning_graph


class DistillEngine:
    """蒸馏引擎 — 连接 Teacher, 生成 SFT/DPO 数据"""

=======
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from typing import Optional

from zhengliu.config import DistillConfig, parse_args
from zhengliu.seeds import generate_prompts


class DistillEngine:
>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329
    def __init__(self, config: DistillConfig):
        self.config = config
        self.teacher = None
        self._use_requests = False
<<<<<<< HEAD
        self._dashboard = None
        self._session = None        # HTTP 连接池 (惰性初始化)
        self._session_lock = None    # 线程安全锁
        if not config.dry_run:
            self._init_teacher()

    # ========== Teacher 连接 ==========

    def _init_teacher(self):
        bt = self.config.backend_type
        try:
            from unified_inference import UnifiedInferenceEngine, BackendConfig
        except ImportError:
            self._print("unified_inference 不可用, HTTP 直连模式")
            self._use_requests = True
            self._warmup_session()
            return
        c = BackendConfig(backend_type=bt, model_name=self.config.model_name,
                          base_url=self.config.resolved_base_url, api_key=self.config.resolved_api_key)
        self.teacher = UnifiedInferenceEngine(c)
        if self.teacher.is_available():
            self._print(f"Teacher 已就绪: {bt} / {c.model_name}")
        else:
            self._print(f"后端 {bt} 不可用, 回退 HTTP 直连")
            self._use_requests = True
            self._warmup_session()

    def _get_session(self):
        """线程安全的连接池 Session"""
        import requests
        import threading
        if self._session is None:
            if self._session_lock is None:
                self._session_lock = threading.Lock()
            with self._session_lock:
                if self._session is None:
                    s = requests.Session()
                    adapter = requests.adapters.HTTPAdapter(
                        pool_connections=self.config.workers + 5,
                        pool_maxsize=self.config.workers + 5,
                        max_retries=0,
                    )
                    s.mount("https://", adapter)
                    s.mount("http://", adapter)
                    self._session = s
        return self._session

    def _warmup_session(self):
        """预热连接 — 提前建立 TCP 连接"""
        try:
            s = self._get_session()
            url = f"{self.config.resolved_base_url.rstrip('/')}/v1/models"
            headers = {"Authorization": f"Bearer {self.config.resolved_api_key}"}
            s.head(url, headers=headers, timeout=5)
        except Exception:
            pass

    def _print(self, msg: str):
        """统一日志输出"""
        if self._dashboard:
            self._dashboard.update(event=msg)
        else:
            print(msg)

    # ========== API 调用 ==========

    def _call_api(self, prompt: str, temperature: Optional[float] = None) -> str:
        cfg = self.config
        session = self._get_session()
        headers = {"Content-Type": "application/json"}
        if cfg.resolved_api_key:
            headers["Authorization"] = f"Bearer {cfg.resolved_api_key}"
=======
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

>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329
        payload = {
            "model": cfg.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.max_tokens,
<<<<<<< HEAD
            "temperature": temperature if temperature is not None else cfg.temperature,
        }
        url = f"{cfg.resolved_base_url.rstrip('/')}/v1/chat/completions"

        delays = [0.5, 1.5, 4.0]
        for attempt, delay in enumerate(delays):
            try:
                resp = session.post(url, headers=headers, json=payload, timeout=90)
                if resp.status_code == 200:
                    d = resp.json()
                    return d["choices"][0]["message"]["content"]
                # 鉴权错误: 不重试
                if resp.status_code in (401, 403):
                    self._print(f"鉴权失败 HTTP {resp.status_code} — 跳过此模型")
                    return ""
                if resp.status_code == 400:
                    body = resp.text[:300]
                    self._print(f"请求错误 HTTP 400: {body}")
                    return ""  # 400 不重试
                if resp.status_code == 429:
                    time.sleep(delay * 2)
                    continue
                self._print(f"HTTP {resp.status_code} (尝试 {attempt+1}/{len(delays)})")
                if attempt < len(delays) - 1:
                    time.sleep(delay)
            except requests.exceptions.ConnectionError:
                self._print(f"连接失败 (尝试 {attempt+1}/{len(delays)})")
                time.sleep(delay)
            except requests.exceptions.Timeout:
                self._print(f"超时 (尝试 {attempt+1}/{len(delays)})")
                time.sleep(delay)
            except Exception as e:
                self._print(f"API异常 (尝试 {attempt+1}/{len(delays)}): {e}")
                time.sleep(delay)
        return ""

    def _call_teacher(self, prompt: str, temperature: Optional[float] = None) -> Optional[str]:
        temp = temperature if temperature is not None else self.config.temperature
        try:
            if self._use_requests:
                return self._call_api(prompt, temperature=temp)
            return self.teacher.generate(prompt, max_tokens=self.config.max_tokens, temperature=temp)
        except Exception as e:
            self._print(f"生成失败: {e}")
            return None

    # ========== Prompt 构建 ==========

    def _build_prompt(self, item: dict, use_cot: bool) -> str:
        """
        构建 prompt。
        核心原则：
        - identity 仅对 identity 类注入
        - 推理类使用 Thinking Styles 注入真实推理轨迹
        - 代码类无额外引导，让 seed 自身驱动思考
        - 不做 instruction engineering（不要"请体现"/"请展示"等）
        """
        prompt = item["prompt"]
        dtype = item.get("type", "")

        # identity 仅对 identity 类
        if dtype == "identity":
            identity = self.config.identity
            return f"{identity}\n\n{prompt}"

        # 反幻觉：不注入引导，seed 自身驱动
        if not use_cot:
            return prompt

        # 推理类 → 注入真实推理轨迹 (Thinking Styles 驱动)
        if dtype in ("reasoning", "deep_reasoning", "error_reasoning"):
            return inject_reasoning_trace(prompt, use_trace=True)

        return prompt

    # ========== 质量控制 ==========

    def _validate(self, response: Optional[str]) -> bool:
        if not response or len(response.strip()) < 10:
            return False
        r = response.strip()
        cfg = self.config
        if len(r) < cfg.min_response_length:
            return False
        if cfg.quality_check:
            score = self._quality_score(r)
            if score < cfg.min_quality_score:
                self._print(f"字符质量过滤: {score:.2f} < {cfg.min_quality_score}")
                return False
        return True

    def _semantic_quality(self, prompt: str, response: str) -> float:
        """
        P1: Critic 驱动的语义质量评分。
        不同于 _quality_score 的字符层过滤，这里让 Teacher 评估
        逻辑一致性、事实准确性、完整性和推理深度。
        返回 0.0~1.0。
        """
        sp = (
            f"评估以下回答的质量（0~10分）。\n\n"
            f"问题：{prompt[:300]}\n\n"
            f"回答：{response[:800]}\n\n"
            "请从以下四个维度分别打分（1~10），然后给出平均分：\n"
            "- 逻辑一致性：推理链是否无矛盾\n"
            "- 事实准确性：是否存在明显的事实错误\n"
            "- 完整性：是否覆盖了问题的关键方面\n"
            "- 边界处理：是否讨论了边界条件或特殊情况\n"
            "只输出一个0~10之间的数字，不要加任何解释。"
        )
        rating = self._call_teacher(sp, temperature=0.1)
        if not rating:
            return 0.5  # 默认中等
        try:
            # 提取第一个浮点数或整数
            import re
            nums = re.findall(r'\d+\.?\d*', rating.strip())
            if nums:
                score = float(nums[0]) / 10.0
                return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            pass
        return 0.5

    @staticmethod
    def _quality_score(text: str) -> float:
        """0~1 表面质量评分 (字符多样性 + 空行比 + 重复度)"""
        if len(text) < 50:
            return 0.0
        lines = text.split("\n")
        total = max(1, len(text))
        unique_ratio = len(set(text)) / min(2000, total)
        empty_ratio = sum(1 for l in lines if not l.strip()) / max(1, len(lines))
        n = max(1, len(text) - 3)
        repeat_ratio = len(set(text[i:i+3] for i in range(len(text)-3))) / n
        return max(0.0, min(1.0, unique_ratio * 0.3 + (1 - empty_ratio) * 0.3 + repeat_ratio * 0.4))

    @staticmethod
    def _ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
        """
        P1:         n-gram 余弦相似度。
        替代 MD5 字面去重 — 捕获语义相近的重复。
        """
        def _ngrams(t):
            t = t.lower().replace(" ", "")[:2000]
            return set(t[i:i+n] for i in range(len(t) - n + 1))

        a = _ngrams(text_a)
        b = _ngrams(text_b)
        if not a or not b:
            return 0.0
        intersection = a & b
        return len(intersection) / math.sqrt(len(a) * len(b))

    def _semantic_dedup(self, results: list, is_dpo: bool, threshold: float = 0.92) -> list:
        """
        P1: n-gram 语义去重。
        结合 MD5 精确去重 + n-gram 相似度模糊去重。
        threshold: 相似度 > 此值视为重复。
        """
        out = []
        seen_hashes = set()
        for r in results:
            if is_dpo:
                text = r["prompt"] + r["chosen"] + r.get("rejected", "")
            else:
                text = r["prompt"] + r["response"]

            # MD5 精确去重
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # n-gram 模糊去重 (仅对已有样本检查，避免 O(n²))
            is_dup = False
            for prev in out[-50:]:  # 只检查最近 50 条
                if is_dpo:
                    prev_text = prev["prompt"] + prev["chosen"] + prev.get("rejected", "")
                else:
                    prev_text = prev["prompt"] + prev["response"]
                if self._ngram_similarity(text, prev_text) > threshold:
                    is_dup = True
                    break
            if not is_dup:
                out.append(r)
        return out

    def _critic(self, prompt: str, response: str) -> Optional[str]:
        """
        Critic 模式 — 攻击性审查。
        不询问"是否正确"（self-bias），而是要求"找出至少一个问题"。
        如果找不到任何问题，返回 None（极少情况）。
        否则返回修正后的完整回答。
        """
        cp = (
            f"严格审查以下对问题 \"{prompt[:200]}\" 的回答：\n\n"
            f"{response[:1200]}\n\n"
            "请找出其中至少一个问题（推理漏洞、边界遗漏、错误复杂度、不完整分析、事实错误等）。\n"
            "如果确实找不到任何问题（极其罕见），请回复 'NO_ISSUES'。\n"
            "否则，给出修正后的完整回答。"
        )
        review = self._call_teacher(cp, temperature=0.3)
        if not review:
            return None
        s = review.strip()
        if s.upper().startswith("NO_ISSUES") and len(s) < 20:
            return None
        return review

    # ========== 质量审查 + 修正管线 ==========

    def _review_quality(self, prompt: str, response: str) -> dict:
        """
        单条数据质量审查。
        返回评分 + 问题列表 + 修正建议。
        """
        rp = (
            f"质量审查：\n\n问题：{prompt[:300]}\n\n回答：{response[:1000]}\n\n"
            "请输出 JSON（不要 markdown 代码块）：\n"
            "{\n"
            '  "score": 1-10,\n'
            '  "issues": ["问题1", "问题2"],\n'
            '  "pass": true/false,\n'
            '  "fixable": true/false\n'
            "}\n\n"
            "评分标准：\n"
            "- 推理链条是否完整无跳跃 (权重 3)\n"
            "- 是否有事实错误或幻觉 (权重 3)\n"
            "- 边界条件是否讨论 (权重 2)\n"
            "- 表达是否清晰自然 (权重 2)\n"
            "pass=false 表示严重缺陷需要丢弃。fixable=false 表示无法自动修复。"
        )
        review_raw = self._call_teacher(rp, temperature=0.1)
        if not review_raw:
            return {"score": 5, "issues": ["review_failed"], "pass": True, "fixable": True}

        try:
            import re
            clean = re.sub(r'```\w*\n?', '', review_raw.strip()).strip()
            result = json.loads(clean)
            result.setdefault("score", 5)
            result.setdefault("issues", [])
            result.setdefault("pass", result.get("score", 5) >= 4)
            result.setdefault("fixable", True)
            return result
        except json.JSONDecodeError:
            nums = re.findall(r'\d+', review_raw)
            score = int(nums[0]) if nums else 5
            return {"score": score, "issues": ["parse_error"], "pass": score >= 4, "fixable": True}

    def _auto_fix(self, prompt: str, response: str, issues: list) -> Optional[str]:
        """基于审查发现的问题，自动修正回答"""
        if not issues:
            return None
        fix_prompt = (
            f"以下回答存在这些问题：{', '.join(issues)}。\n\n"
            f"问题：{prompt[:300]}\n\n"
            f"原回答：{response[:1000]}\n\n"
            "请给出修正后的完整回答。只输出修正后的回答，不要解释修正了什么。"
        )
        fixed = self._call_teacher(fix_prompt, temperature=0.3)
        return fixed.strip() if fixed and len(fixed.strip()) > 20 else None

    def review_batch(self, results: list) -> tuple:
        """
        批量质量审查管线。
        返回 (passed, failed, report)。
        passed: 通过审查的数据
        failed: 未能通过但尝试修正的数据
        report: 审查报告
        """
        passed = []
        failed = []
        stats = {"total": len(results), "passed": 0, "failed": 0, "fixed": 0, "by_type": {}, "issues": {}}

        for i, r in enumerate(results):
            prompt = r.get("prompt", "")
            response = r.get("response", r.get("chosen", ""))
            dtype = r.get("type", "unknown")

            review = self._review_quality(prompt, response)
            score = review["score"]
            r["quality_score"] = score
            r["quality_issues"] = review.get("issues", [])

            # 统计
            stats["by_type"][dtype] = stats["by_type"].get(dtype, {"total": 0, "passed": 0, "avg_score": 0})
            stats["by_type"][dtype]["total"] += 1
            stats["by_type"][dtype]["avg_score"] = (
                (stats["by_type"][dtype]["avg_score"] * (stats["by_type"][dtype]["total"] - 1) + score)
                / stats["by_type"][dtype]["total"]
            )
            for iss in review.get("issues", []):
                stats["issues"][iss] = stats["issues"].get(iss, 0) + 1

            if review["pass"]:
                passed.append(r)
                stats["passed"] += 1
                stats["by_type"][dtype]["passed"] += 1
            elif review.get("fixable", False):
                fixed = self._auto_fix(prompt, response, review.get("issues", []))
                if fixed and self._validate(fixed):
                    r["response"] = fixed
                    r["quality_fixed"] = True
                    passed.append(r)
                    stats["fixed"] += 1
                    stats["passed"] += 1
                    stats["by_type"][dtype]["passed"] += 1
                else:
                    failed.append(r)
                    stats["failed"] += 1
            else:
                failed.append(r)
                stats["failed"] += 1

            if (i + 1) % 10 == 0:
                self._print(f"审查进度: {i+1}/{len(results)}  通过:{stats['passed']}  修正:{stats['fixed']}  失败:{stats['failed']}")

        return passed, failed, stats

    # ========== 核心蒸馏方法 ==========

    def generate_sft(self, item: dict, use_cot: bool) -> Optional[dict]:
        """基础蒸馏 — 生成单条 SFT 数据 (支持多轮对话 + 推理图)"""
        # 推理图模式：多阶段顺序蒸馏
        if item.get("type") == "reasoning_graph" and "stages" in item:
            return self._generate_reasoning_graph(item)

        # 多轮对话模式
        if item.get("type") == "conversation" and "turns" in item:
            return self._generate_conversation(item, use_cot)

        fp = self._build_prompt(item, use_cot)
        resp = self._call_teacher(fp)
        if not self._validate(resp):
            return None
        if self.config.multi_turn_correct:
            corrected = self._critic(fp[:200], resp)
            if corrected:
                resp = corrected
        if self.config.semantic_quality:
            sq = self._semantic_quality(item.get("prompt", ""), resp)
            if sq < self.config.min_quality_score:
                self._print(f"语义质量过滤: {sq:.2f} < {self.config.min_quality_score}")
                return None
        r = {"response": resp.strip(), "strategy": "cot" if use_cot else "direct", "source": "zhengliu"}
        r.update(item)
        return r

    def _generate_reasoning_graph(self, item: dict) -> Optional[dict]:
        """
        P2: 推理图蒸馏。
        多阶段顺序生成：观察→假设→验证→替代→比较→收敛。
        每个阶段基于前一个阶段的输出，形成完整的推理演化轨迹。
        """
        stages = item.get("stages", [])
        if not stages:
            return None

        base = item["prompt"]
        full_trace = []
        ctx = [f"问题：{base[:500]}"]

        for i, stage in enumerate(stages):
            stage_prompt = f"{stage['guide']}\n\n背景：\n" + "\n".join(ctx)
            resp = self._call_teacher(stage_prompt, temperature=0.7)
            if not resp or len(resp.strip()) < 10:
                break
            stage_text = resp.strip()
            full_trace.append(f"## {stage['stage']}\n{stage_text}")
            ctx.append(f"上一阶段({stage['stage']})分析：{stage_text[:400]}")

        if not full_trace:
            return None

        full_response = "\n\n".join(full_trace)
        r = {"response": full_response, "stages": [s["stage"] for s in stages],
             "strategy": "reasoning_graph", "source": "zhengliu"}
        r.update(item)
        return r

    def _generate_conversation(self, item: dict, use_cot: bool) -> Optional[dict]:
        """多轮对话蒸馏 — 逐轮生成，保留上下文"""
        turns = item.get("turns", [])
        if not turns:
            return None

        conversation = []

        for i, turn in enumerate(turns):
            if i == 0:
                fp = turn
            else:
                ctx = "\n".join(conversation)
                fp = f"对话历史：\n{ctx}\n\n请基于以上历史回答：{turn}"

            resp = self._call_teacher(fp)
            if not self._validate(resp):
                return None
            conversation.append(f"问：{turn}")
            conversation.append(f"答：{resp.strip()}")

        full_response = "\n".join(conversation)
        r = {"response": full_response, "turns": turns,
             "strategy": "multi_turn", "source": "zhengliu"}
        r.update(item)
        return r

    def generate_dpo(self, item: dict, use_cot: bool) -> Optional[dict]:
        """进阶蒸馏 — 生成 DPO 偏好对 (rejected=指定缺陷, 非高温)"""
        fp = self._build_prompt(item, use_cot)
        cfg = self.config

        chosen = self._call_teacher(fp, temperature=cfg.dpo_chosen_temp)
        if not self._validate(chosen):
            return None

        # "几乎正确"的 subtle flaws — DPO 最有效的 rejected
        flaw_prompts = [
            fp + "\n\n给出一个包含**一处微妙错误**的回答版本。这个错误应该：\n"
                 "- 不容易一眼看出来（比如忽略了一个不常见的边界条件）\n"
                 "- 或者复杂度分析在大 O 记号下看起来对但实际常数项有问题\n"
                 "- 或者推理链条整体正确但跳过了看似不重要的中间步骤\n"
                 "注意：回答看起来应该是认真的、完整的，不要明显出错。",

            fp + "\n\n给出一个在**一个隐含假设上就错了**的回答版本。\n"
                 "例如：假设输入已排序但实际未排序，假设数据量小但实际很大。\n"
                 "其他部分的分析和代码应该是正确的，只有这一个隐含假设导致了整个方案的偏差。\n"
                 "不要在回答中标注这是错误假设。",

            fp + "\n\n给出一个在**边界条件处理上不完整**的回答版本。\n"
                 "处理了常规情况，代码和分析都看起来不错，但：\n"
                 "- 空输入会崩溃，或者\n"
                 "- 重复元素会导致死循环，或者\n"
                 "- 极限值会溢出\n"
                 "选择一个不易察觉的边界问题。回答其他部分保持高质量。",
        ]
        import random as _rnd
        rejected = self._call_teacher(_rnd.choice(flaw_prompts), temperature=0.8)
        if not rejected or not self._validate(rejected):
            return None

        if chosen.strip() == rejected.strip():
            return None

        r = {"chosen": chosen.strip(), "rejected": rejected.strip(),
             "strategy": "cot" if use_cot else "direct", "source": "zhengliu"}
        r.update(item)
        return r

    # ========== 批量运行 ==========

    def run(self) -> list:
        """执行完整的蒸馏流程"""
=======
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
>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329
        cfg = self.config

        all_items = []
        for dtype, count in cfg.type_counts.items():
<<<<<<< HEAD
            for p in generate_prompts(dtype, count):
                if isinstance(p, str):
                    all_items.append({"prompt": p, "type": dtype})
                else:
                    item = {"type": dtype}; item.update(p); all_items.append(item)

        total = len(all_items)
        is_dpo = cfg.mode == "dpo"

        if cfg.dry_run:
            return self._dry_run(all_items, is_dpo)

        self._print(f"开始蒸馏: {total} 条, 模式={'DPO' if is_dpo else 'SFT'}")
        for dtype, c in cfg.type_counts.items():
            self._print(f"  {dtype}: {c}")

        self._maybe_start_dashboard(total, cfg)

        results = []
        done = success = fail = consec_fail = 0
        _FAIL_ABORT = max(10, total // 10)  # 连续失败超过 10% 则熔断

        def worker(item):
            use_cot = not cfg.no_cot and item.get("type") in ("reasoning", "deep_reasoning", "error_reasoning")
            if item.get("type") == "reasoning_graph":
                return self.generate_sft(item, use_cot)
            return self.generate_dpo(item, use_cot) if is_dpo else self.generate_sft(item, use_cot)
=======
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
>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329

        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = [pool.submit(worker, item) for item in all_items]
            for f in as_completed(futures):
                r = f.result()
<<<<<<< HEAD
                done += 1
                dtype = r.get("type", "?") if r else "?"
                if r:
                    results.append(r); success += 1; consec_fail = 0
                    # 传递 sample 预览和质量分到仪表盘
                    preview = r.get("response", r.get("chosen", ""))[:80].strip()
                    qs = r.get("quality_score", None)
                    self._update_progress(done, total, success, fail,
                                          by_type=dtype,
                                          sample={"type": dtype, "text": preview, "quality": qs},
                                          quality_score=qs)
                else:
                    fail += 1; consec_fail += 1
                    if consec_fail >= _FAIL_ABORT:
                        self._print(f"⚠ 连续失败 {consec_fail} 次, 熔断 — 模型不可用")
                        break
                    self._update_progress(done, total, success, fail)

        self._maybe_stop_dashboard()

        return self._semantic_dedup(results, is_dpo)

    def _dry_run(self, items, is_dpo):
        print("\n=== 预览模式 (--dry-run) ===")
        print(f"模式: {'DPO' if is_dpo else 'SFT'}")
        for i, item in enumerate(items, 1):
            print(f"\n--- [{item.get('type','?')}] Prompt {i} ---")
            print(item["prompt"][:500])
        print(f"\n共 {len(items)} 条 prompt, 未调用 API")
        return []

    def _maybe_start_dashboard(self, total, cfg=None):
        if not self.config.no_visual:
            try:
                from zhengliu.visual import DistillDashboard
                self._dashboard = DistillDashboard(total, self.config.type_counts)
                if cfg:
                    self._dashboard.set_mode("DPO" if cfg.mode == "dpo" else "SFT")
                    self._dashboard.set_model(cfg.model_name or cfg.teacher or "API")
                self._dashboard.start()
            except Exception:
                self._dashboard = None
                self._print(f"[{total} 条]")

    def _maybe_stop_dashboard(self):
        if self._dashboard:
            try:
                self._dashboard.finish()
            except Exception:
                pass
            self._dashboard = None

    def _update_progress(self, done, total, success, fail,
                          by_type=None, sample=None, quality_score=None):
        if self._dashboard:
            self._dashboard.update(completed=done, success=success, fail=fail,
                                    by_type={by_type: 1} if by_type else None,
                                    sample=sample, quality_score=quality_score)
        elif done % 10 == 0 or done == total:
            pct = done / total * 100
            bar = "█" * int(20 * done / total) + "░" * (20 - int(20 * done / total))
            print(f"  [{bar}] {done}/{total} ({pct:.0f}%) 成功:{success} 失败:{fail}")


def main():
    """CLI 入口 — python -m zhengliu.distill 或 zl"""
    from zhengliu.config import parse_args
=======
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
>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329
    cfg = parse_args()
    engine = DistillEngine(cfg)
    data = engine.run()

    if cfg.dry_run:
        return

<<<<<<< HEAD
    # 质量审查管线
    if cfg.quality_review and data:
        print("\n▶ 质量审查管线启动...")
        passed, failed, report = engine.review_batch(data)
        data = passed
        print(f"\n审查完成: 通过 {len(passed)} | 修正 {report['fixed']} | 丢弃 {len(failed)}")
        for iss, cnt in sorted(report.get("issues", {}).items(), key=lambda x: -x[1])[:5]:
            print(f"  问题-{iss}: {cnt} 次")

    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suffix = "dpo" if cfg.mode == "dpo" else "sft"
    out = os.path.join(cfg.output_dir, f"{ts}_distilled_{suffix}.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 输出失败样本
    if cfg.quality_review and failed:
        fail_out = os.path.join(cfg.output_dir, f"{ts}_failed.jsonl")
        with open(fail_out, "w", encoding="utf-8") as f:
            for r in failed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"失败样本: {fail_out}")

    total = cfg.total_attempted
    by_type = Counter(r["type"] for r in data)
    print(f"\n完成! {len(data)} 条 (尝试 {total}, 失败 {total-len(data)})")
    if data:
        print(f"输出: {out}")
        print(f"类型分布: {dict(by_type)}")
    else:
        print("建议: 1) python -m zhengliu --dry-run --code 5 预览")
        print("      2) 检查 --api-key / --base-url / --model")
=======
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
>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329


if __name__ == "__main__":
    main()
