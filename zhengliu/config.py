"""
蒸馏配置 — 完整参数定义和 CLI 解析
"""
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class DistillConfig:
    # === Teacher 连接 ===
    teacher: str = "deepseek"
    api_key: str = ""
    base_url: str = ""
    model: str = ""

    # === 身份设定 ===
    identity: str = "你是一个名为 Emind·智脑 的 AI 助手，由亦梓科技开发。"

    # === 数据参数 ===
    type_counts: Dict[str, int] = field(default_factory=dict)

    # === 生成参数 ===
    max_tokens: int = 2048
    temperature: float = 0.7
    workers: int = 5

    # === 输出 ===
    output_dir: str = "zhengliu/output"

    # === 蒸馏模式 ===
    mode: str = "sft"            # sft / dpo
    no_cot: bool = False
    cot_depth: int = 1           # 1/2/3
    dry_run: bool = False

    # === 质量控制 ===
    quality_check: bool = False
    min_response_length: int = 20
    min_quality_score: float = 0.3
    multi_turn_correct: bool = False
    semantic_quality: bool = False
    quality_review: bool = False

    # === DPO ===
    dpo_chosen_temp: float = 0.3
    dpo_rejected_temp: float = 1.2

    # === Pipeline 自动运行 ===
    auto_runs: int = 0           # 0=禁用 / -1=无限 / N=固定轮次
    auto_models: str = ""        # 手动模型池, 逗号分隔
    auto_discover: bool = True   # 自动发现 API 可用模型
    exclude_models: str = ""     # 排除模型, 逗号分隔

    # === 断点续蒸馏 ===
    resume: bool = False

    # === 可视化 ===
    no_visual: bool = False

    # ========== 计算属性 ==========
    BACKEND_MAP = {"deepseek": "cloud_api", "openai": "cloud_api", "ollama": "ollama",
                   "vllm": "vllm", "huggingface": "huggingface", "local": "local"}
    MODEL_MAP = {"deepseek": "deepseek-v4-flash"}
    BASE_URL_MAP = {"deepseek": "https://api.deepseek.com"}

    @property
    def backend_type(self) -> str:
        return self.BACKEND_MAP.get(self.teacher, "cloud_api")

    @property
    def model_name(self) -> str:
        return self.model or self.MODEL_MAP.get(self.teacher, "gpt-4o-mini")

    @property
    def resolved_base_url(self) -> str:
        return self.base_url or self.BASE_URL_MAP.get(self.teacher, "")

    @property
    def resolved_api_key(self) -> str:
        return self.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""

    @property
    def total_attempted(self) -> int:
        return sum(self.type_counts.values())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zhengliu",
        description="蒸馏 — 用 Teacher 模型自动生成 SFT/DPO 训练数据")

    # Teacher
    g = p.add_argument_group("Teacher 连接")
    g.add_argument("--teacher", default="deepseek", help="后端 (deepseek/openai/ollama/vllm/local, 或自定义)")
    g.add_argument("--api-key", default="", help="API Key")
    g.add_argument("--base-url", default="", help="API 地址")
    g.add_argument("--model", default="", help="模型名")

    # 身份
    p.add_argument("--identity", default="", help="Teacher 身份设定")

    # 数据量
    g = p.add_argument_group("数据配比")
    g.add_argument("--code", type=int, default=0)
    g.add_argument("--reasoning", type=int, default=0)
    g.add_argument("--deep-reasoning", type=int, default=0)
    g.add_argument("--anti-hallucination", type=int, default=0)
    g.add_argument("--identity-data", type=int, default=0)
    g.add_argument("--noise-code", type=int, default=0, help="噪声化代码数据 (真实用户口吻)")
    g.add_argument("--failure-path", type=int, default=0, help="真实失败→修正路径")
    g.add_argument("--uncertainty", type=int, default=0, help="不确定性校准")
    g.add_argument("--long-conversation", type=int, default=0, help="长程递归对话(5+轮)")
    g.add_argument("--all", type=int, default=0, help="每种类型各 N 条 (快捷)")

    # 生成参数
    g = p.add_argument_group("生成参数")
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--workers", type=int, default=5)
    g.add_argument("--output-dir", default="zhengliu/output")

    # 模式
    g = p.add_argument_group("蒸馏模式")
    g.add_argument("--mode", default="sft", choices=["sft", "dpo"], help="sft=标准 / dpo=偏好对")
    g.add_argument("--no-cot", action="store_true", help="禁用 CoT")
    g.add_argument("--cot-depth", type=int, default=1, choices=[1, 2, 3], help="CoT 深度 1/2/3")
    g.add_argument("--dry-run", action="store_true", help="只预览 prompt")

    # 质量控制
    g = p.add_argument_group("质量控制")
    g.add_argument("--quality-check", action="store_true", help="启用质量评分过滤")
    g.add_argument("--min-response-length", type=int, default=20)
    g.add_argument("--min-quality-score", type=float, default=0.3)
    g.add_argument("--multi-turn-correct", action="store_true", help="Teacher 自我纠错 (使用 critic 审查询问)")
    g.add_argument("--semantic-quality", action="store_true", help="语义质量评分 (额外 API 调用, 评估逻辑一致性)")
    g.add_argument("--quality-review", action="store_true", help="质量审查 + 自动修正管线 (额外 API 调用)")

    # DPO
    g = p.add_argument_group("DPO 参数")
    g.add_argument("--dpo-chosen-temp", type=float, default=0.3)
    g.add_argument("--dpo-rejected-temp", type=float, default=1.2)

    # Pipeline
    g = p.add_argument_group("自动运行 (Pipeline)")
    g.add_argument("--auto-runs", type=int, default=0, help="-1=无限 / 0=禁用 / N=固定轮次")
    g.add_argument("--auto-models", default="", help="手动模型池 (逗号分隔)")
    g.add_argument("--no-discover", action="store_false", dest="discover", help="关闭自动发现模型")
    g.add_argument("--exclude-models", default="", help="排除模型 (逗号分隔)")

    # 断点续蒸馏
    p.add_argument("--resume", action="store_true", help="从上次 checkpoint 续蒸馏")

    # 可视化
    p.add_argument("--no-visual", action="store_true", help="纯文本进度")

    return p


def parse_args(argv=None) -> DistillConfig:
    p = build_parser()
    args = p.parse_args(argv)

    type_counts = {}
    if args.all:
        for t in ("code", "reasoning", "deep_reasoning", "anti_hallucination", "identity"):
            type_counts[t] = args.all
    else:
        pairs = [("code","code"), ("reasoning","reasoning"), ("deep_reasoning","deep_reasoning"),
                 ("anti_hallucination","anti_hallucination"), ("identity_data","identity"),
                 ("noise_code","noise_code"), ("failure_path","failure_path"),
                 ("uncertainty","uncertainty"), ("long_conversation","long_conversation")]
        for attr, key in pairs:
            v = getattr(args, attr, 0)
            if v:
                type_counts[key] = v

    if not type_counts:
        type_counts = {"code": 20, "reasoning": 10}

    return DistillConfig(
        teacher=args.teacher, api_key=args.api_key, base_url=args.base_url, model=args.model,
        identity=args.identity or "你是一个名为 Emind·智脑 的 AI 助手，由亦梓科技开发。",
        type_counts=type_counts,
        max_tokens=args.max_tokens, temperature=args.temperature, workers=args.workers,
        output_dir=args.output_dir,
        mode=args.mode, no_cot=args.no_cot, cot_depth=args.cot_depth, dry_run=args.dry_run,
        quality_check=args.quality_check, min_response_length=args.min_response_length,
        min_quality_score=args.min_quality_score,         multi_turn_correct=args.multi_turn_correct,
        semantic_quality=args.semantic_quality,
        quality_review=args.quality_review,
        dpo_chosen_temp=args.dpo_chosen_temp, dpo_rejected_temp=args.dpo_rejected_temp,
        auto_runs=args.auto_runs, auto_models=args.auto_models, auto_discover=args.discover,
        exclude_models=args.exclude_models, resume=args.resume, no_visual=args.no_visual,
    )
