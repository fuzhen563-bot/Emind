"""
蒸馏配置
"""
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class DistillConfig:
    teacher: str = "deepseek"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    identity: str = "你是一个名为 Emind·智脑 的 AI 助手，由亦梓科技开发。"
    type_counts: Dict[str, int] = field(default_factory=lambda: {"code": 20, "reasoning": 10})
    max_tokens: int = 2048
    temperature: float = 0.7
    workers: int = 5
    output_dir: str = "zhengliu/output"
    no_cot: bool = False
    dry_run: bool = False

    BACKEND_MAP = {
        "deepseek": "cloud_api", "openai": "cloud_api",
        "ollama": "ollama", "vllm": "vllm",
        "huggingface": "huggingface", "local": "local",
    }
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


def parse_args(argv=None) -> DistillConfig:
    parser = argparse.ArgumentParser(
        prog="zhengliu",
        description="蒸馏 — 用 Teacher 模型生成 SFT 数据并打包为 JSONL",
    )
    # Teacher 后端
    parser.add_argument("--teacher", default="deepseek",
                        choices=["deepseek", "openai", "ollama", "vllm", "huggingface", "local"],
                        help="Teacher 模型后端")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--model", default=None, help="模型名或路径")
    # 身份设定
    parser.add_argument("--identity", default=None,
                        help="Teacher 的身份提示（默认: 你是一个名为 Emind·智脑 的 AI 助手，由亦梓科技开发。）")
    # 数据量
    parser.add_argument("--code", type=int, default=0, help="代码数据条数")
    parser.add_argument("--reasoning", type=int, default=0, help="推理数据条数")
    parser.add_argument("--deep-reasoning", type=int, default=0, help="深度推理数据条数")
    parser.add_argument("--anti-hallucination", type=int, default=0, help="反幻觉数据条数")
    parser.add_argument("--identity-data", type=int, default=0, help="身份认知数据条数")
    parser.add_argument("--all", type=int, default=0, help="每种类型各 N 条（快捷方式）")
    # 生成参数
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=5, help="并行线程数")
    parser.add_argument("--output-dir", default="zhengliu/output", help="输出目录")
    parser.add_argument("--no-cot", action="store_true", help="禁用 CoT 策略")
    parser.add_argument("--dry-run", action="store_true", help="仅预览生成的 prompt，不调用 API")

    args = parser.parse_args(argv)

    type_counts = {}
    if args.all:
        for t in ("code", "reasoning", "deep_reasoning", "anti_hallucination", "identity"):
            type_counts[t] = args.all
    else:
        pairs = [
            ("code", "code"), ("reasoning", "reasoning"),
            ("deep_reasoning", "deep_reasoning"),
            ("anti_hallucination", "anti_hallucination"),
            ("identity_data", "identity"),
        ]
        for attr, key in pairs:
            v = getattr(args, attr, 0)
            if v:
                type_counts[key] = v

    if not type_counts:
        type_counts = {"code": 20, "reasoning": 10}

    return DistillConfig(
        teacher=args.teacher,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        identity=args.identity or "你是一个名为 Emind·智脑 的 AI 助手，由亦梓科技开发。",
        type_counts=type_counts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        workers=args.workers,
        output_dir=args.output_dir,
        no_cot=args.no_cot,
        dry_run=args.dry_run,
    )
