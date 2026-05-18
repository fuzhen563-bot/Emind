"""
zhengliu — 蒸馏工具箱
用 Teacher 模型自动生成 SFT 训练数据，打包为 JSONL。

用法:
    python -m zhengliu.distill --teacher deepseek --api-key sk-xxx --code 50
    python -m zhengliu.distill --teacher ollama --model qwen2.5:7b --reasoning 20
"""

__all__ = [
    "DistillConfig", "parse_args",
    "DistillEngine",
    "SEEDS", "VOCAB", "fill", "generate_prompts",
]


def __getattr__(name):
    import importlib
    _LAZY = {
        "DistillConfig": ("zhengliu.config", "DistillConfig"),
        "parse_args": ("zhengliu.config", "parse_args"),
        "DistillEngine": ("zhengliu.distill", "DistillEngine"),
        "SEEDS": ("zhengliu.seeds", "SEEDS"),
        "VOCAB": ("zhengliu.seeds", "VOCAB"),
        "fill": ("zhengliu.seeds", "fill"),
        "generate_prompts": ("zhengliu.seeds", "generate_prompts"),
    }
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'zhengliu' has no attribute '{name}'")
