"""
zhengliu — 蒸馏工具箱 v2.0
用 Teacher 模型自动生成 SFT/DPO 训练数据

用法:
    python -m zhengliu                        交互菜单
    python -m zhengliu --code 200 ...         直接蒸馏
"""

__all__ = [
    "DistillConfig", "parse_args",
    "DistillEngine", "Pipeline",
    "SEEDS", "VOCAB", "fill", "generate_prompts",
]


def __getattr__(name):
    import importlib
    _LAZY = {
        "DistillConfig": ("zhengliu.config", "DistillConfig"),
        "parse_args": ("zhengliu.config", "parse_args"),
        "DistillEngine": ("zhengliu.distill", "DistillEngine"),
        "Pipeline": ("zhengliu.pipeline", "Pipeline"),
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
