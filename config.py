"""
[DEPRECATED] This file is superseded by training/config.py and model.py EmindConfig. Use those instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/config.py and model.py EmindConfig instead.", DeprecationWarning, stacklevel=2)

# ============================================================
# 预设模型配置 - 精确目标参数量
# ============================================================

# 小型模型 (~10M = 0.1亿)
CONFIG_SMALL = {
    "name": "small",
    "vocab_size": 5000,
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 6,
    "d_ff": 512,
    "max_seq_len": 128,
    "dropout": 0.1
}

# 微型模型 (~100M = 1亿)
CONFIG_TINY = {
    "name": "tiny",
    "vocab_size": 10000,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 12,
    "d_ff": 2048,
    "max_seq_len": 256,
    "dropout": 0.1
}

# 1B 参数模型 - 精确目标: 10亿参数 (1B)
# 计算: vocab*d_model + n_layers*(4*d_model^2 + 2*d_model*d_ff)
# 嵌入与输出共享权重，只计算一次
# 25000*2048 + 16*(4*2048^2 + 2*2048*4096) ≈ 0.95B
CONFIG_1B = {
    "name": "1b",
    "vocab_size": 25000,
    "d_model": 2048,
    "n_heads": 16,
    "n_layers": 16,
    "d_ff": 4096,
    "max_seq_len": 2048,
    "dropout": 0.1,
    "use_rotary_embedding": True
}

# 3B 参数模型 - 精确目标: 30亿参数 (3B)
# 30000*2560 + 24*(4*2560^2 + 2*2560*6800) ≈ 2.9B
CONFIG_3B = {
    "name": "3b",
    "vocab_size": 30000,
    "d_model": 2560,
    "n_heads": 20,
    "n_layers": 24,
    "d_ff": 6800,
    "max_seq_len": 2048,
    "dropout": 0.1,
    "use_rotary_embedding": True
}

# 7B 参数模型 - 精确目标: 70亿参数 (7B)
# 类似LLaMA-7B
CONFIG_7B = {
    "name": "7b",
    "vocab_size": 32000,
    "d_model": 4096,
    "n_heads": 32,
    "n_layers": 28,
    "d_ff": 11008,
    "max_seq_len": 4096,
    "dropout": 0.1,
    "use_rotary_embedding": True
}


# ============================================================
# 默认配置 - 推荐使用更大模型
# ============================================================

CONFIG = CONFIG_TINY  # 100M 参数模型

TRAIN_CONFIG = {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 2e-4,
    "min_lr_ratio": 0.1,
    "grad_clip": 1.0,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 500,
    "weight_decay": 0.1,
    "train_data_path": "data/yu.jsonl",
    "model_save_path": "checkpoints/model.pt",
    "eval_interval": 100,
    "save_interval": 500,
    "seed": 42,
    "device": "cuda"
}


# ============================================================
# 辅助函数
# ============================================================

def get_model_config(size: str = "1b") -> dict:
    """获取模型配置
    
    Args:
        size: 模型规模，可选 "small", "tiny", "1b", "3b", "7b"
        
    Returns:
        模型配置字典
    """
    configs = {
        "small": CONFIG_SMALL,
        "tiny": CONFIG_TINY,
        "1b": CONFIG_1B,
        "3b": CONFIG_3B,
        "7b": CONFIG_7B
    }
    return configs.get(size, CONFIG_1B)


def estimate_params(config: dict) -> float:
    """精确估算参数量（单位：亿）
    
    Transformer语言模型参数计算:
    - 词嵌入: vocab_size × d_model (与输出共享)
    - 每层: QKV(3×d_model²) + O(d_model²) + FFN(2×d_model×d_ff)
    
    Args:
        config: 模型配置字典
        
    Returns:
        参数量（单位：亿）
    """
    vocab_size = int(config["vocab_size"])
    d_model = int(config["d_model"])
    n_layers = int(config["n_layers"])
    d_ff = int(config["d_ff"])
    
    # 词嵌入 (与lm_head共享)
    embed = vocab_size * d_model
    
    # 每层参数
    # QKV: 3 × d_model × d_model
    # O projection: d_model × d_model  
    # FFN: 2 × d_model × d_ff
    per_layer = 4 * d_model * d_model + 2 * d_model * d_ff
    
    # 总计
    total = embed + n_layers * per_layer
    
    return total / 1e8


def main():
    """测试各配置"""
    print("\n" + "="*70)
    print("  Emind 大模型 - 参数规模配置")
    print("="*70)
    
    configs = [
        ("小型模型 (学习用)", "small", CONFIG_SMALL),
        ("微型模型 (实验用)", "tiny", CONFIG_TINY),
        ("1B 模型 (10亿参数)", "1b", CONFIG_1B),
        ("3B 模型 (30亿参数)", "3b", CONFIG_3B),
        ("7B 模型 (70亿参数)", "7b", CONFIG_7B)
    ]
    
    for name, size, config in configs:
        params = estimate_params(config)
        print(f"\n>>> {name}")
        print(f"    参数量: {params:.2f}B (约 {params*10:.0f}亿)")
        print(f"    词汇表: {config['vocab_size']:,}")
        print(f"    隐藏维度: {config['d_model']}")
        print(f"    层数: {config['n_layers']}")
        print(f"    注意力头: {config['n_heads']}")
        print(f"    前馈维度: {config['d_ff']}")
    
    print("\n" + "="*70)
    print("  使用方法:")
    print("  from config import get_model_config")
    print("  ")
    print("  # 获取1B模型配置")
    print("  config = get_model_config('1b')")
    print("  ")
    print("  # 获取7B模型配置")  
    print("  config = get_model_config('7b')")
    print("="*70)


if __name__ == "__main__":
    main()
