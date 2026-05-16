"""
Emind 使用示例
快速开始指南
"""

import torch
from Emind import create_model, SimpleTokenizer, get_model_config


def example_basic():
    """基本使用示例"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)
    
    # 获取配置
    config = get_model_config("1b")
    print(f"模型配置: {config['name']}")
    print(f"  d_model: {config['d_model']}")
    print(f"  n_layers: {config['n_layers']}")
    print(f"  n_heads: {config['n_heads']}")
    
    # 创建模型
    model = create_model(config)
    print(f"\n模型创建成功!")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    loss, logits = model(input_ids, labels=input_ids)
    print(f"损失: {loss.item():.4f}")
    print(f"输出形状: {logits.shape}")


def example_train():
    """训练示例"""
    print("\n" + "=" * 60)
    print("示例2: 模型训练")
    print("=" * 60)
    
    # 代码见 trainer.py
    print("""
# 训练步骤:
1. 准备训练数据 (文本文件，每行一条)
2. 创建数据加载器
3. 初始化模型和优化器
4. 训练循环:
   - 前向传播计算损失
   - 反向传播更新参数
   - 定期保存模型

# 训练命令:
python trainer.py
""")


def example_inference():
    """推理示例"""
    print("\n" + "=" * 60)
    print("示例3: 文本生成")
    print("=" * 60)
    
    print("""
# 推理步骤:
1. 加载训练好的模型
2. 编码输入文本
3. 自回归生成
4. 解码输出

# 推理命令:
python inference.py --prompt "深度学习是"
python inference.py --interactive
""")


def example_different_sizes():
    """不同规模模型示例"""
    print("\n" + "=" * 60)
    print("示例4: 不同规模模型")
    print("=" * 60)
    
    sizes = ["small", "tiny", "1b", "3b", "7b"]
    
    for size in sizes:
        config = get_model_config(size)
        from Emind.config import estimate_params
        params = estimate_params(config)
        print(f"  {size}: {params:.2f}B 参数")


if __name__ == "__main__":
    example_basic()
    example_train()
    example_inference()
    example_different_sizes()
    
    print("\n" + "=" * 60)
    print("更多内容请查看 README.md")
    print("=" * 60)
