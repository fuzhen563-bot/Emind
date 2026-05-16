"""
[DEPRECATED] This file is superseded by unified_inference.py. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use unified_inference.py instead.", DeprecationWarning, stacklevel=2)

import torch
import argparse
import os

from model import create_model, EmindConfig
from tokenizer import SimpleTokenizer


def load_model(model_path: str, device: torch.device):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 设备
        
    Returns:
        model: 加载的模型
    """
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint.get('model_config', {})
    
    # 创建模型
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型已从: {model_path} 加载")
    print(f"训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"验证损失: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    return model


def load_tokenizer(tokenizer_path: str):
    """
    加载分词器
    
    Args:
        tokenizer_path: 分词器文件路径
        
    Returns:
        tokenizer: 加载的分词器
    """
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = None,
    device: torch.device = None
):
    """
    生成文本
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        prompt: 提示文本
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数（越高越随机）
        top_k: top-k 采样参数
        device: 设备
        
    Returns:
        generated_text: 生成的文本
    """
    model.eval()
    
    # 编码提示文本
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # 解码
    generated_text = tokenizer.decode(generated_ids[0].cpu().tolist())
    
    return generated_text


def interactive_mode(model, tokenizer, device):
    """
    交互模式
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        device: 设备
    """
    print("\n" + "=" * 50)
    print("Emind 交互式对话")
    print("=" * 50)
    print("输入文本开始生成，输入 'quit' 或 'exit' 退出")
    print("-" * 50)
    
    while True:
        try:
            # 获取用户输入
            prompt = input("\n用户: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break
                
            if not prompt:
                continue
                
            # 生成回复
            print("\nEmind: ", end="")
            response = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=100,
                temperature=0.8,
                device=device
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"错误: {e}")


def batch_generate(model, tokenizer, prompts: list, device, **kwargs):
    """
    批量生成
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        prompts: 提示列表
        device: 设备
        **kwargs: 生成参数
        
    Returns:
        results: 生成的文本列表
    """
    results = []
    
    for prompt in prompts:
        print(f"生成: {prompt}")
        generated = generate_text(model, tokenizer, prompt, device=device, **kwargs)
        results.append(generated)
        print(f"结果: {generated}\n")
        
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Emind 推理")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="checkpoints/model.pt", help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, default="checkpoints/tokenizer.json", help="分词器路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    # 生成参数
    parser.add_argument("--prompt", type=str, default="深度学习是", help="输入提示")
    parser.add_argument("--max_tokens", type=int, default=50, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--top_k", type=int, default=None, help="top-k 采样")
    
    # 模式
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--batch", nargs="+", help="批量生成")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        print("请先运行 trainer.py 训练模型")
        return
        
    # 加载模型
    model = load_model(args.model_path, device)
    
    # 加载分词器
    if not os.path.exists(args.tokenizer_path):
        print(f"错误: 分词器文件不存在: {args.tokenizer_path}")
        return
        
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # 根据模式运行
    if args.interactive:
        # 交互模式
        interactive_mode(model, tokenizer, device)
        
    elif args.batch:
        # 批量生成
        batch_generate(
            model,
            tokenizer,
            args.batch,
            device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
    else:
        # 单次生成
        print(f"\n提示: {args.prompt}")
        print("生成中...")
        
        generated = generate_text(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        print(f"\n生成结果: {generated}")


if __name__ == "__main__":
    main()
