"""
[DEPRECATED] This file is superseded by training/trainer.py. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/trainer.py instead.", DeprecationWarning, stacklevel=2)

"""
Emind 大模型训练脚本 - 精度优化版
包含：混合精度训练、早停、学习率预热、梯度累积
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import random
import os
from pathlib import Path
import math

from model import create_model, EmindConfig
from tokenizer import SimpleTokenizer


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, data, tokenizer, max_seq_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # 编码文本
        encoded = self.tokenizer.encode(text)
        
        # 截断或填充到固定长度
        if len(encoded) > self.max_seq_len:
            encoded = encoded[:self.max_seq_len]
        else:
            # 填充
            encoded = encoded + [0] * (self.max_seq_len - len(encoded))
            
        # 输入和目标（目标比输入少一个token，用于预测下一个token）
        input_ids = torch.tensor(encoded[:-1], dtype=torch.long)
        labels = torch.tensor(encoded[1:], dtype=torch.long)
        
        return input_ids, labels


def load_data(file_path: str, train_ratio: float = 0.9):
    """
    加载训练数据
    
    Args:
        file_path: 数据文件路径
        train_ratio: 训练集比例
        
    Returns:
        train_data, val_data: 训练和验证数据
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"数据文件不存在: {file_path}")
        print("使用默认训练数据...")
        
        # 默认训练数据（中文示例）
        default_data = [
            "深度学习是人工智能的核心技术。",
            "神经网络模型在自然语言处理中应用广泛。",
            "机器学习算法可以自动从数据中学习规律。",
            "Transformer 架构改变了序列建模的方式。",
            "注意力机制让模型关注重要的信息。",
            "预训练模型大幅提升了下游任务性能。",
            "大规模语言模型展现出惊人的能力。",
            "提示工程是使用大模型的关键技巧。",
            "模型量化可以减少推理计算量。",
            "分布式训练加速了大模型的训练过程。",
            "GPU 是深度学习的主要硬件平台。",
            "Python 是深度学习领域最流行的编程语言。",
            "PyTorch 提供了灵活的动态计算图。",
            "TensorFlow 支持静态和动态图模式。",
            "数据清洗是机器学习的重要步骤。",
            "特征工程对模型性能影响很大。",
            "过拟合是训练深度模型常见的问题。",
            "正则化技术可以防止过拟合。",
            "Dropout 是一种有效的正则化方法。",
            "批量归一化加速了网络训练。",
            "学习率调度可以改善收敛效果。",
            "优化器决定了参数更新的方式。",
            "Adam 是最常用的优化器之一。",
            "梯度裁剪防止了梯度爆炸问题。",
            "早停策略可以避免过拟合。",
            "交叉验证评估了模型的泛化能力。",
            "混淆矩阵展示了分类结果。",
            "精确率和召回率衡量了模型性能。",
            "F1 分数综合考虑了精确率和召回率。",
            "AUC 是评估排序模型的重要指标。",
        ]
        
        # 扩展数据（重复并添加变化）
        data = []
        for _ in range(100):
            for text in default_data:
                data.append(text)
                
        return data[:int(len(data) * train_ratio)], data[int(len(data) * train_ratio):]
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 清理数据
    data = [line.strip() for line in lines if line.strip()]
    
    # 划分训练集和验证集
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    return train_data, val_data


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip: float = 1.0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        loss, _ = model(input_ids, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  批次 [{batch_idx + 1}/{len(dataloader)}], 损失: {loss.item():.4f}")
            
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            loss, _ = model(input_ids, labels)
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / num_batches


def train_with_accuracy_improvements(
    model_config: dict,
    train_config: dict,
    train_data: list,
    val_data: list,
    tokenizer: SimpleTokenizer
):
    """
    训练模型 - 精度优化版
    
    优化策略：
    1. 学习率预热 (Warmup)
    2. 余弦退火学习率
    3. 梯度累积
    4. 早停 (Early Stopping)
    5. 混合精度训练 (FP16)
    6. 更强的权重衰减
    """
    # 设置设备
    device = torch.device(train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"使用设备: {device}")

    # 检查 CUDA 和 FP16 支持
    use_amp = device.type == "cuda" and train_config.get("use_amp", True)
    print(f"混合精度训练: {'开启' if use_amp else '关闭'}")

    # 创建模型
    model = create_model(model_config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")

    # 创建数据集
    train_dataset = TextDataset(train_data, tokenizer, model_config["max_seq_len"])
    val_dataset = TextDataset(val_data, tokenizer, model_config["max_seq_len"])

    # 使用更大的 batch_size（配合梯度累积）
    actual_batch_size = train_config.get("batch_size", 16)
    grad_accum_steps = train_config.get("gradient_accumulation_steps", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 优化器 - 使用 AdamW with decoupled weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config.get("weight_decay", 0.1),
        betas=(0.9, 0.95)
    )

    # 学习率预热 + 余弦退火
    num_epochs = train_config["epochs"]
    num_warmup_steps = train_config.get("warmup_steps", 500)
    total_steps = len(train_loader) * num_epochs

    def lr_lambda(step):
        """学习率调度函数"""
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        else:
            min_lr_ratio = train_config.get("min_lr_ratio", 0.1)
            progress = float(step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # 早停配置
    patience = train_config.get("patience", 10)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # 训练循环
    print(f"\n开始训练，共 {num_epochs} 个 epochs...")
    print(f"实际 batch_size: {actual_batch_size * grad_accum_steps} (累积步数: {grad_accum_steps})")
    print(f"学习率: {train_config['learning_rate']}, 预热步数: {num_warmup_steps}")

    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        # 训练
        train_loss = train_epoch_advanced(
            model, train_loader, optimizer, scheduler,
            criterion, device, train_config["grad_clip"],
            grad_accum_steps, scaler, use_amp
        )
        print(f"训练损失: {train_loss:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"验证损失: {val_loss:.4f}")

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_path = train_config["model_save_path"]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': model_config
            }, save_path)
            print(f"✓ 模型已保存到: {save_path}")
        else:
            epochs_without_improvement += 1
            print(f"验证损失未改善 ({epochs_without_improvement}/{patience})")

            if epochs_without_improvement >= patience:
                print(f"\n早停触发！最佳验证损失: {best_val_loss:.4f}")
                break

    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")


def train_epoch_advanced(model, dataloader, optimizer, scheduler, criterion, device,
                        grad_clip: float = 1.0, grad_accum_steps: int = 4,
                        scaler=None, use_amp: bool = False):
    """高级训练 epoch - 支持混合精度和梯度累积"""
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 混合精度前向传播
        if use_amp:
            with torch.cuda.amp.autocast():
                loss, _ = model(input_ids, labels)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            loss, _ = model(input_ids, labels)
            loss = loss / grad_accum_steps
            loss.backward()

        # 梯度累积
        if (batch_idx + 1) % grad_accum_steps == 0:
            # 梯度裁剪
            if grad_clip > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # 更新参数
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # 更新学习率
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        if (batch_idx + 1) % 20 == 0:
            print(f"  批次 [{batch_idx + 1}/{len(dataloader)}], 损失: {loss.item() * grad_accum_steps:.4f}")

    return total_loss / num_batches


# 保留原有函数（兼容旧代码）
def train(
    model_config: dict,
    train_config: dict,
    train_data: list,
    val_data: list,
    tokenizer: SimpleTokenizer
):
    """兼容旧接口的训练函数"""
    # 设置默认优化参数
    train_config.setdefault("use_amp", True)
    train_config.setdefault("weight_decay", 0.1)
    train_config.setdefault("min_lr_ratio", 0.1)
    train_config.setdefault("patience", 10)
    train_config.setdefault("warmup_steps", 500)

    train_with_accuracy_improvements(model_config, train_config, train_data, val_data, tokenizer)


def main():
    """主函数 - 使用优化配置"""
    # 模型配置 - 推荐使用更大模型以提高精度
    model_config = {
        "vocab_size": 10000,
        "d_model": 512,       # 从 256 提升到 512
        "n_heads": 8,
        "n_layers": 12,       # 从 6 提升到 12
        "d_ff": 2048,         # 从 512 提升到 2048
        "max_seq_len": 256,   # 从 128 提升到 256
        "dropout": 0.1
    }

    # 训练配置 - 优化版
    train_config = {
        "epochs": 50,         # 增加训练轮数
        "batch_size": 32,
        "learning_rate": 2e-4,  # 优化学习率
        "grad_clip": 1.0,
        "use_amp": True,      # 混合精度
        "weight_decay": 0.1,  # 正则化
        "min_lr_ratio": 0.1,
        "warmup_steps": 500,  # 学习率预热
        "patience": 10,       # 早停耐心值
        "gradient_accumulation_steps": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_save_path": "checkpoints/model.pt",
        "seed": 42
    }

    # 设置随机种子
    random.seed(train_config["seed"])
    torch.manual_seed(train_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config["seed"])

    # 创建分词器并训练
    print("初始化分词器...")
    tokenizer = SimpleTokenizer()

    # 加载数据
    train_data, val_data = load_data(train_config.get("train_data_path", "data/train.txt"))

    # 如果数据太少，生成更多高质量数据
    if len(train_data) < 1000:
        print("数据量较少，生成增强训练数据...")
        train_data = generate_enhanced_data(train_data)

    print(f"训练数据: {len(train_data)} 条")
    print(f"验证数据: {len(val_data)} 条")

    # 训练分词器
    all_data = train_data + val_data
    combined_text = "".join(all_data)
    tokenizer.train(combined_text, vocab_size=model_config["vocab_size"])

    # 更新模型配置的 vocab_size
    model_config["vocab_size"] = tokenizer.vocab_size

    # 保存分词器
    tokenizer.save("checkpoints/tokenizer.json")

    # 计算参数量
    param_count = (model_config["vocab_size"] * model_config["d_model"] +
                   model_config["n_layers"] * (4 * model_config["d_model"]**2 +
                   2 * model_config["d_model"] * model_config["d_ff"]))
    print(f"模型参数量: {param_count / 1e6:.2f}M")

    # 训练模型
    train(model_config, train_config, train_data, val_data, tokenizer)


def generate_enhanced_data(base_data: list) -> list:
    """生成增强训练数据"""
    import re

    # 领域特定模板
    templates = {
        "技术": [
            "人工智能技术正在改变{行业}的发展格局。",
            "深度学习在{行业}领域展现出强大的应用潜力。",
            "Transformer模型架构革新了{行业}的技术方案。",
            "大语言模型为{行业}带来了全新的智能解决方案。",
            "{行业}的数字化转型离不开AI技术的支撑。",
        ],
        "业务": [
            "我们的{产品}旨在解决{问题}的核心痛点。",
            "通过{方法}，企业可以显著提升{指标}。",
            "{产品}采用先进的算法，确保{优势}。",
            "客户反馈显示，使用{产品}后{收益}提升了30%。",
            "我们的服务覆盖{场景}等多种业务需求。",
        ],
        "服务": [
            "提供7×24小时的{服务}专业支持。",
            "{服务}团队由经验丰富的{角色}组成。",
            "我们致力于为客户提供最优质的{服务}体验。",
            "{服务}流程经过严格优化，确保高效交付。",
            "专业的{服务}体系帮助企业实现目标。",
        ]
    }

    # 领域词汇
    industries = ["医疗", "教育", "金融", "零售", "制造", "物流", "科技"]
    products = ["智能客服", "数据分析平台", "自动化工具", "决策系统"]
    problems = ["效率低下", "成本过高", "用户体验差", "响应速度慢"]
    metrics = ["运营效率", "客户满意度", "转化率", "决策准确率"]
    methods = ["流程优化", "AI自动化", "智能分析", "预测模型"]
    advantages = ["高准确性", "低延迟", "高可用性", "安全性"]
    benefits = ["效率", "收益", "客户留存", "转化"]
    scenes = ["客户服务", "风险控制", "精准营销", "运营管理"]
    services = ["技术支持", "咨询", "定制开发", "培训"]
    roles = ["专家", "工程师", "分析师", "顾问"]

    enhanced_data = list(base_data)

    # 每个模板生成20条变体
    for category, patterns in templates.items():
        for pattern in patterns:
            for _ in range(20):
                # 随机填充
                text = pattern
                replacements = {
                    "{行业}": random.choice(industries),
                    "{产品}": random.choice(products),
                    "{问题}": random.choice(problems),
                    "{指标}": random.choice(metrics),
                    "{方法}": random.choice(methods),
                    "{优势}": random.choice(advantages),
                    "{收益}": random.choice(benefits),
                    "{场景}": random.choice(scenes),
                    "{服务}": random.choice(services),
                    "{角色}": random.choice(roles),
                }
                for k, v in replacements.items():
                    text = text.replace(k, v)

                # 扩展句子
                extended = f"{text} 这种方案具有显著的优势，可以有效提升整体性能和用户体验。"
                enhanced_data.append(extended)

    return enhanced_data


if __name__ == "__main__":
    main()
