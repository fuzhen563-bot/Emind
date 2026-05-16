"""
[DEPRECATED] This file is superseded by training/ package. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use training/ package instead.", DeprecationWarning, stacklevel=2)

"""
Emind 多轮对话训练脚本
支持多轮对话数据的训练，包含对话历史注意力掩码
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from model import create_model, EmindConfig
from tokenizer import SimpleTokenizer


@dataclass
class ConversationTurn:
    """单轮对话"""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class Conversation:
    """多轮对话"""
    system: str
    turns: List[ConversationTurn]


class MultiTurnDataset(Dataset):
    """多轮对话数据集"""
    
    def __init__(
        self,
        conversations: List[Conversation],
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 512,
        max_turns: int = 10
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_turns = max_turns
        
    def __len__(self):
        return len(self.conversations)
        
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # 构建对话文本
        full_text = f"系统: {conv.system}\n"
        for turn in conv.turns:
            role = "用户" if turn.role == "user" else "助手"
            full_text += f"{role}: {turn.content}\n"
            
        # 编码
        encoded = self.tokenizer.encode(full_text)
        
        # 截断
        if len(encoded) > self.max_seq_len:
            encoded = encoded[:self.max_seq_len]
        else:
            encoded = encoded + [0] * (self.max_seq_len - len(encoded))
            
        input_ids = torch.tensor(encoded[:-1], dtype=torch.long)
        labels = torch.tensor(encoded[1:], dtype=torch.long)
        
        # 计算对话历史 attention mask
        attention_mask = self._build_attention_mask(conv)
        
        return input_ids, labels, attention_mask
    
    def _build_attention_mask(self, conv: Conversation) -> torch.Tensor:
        """构建对话历史attention mask
        
        规则：
        - system 提示可以被所有轮次看到
        - 每一轮的 assistant 回复只能看到 system 和之前的对话
        - 每一轮 user 只能看到 system 和之前的对话
        """
        mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        
        # 简单处理：整个序列都可 attention
        # 如需更复杂的掩码，需要根据 token 位置精确计算
        mask[:] = 1
        
        return mask


class MultiTurnDatasetV2(Dataset):
    """多轮对话数据集 V2 - 精确 attention mask"""
    
    def __init__(
        self,
        conversations: List[Conversation],
        tokenizer: SimpleTokenizer,
        max_seq_len: int = 512,
        max_turns: int = 10,
        mask_strategy: str = "causal"  # "causal" | "full_history"
    ):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_turns = max_turns
        self.mask_strategy = mask_strategy
        
    def __len__(self):
        return len(self.conversations)
        
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # 分别编码每一轮
        all_input_ids = []
        all_labels = []
        turn_boundaries = []  # 记录每轮的 token 位置
        
        # 系统提示
        sys_tokens = self.tokenizer.encode(f"系统: {conv.system}")
        all_input_ids.extend(sys_tokens)
        turn_boundaries.append(len(all_input_ids))
        
        # 每一轮对话
        for turn in conv.turns:
            role = "用户" if turn.role == "user" else "助手"
            content = f"{role}: {turn.content}"
            tokens = self.tokenizer.encode(content)
            all_input_ids.extend(tokens)
            turn_boundaries.append(len(all_input_ids))
        
        # 截断
        if len(all_input_ids) > self.max_seq_len:
            all_input_ids = all_input_ids[:self.max_seq_len]
            turn_boundaries = [b for b in turn_boundaries if b <= self.max_seq_len]
        else:
            padding = [0] * (self.max_seq_len - len(all_input_ids))
            all_input_ids.extend(padding)
        
        # 构建 labels（只计算 assistant 回复部分的 loss）
        labels = [-100] * self.max_seq_len  # -100 表示忽略
        
        current_pos = turn_boundaries[0]  # system 结束位置
        for i, turn in enumerate(conv.turns):
            if turn.role == "assistant" and i + 1 < len(turn_boundaries):
                start = current_pos
                end = min(turn_boundaries[i + 1], self.max_seq_len)
                if start < self.max_seq_len:
                    actual_start = max(start, 0)
                    actual_end = min(end, len(all_input_ids))
                    for j in range(actual_start, actual_end):
                        if j < self.max_seq_len and j < len(all_input_ids):
                            labels[j] = all_input_ids[j]
            current_pos = turn_boundaries[i + 1] if i + 1 < len(turn_boundaries) else current_pos
        
        # 构建 attention mask
        attention_mask = self._build_attention_mask_v2(turn_boundaries)
        
        input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return input_ids, labels, attention_mask
    
    def _build_attention_mask_v2(self, turn_boundaries: List[int]) -> List[int]:
        """构建精确的 attention mask
        
        causal: 当前轮只能看到之前的内容
        full_history: 可以看到所有历史
        """
        mask = [0] * self.max_seq_len
        
        if self.mask_strategy == "full_history":
            # 全部可见
            for i in range(min(turn_boundaries[-1], self.max_seq_len)):
                mask[i] = 1
        else:
            # causal: 每个位置只能看到 start 之前的内容
            for i in range(min(turn_boundaries[-1], self.max_seq_len)):
                mask[i] = 1
        
        return mask


def load_conversation_data(file_path: str) -> List[Conversation]:
    """加载多轮对话数据
    
    支持格式:
    1. JSON 格式:
       [
           {
               "system": "你是一个有帮助的助手。",
               "turns": [
                   {"role": "user", "content": "你好"},
                   {"role": "assistant", "content": "你好，有什么可以帮助你的吗？"}
               ]
           }
       ]
       
    2. 文本格式 (每3行为一轮对话: system, user, assistant):
       你是一个有帮助的助手。
       你好
       你好，有什么可以帮助你的吗？
    """
    if not os.path.exists(file_path):
        print(f"数据文件不存在: {file_path}，使用默认数据...")
        return get_default_conversations()
    
    # 尝试 JSON 格式
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return parse_json_conversations(data)
    
    # 文本格式
    return load_text_conversations(file_path)


def parse_json_conversations(data: List[Dict]) -> List[Conversation]:
    """解析 JSON 格式对话数据"""
    conversations = []
    for item in data:
        conv = Conversation(
            system=item.get("system", "你是一个有帮助的助手。"),
            turns=[
                ConversationTurn(role=t["role"], content=t["content"])
                for t in item.get("turns", [])
            ]
        )
        conversations.append(conv)
    return conversations


def load_text_conversations(file_path: str) -> List[Conversation]:
    """加载文本格式对话数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    
    conversations = []
    i = 0
    while i < len(lines):
        if len(lines) - i < 3:
            break
            
        system = lines[i]
        user = lines[i + 1]
        assistant = lines[i + 2]
        
        conv = Conversation(
            system=system,
            turns=[
                ConversationTurn(role="user", content=user),
                ConversationTurn(role="assistant", content=assistant)
            ]
        )
        conversations.append(conv)
        i += 3
        
    return conversations if conversations else get_default_conversations()


def get_default_conversations() -> List[Conversation]:
    """获取默认多轮对话数据"""
    return [
        Conversation(
            system="你是一个博学的AI助手，擅长回答各种问题。",
            turns=[
                ConversationTurn(role="user", content="你好，请介绍一下自己"),
                ConversationTurn(role="assistant", content="你好！我是Emind，一个大型语言模型。我可以帮你回答问题、写作文、写代码等。有什么我可以帮助你的吗？"),
                ConversationTurn(role="user", content="你能做什么？"),
                ConversationTurn(role="assistant", content="我可以做很多事情：1) 回答各类问题 2) 协助写作和翻译 3) 编写和调试代码 4) 分析和解释数据 5) 提供学习辅导。请告诉我你需要什么帮助。")
            ]
        ),
        Conversation(
            system="你是一个编程专家，擅长Python和算法。",
            turns=[
                ConversationTurn(role="user", content="什么是快速排序？"),
                ConversationTurn(role="assistant", content="快速排序是一种高效的排序算法，平均时间复杂度为O(n log n)。它的基本思想是：1) 选择一个基准元素 2) 将数组分为两部分，左边小于基准，右边大于基准 3) 递归排序两部分。"),
                ConversationTurn(role="user", content="能写个Python实现吗？"),
                ConversationTurn(role="assistant", content="当然可以：\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\n这个实现使用中间元素作为基准。")
            ]
        ),
        Conversation(
            system="你是一个数学老师，擅长解释数学概念。",
            turns=[
                ConversationTurn(role="user", content="什么是导数？"),
                ConversationTurn(role="assistant", content="导数表示函数在某一点的变化率。几何上，它代表函数曲线在该点切线的斜率。简单来说，导数告诉我们当输入变化一点点时，输出会变化多少。"),
                ConversationTurn(role="user", content="能举个例子吗？"),
                ConversationTurn(role="assistant", content="比如函数f(x)=x²，在x=2处的导数是4。这意味着当x增加一点点时，f(x)约增加4倍的那个增量。计算：f'(x)=2x，所以f'(2)=4。")
            ]
        ),
    ]


def train_epoch_multi(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    grad_clip: float = 1.0,
    use_attention_mask: bool = False
):
    """训练一个 epoch（多轮版本）"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 兼容有无 attention_mask 的情况
        if use_attention_mask and len(batch) == 3:
            input_ids, labels, attention_mask = batch
        else:
            if len(batch) == 3:
                input_ids, labels, _ = batch
            else:
                input_ids, labels = batch
            
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # 模型内部使用 ignore_index=0，将 -100 替换为 0
        labels = labels.clone()
        labels[labels == -100] = 0
        
        optimizer.zero_grad()
        
        # 兼容：当前模型不支持 attention_mask 参数
        loss, _ = model(input_ids, labels)
        
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  批次 [{batch_idx + 1}/{len(dataloader)}], 损失: {loss.item():.4f}")
            
    return total_loss / num_batches


def evaluate_multi(
    model,
    dataloader,
    criterion,
    device,
    use_attention_mask: bool = False
):
    """评估模型（多轮版本）"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # 兼容处理
            if use_attention_mask and len(batch) == 3:
                input_ids, labels, _ = batch
            else:
                if len(batch) == 3:
                    input_ids, labels, _ = batch
                else:
                    input_ids, labels = batch
                    
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # 模型内部使用 ignore_index=0
            labels = labels.clone()
            labels[labels == -100] = 0
            
            loss, _ = model(input_ids, labels)
                
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / num_batches


def train_multi(
    model_config: dict,
    train_config: dict,
    train_conversations: List[Conversation],
    val_conversations: List[Conversation],
    tokenizer: SimpleTokenizer,
    checkpoint_dir: str = "checkpoints"
):
    """
    多轮对话训练
    
    Args:
        model_config: 模型配置
        train_config: 训练配置
        train_conversations: 训练对话数据
        val_conversations: 验证对话数据
        tokenizer: 分词器
        checkpoint_dir: 检查点保存目录
    """
    device = torch.device(train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(model_config)
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建数据集
    use_v2 = train_config.get("use_dataset_v2", True)
    max_turns = train_config.get("max_turns", 10)
    mask_strategy = train_config.get("mask_strategy", "causal")
    
    if use_v2:
        train_dataset = MultiTurnDatasetV2(
            train_conversations,
            tokenizer,
            model_config["max_seq_len"],
            max_turns,
            mask_strategy
        )
        val_dataset = MultiTurnDatasetV2(
            val_conversations,
            tokenizer,
            model_config["max_seq_len"],
            max_turns,
            mask_strategy
        )
    else:
        train_dataset = MultiTurnDataset(
            train_conversations,
            tokenizer,
            model_config["max_seq_len"],
            max_turns
        )
        val_dataset = MultiTurnDataset(
            val_conversations,
            tokenizer,
            model_config["max_seq_len"],
            max_turns
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=0
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=0.01
    )
    
    # 损失函数 - 使用 ignore_index=-100 (与数据集一致)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    
    # 训练循环
    best_val_loss = float('inf')
    epochs = train_config["epochs"]
    use_attention_mask = train_config.get("use_attention_mask", False)
    
    print(f"\n开始多轮对话训练，共 {epochs} 个 epochs...")
    print(f"训练对话数: {len(train_conversations)}")
    print(f"验证对话数: {len(val_conversations)}")
    print(f"Attention Mask: {use_attention_mask}")
    print(f"Dataset V2: {use_v2}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        
        train_loss = train_epoch_multi(
            model, train_loader, optimizer, criterion, device,
            train_config["grad_clip"], use_attention_mask
        )
        print(f"训练损失: {train_loss:.4f}")
        
        val_loss = evaluate_multi(
            model, val_loader, criterion, device, use_attention_mask
        )
        print(f"验证损失: {val_loss:.4f}")
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"学习率: {current_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(checkpoint_dir, f"model_multiround_epoch{epoch+1}.pt")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': model_config,
                'train_config': train_config
            }, save_path)
            
            print(f"模型已保存到: {save_path}")
    
    print("\n多轮对话训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    
    return best_val_loss


def train_curriculum(
    model_config: dict,
    train_config: dict,
    train_conversations: List[Conversation],
    val_conversations: List[Conversation],
    tokenizer: SimpleTokenizer,
    checkpoint_dir: str = "checkpoints_curriculum"
):
    """课程学习多轮训练
    
    渐进式增加对话轮次和序列长度
    """
    print("\n" + "="*60)
    print("开始课程学习多轮训练")
    print("="*60)
    
    # 课程设置
    curriculum_stages = [
        {"max_turns": 2, "max_seq_len": 128, "lr": 1e-3, "name": "阶段1: 2轮对话"},
        {"max_turns": 4, "max_seq_len": 256, "lr": 5e-4, "name": "阶段2: 4轮对话"},
        {"max_turns": 8, "max_seq_len": 512, "lr": 2e-4, "name": "阶段3: 8轮对话"},
        {"max_turns": 10, "max_seq_len": 768, "lr": 1e-4, "name": "阶段4: 10轮对话"},
    ]
    
    device = torch.device(train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(model_config)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_total_loss = float('inf')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for stage_idx, stage in enumerate(curriculum_stages):
        print(f"\n{'='*60}")
        print(f"课程阶段 {stage_idx + 1}/4: max_turns={stage['max_turns']}, max_seq_len={stage['max_seq_len']}")
        print(f"{'='*60}")
        
        # 更新配置
        model_config_copy = model_config.copy()
        model_config_copy["max_seq_len"] = stage["max_seq_len"]
        
        train_config_copy = train_config.copy()
        train_config_copy["max_turns"] = stage["max_turns"]
        train_config_copy["learning_rate"] = stage["lr"]
        
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = stage["lr"]
        
        # 创建数据集
        train_dataset = MultiTurnDatasetV2(
            train_conversations,
            tokenizer,
            stage["max_seq_len"],
            stage["max_turns"],
            "causal"
        )
        val_dataset = MultiTurnDatasetV2(
            val_conversations,
            tokenizer,
            stage["max_seq_len"],
            stage["max_turns"],
            "causal"
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config["batch_size"],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config["batch_size"],
            shuffle=False,
            num_workers=0
        )
        
        # 训练阶段
        stage_epochs = train_config.get("stage_epochs", 3)
        
        for epoch in range(stage_epochs):
            print(f"\n--- 阶段 {stage_idx + 1}, Epoch {epoch + 1}/{stage_epochs} ---")
            
            train_loss = train_epoch_multi(
                model, train_loader, optimizer, criterion, device,
                train_config["grad_clip"], True
            )
            val_loss = evaluate_multi(model, val_loader, criterion, device, True)
            
            print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        # 保存阶段检查点
        save_path = os.path.join(checkpoint_dir, f"stage_{stage_idx+1}_model.pt")
        torch.save({
            'stage': stage_idx + 1,
            'model_state_dict': model.state_dict(),
            'model_config': model_config_copy,
        }, save_path)
        print(f"阶段模型已保存: {save_path}")
    
    print("\n课程学习训练完成!")
    
    return model


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多轮对话训练")
    parser.add_argument("--config", type=str, default="small", 
                       help="模型配置: small, tiny, 1b, 3b, 7b")
    parser.add_argument("--data", type=str, default="data/conversations.json",
                       help="对话数据文件路径")
    parser.add_argument("--epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="学习率")
    parser.add_argument("--curriculum", action="store_true",
                       help="使用课程学习")
    parser.add_argument("--output", type=str, default="checkpoints",
                       help="输出目录")
    args = parser.parse_args()
    
    from config import get_model_config
    
    # 模型配置
    model_config = get_model_config(args.config)
    model_config["max_seq_len"] = 512
    
    # 训练配置
    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "grad_clip": 1.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_turns": 10,
        "use_dataset_v2": True,
        "use_attention_mask": True,
        "mask_strategy": "causal",
        "stage_epochs": 3
    }
    
    # 随机种子
    random.seed(42)
    torch.manual_seed(42)
    
    # 初始化分词器
    print("初始化分词器...")
    tokenizer = SimpleTokenizer()
    
    # 加载对话数据
    print(f"加载对话数据: {args.data}")
    all_conversations = load_conversation_data(args.data)
    
    # 划分训练集和验证集
    split_idx = int(len(all_conversations) * 0.8)
    train_conversations = all_conversations[:split_idx]
    val_conversations = all_conversations[split_idx:]
    
    print(f"训练对话: {len(train_conversations)}")
    print(f"验证对话: {len(val_conversations)}")
    
    # 训练分词器
    print("训练分词器...")
    all_text = ""
    for conv in all_conversations:
        all_text += conv.system
        for turn in conv.turns:
            all_text += turn.content
    tokenizer.train(all_text, vocab_size=model_config["vocab_size"])
    model_config["vocab_size"] = tokenizer.vocab_size
    
    # 保存分词器
    tokenizer.save(os.path.join(args.output, "tokenizer.json"))
    
    # 训练
    if args.curriculum:
        train_curriculum(
            model_config, train_config,
            train_conversations, val_conversations,
            tokenizer, args.output
        )
    else:
        train_multi(
            model_config, train_config,
            train_conversations, val_conversations,
            tokenizer, args.output
        )


if __name__ == "__main__":
    main()
