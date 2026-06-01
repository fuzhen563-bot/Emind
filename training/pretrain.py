"""预训练模块：纯文本 next-token prediction

支持将任意纯文本语料转换为所有 token 参与 loss 计算的预训练样本。
与 SFTDataset 的关键区别：labels = input_ids 右移一位，全部 token 都计算 loss。
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Any

from .trainer import TrainerBase


class PretrainDataset(Dataset):
    """纯文本 → 所有 token 参与 next-token prediction 的预训练数据集

    支持两种输入格式:
        - 纯字符串列表: ["文本1", "文本2", ...]
        - JSONL 格式:   [{"text": "文本1"}, {"text": "文本2"}, ...]

    长文本自动按 max_seq_len 滑动窗口切分为多个样本（不重叠）。
    labels 预右移一位，Trainer 端无需再次 shift。
    """

    def __init__(self, data: List, tokenizer, max_seq_len: int = 2048,
                 pad_token_id: int = 0):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.examples = []

        for item in data:
            # 兼容 str 和 {"text": "..."} 两种格式
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text", "")
            else:
                continue

            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) < 2:
                continue

            # 滑动窗口切分长文本，步长 = max_seq_len（不重叠）
            for i in range(0, len(ids) - 1, max_seq_len):
                chunk = ids[i:i + max_seq_len + 1]
                # input_ids:  前 N 个 token
                input_ids = chunk[:-1]
                # labels:     后 N 个 token（等价于右移一位，用于 next-token prediction）
                labels = chunk[1:]

                # padding 到固定长度
                pad_len = max_seq_len - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len  # -100 被 CE ignore_index 跳过

                self.examples.append({
                    "input_ids": torch.tensor(input_ids[:max_seq_len], dtype=torch.long),
                    "labels": torch.tensor(labels[:max_seq_len], dtype=torch.long),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class PretrainTrainer(TrainerBase):
    """预训练 Trainer：所有 token（除 padding 外）都参与交叉熵损失计算

    与 SFTTrainer 的区别:
        - SFTTrainer 只计算 assistant token 的 loss（通过 loss_mask 过滤）
        - PretrainTrainer 计算全部 token 的 loss（padding 位通过 ignore_index=-100 跳过）
    """

    def compute_loss(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        _, logits, _, _, aux_loss, _ = self.model(input_ids)
        vocab_size = logits.shape[-1]

        # 标准 next-token prediction CE loss
        # PretrainDataset 已预右移 labels，此处直接逐位置计算即可
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )

        return loss + aux_loss
