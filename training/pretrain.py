"""预训练模块：纯文本 next-token prediction

支持将任意纯文本语料转换为所有 token 参与 loss 计算的预训练样本。
与 SFTDataset 的关键区别：labels = input_ids 右移一位，全部 token 都计算 loss。

Sequence Packing 优化（v2.1）：
  将多条短句首尾相连打包到 max_seq_len 长度，消除 padding 浪费。
  短句占优数据（如 MiniMind 平均 52 tokens）可提升约 39 倍吞吐。
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple
from training.trainer import TrainerBase


class PretrainDataset(Dataset):
    """纯文本 → 所有 token 参与 next-token prediction 的预训练数据集

    支持两种输入格式:
        - 纯字符串列表: ["文本1", "文本2", ...]
        - JSONL 格式:   [{"text": "文本1"}, {"text": "文本2"}, ...]

    长文本自动按 max_seq_len 滑动窗口切分为多个样本（不重叠）。
    labels 预右移一位，Trainer 端无需再次 shift。

    支持 Sequence Packing（use_packing=True）：
        将多条短句无缝拼接为一个 max_seq_len 样本，消除 padding 浪费。
        适用于短句为主的数据集（如百科、新闻语料），可大幅提升训练吞吐。
    """

    def __init__(self, data: List, tokenizer, max_seq_len: int = 2048,
                 pad_token_id: int = 0, use_packing: bool = True):
        """
        Args:
            data: 文本数据，支持 str list 或 [{"text": "..."}] JSONL 格式
            tokenizer: EmindTokenizer 实例
            max_seq_len: 最大序列长度
            pad_token_id: padding token ID
            use_packing: 是否启用 Sequence Packing
                         True → 打包多条短句，消除 padding（推荐，提速 10-40x）
                         False → 每条样本独立 padding（兼容旧行为）
        """
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.use_packing = use_packing

        if use_packing:
            self.examples = self._build_packed(data, tokenizer, max_seq_len)
        else:
            self.examples = self._build_padded(data, tokenizer, max_seq_len, pad_token_id)

    # ==================== 原始方案：每条独立 padding ====================

    @staticmethod
    def _build_padded(data: List, tokenizer, max_seq_len: int,
                      pad_token_id: int) -> List[Dict]:
        """原始方案：每条样本独立 padding"""
        examples = []
        for item in data:
            text = item if isinstance(item, str) else item.get("text", "")
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) < 2:
                continue

            # 滑动窗口切分长文本
            for i in range(0, len(ids) - 1, max_seq_len):
                chunk = ids[i:i + max_seq_len + 1]
                input_ids = chunk[:-1]
                labels = chunk[1:]

                # padding
                pad_len = max_seq_len - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [pad_token_id] * pad_len
                    labels = labels + [-100] * pad_len

                examples.append({
                    "input_ids": torch.tensor(input_ids[:max_seq_len], dtype=torch.long),
                    "labels": torch.tensor(labels[:max_seq_len], dtype=torch.long),
                })
        return examples

    # ==================== Pack 优化方案 ====================

    @staticmethod
    def _build_packed(data: List, tokenizer, max_seq_len: int) -> List[Dict]:
        """Sequence Packing：将多条短句首尾相连拼满 max_seq_len

        流程:
        1. 将所有文本 tokenize 为 id 列表
        2. 拼接 ids 列表（EOS 作为句子分隔）
        3. 按 max_seq_len+1 滑动窗口切块（+1 是为了右移 labels）
        4. 每个块为 [input_ids[:-1], labels[1:]]

        对比:
        - _build_padded:  每条样本 [52 tokens + 1996 pad] → 3.27M 样本
        - _build_packed:  每条样本 [~2048 tokens]        → ~85k 样本
        """
        # 1) 将所有文本 tokenize 为 id 流
        all_ids: List[int] = []
        for item in data:
            text = item if isinstance(item, str) else item.get("text", "")
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) >= 2:
                all_ids.extend(ids)

        if not all_ids:
            # 兜底：数据为空时生成假数据
            all_ids = [1, 2]  # BOS, EOS

        # 2) 按 max_seq_len + 1 滑动窗口切分（+1 为右移 labels）
        #    步长 = max_seq_len，不重叠
        examples = []
        for i in range(0, len(all_ids) - 1, max_seq_len):
            chunk = all_ids[i:i + max_seq_len + 1]
            if len(chunk) < 2:
                continue

            input_ids = chunk[:-1]
            labels = chunk[1:]

            # 最后一个 chunk 可能不足 max_seq_len，padding
            pad_len = max_seq_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [0] * pad_len          # pad_token_id=0
                labels = labels + [-100] * pad_len

            examples.append({
                "input_ids": torch.tensor(input_ids[:max_seq_len], dtype=torch.long),
                "labels": torch.tensor(labels[:max_seq_len], dtype=torch.long),
            })

        return examples

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
