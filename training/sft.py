import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any

from model import EmindLM, EmindConfig, create_model
from training.config import TrainingConfig
from training.trainer import TrainerBase


class SFTDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_seq_len: int = 2048, pad_token_id: int = 0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if isinstance(item, str):
            encoded = self.tokenizer.encode(item)
            input_ids = encoded[:self.max_seq_len]
            labels = input_ids.copy()
            loss_mask = [1] * len(input_ids)
        else:
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            full_text = prompt + response
            encoded = self.tokenizer.encode(full_text)
            input_ids = encoded[:self.max_seq_len]
            labels = input_ids.copy()
            prompt_len = len(self.tokenizer.encode(prompt))
            loss_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
            if len(loss_mask) < self.max_seq_len:
                loss_mask = loss_mask + [0] * (self.max_seq_len - len(loss_mask))

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids[:self.max_seq_len], dtype=torch.long),
            "labels": torch.tensor(labels[:self.max_seq_len], dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask[:self.max_seq_len], dtype=torch.float),
        }


class SFTTrainer(TrainerBase):
    def compute_loss(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss_mask = batch.get("loss_mask")

        _, logits, _ = self.model(input_ids)
        vocab_size = logits.shape[-1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )

        if loss_mask is not None:
            shift_mask = loss_mask[..., 1:].contiguous().view(-1)
            loss = loss * shift_mask
            return loss.sum() / (shift_mask.sum() + 1e-8)

        return loss.mean()
