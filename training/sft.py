import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any

from model import EmindLM, EmindConfig, create_model
from training.config import TrainingConfig
from training.trainer import TrainerBase


class SFTDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_seq_len: int = 2048, pad_token_id: int = 0):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.examples = []
        for item in data:
            if isinstance(item, str):
                encoded = tokenizer.encode(item)
                input_ids = encoded[:max_seq_len]
                labels = input_ids.copy()
                loss_mask = [1] * len(input_ids)
            else:
                prompt = item.get("prompt", "")
                response = item.get("response", "")
                full_text = prompt + response
                encoded = tokenizer.encode(full_text)
                input_ids = encoded[:max_seq_len]
                labels = input_ids.copy()
                prompt_len = len(tokenizer.encode(prompt))
                loss_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
                if len(loss_mask) < max_seq_len:
                    loss_mask = loss_mask + [0] * (max_seq_len - len(loss_mask))

            pad_len = max_seq_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
                loss_mask = loss_mask + [0] * pad_len

            self.examples.append({
                "input_ids": torch.tensor(input_ids[:max_seq_len], dtype=torch.long),
                "labels": torch.tensor(labels[:max_seq_len], dtype=torch.long),
                "loss_mask": torch.tensor(loss_mask[:max_seq_len], dtype=torch.float),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SFTTrainer(TrainerBase):
    def compute_loss(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        loss_mask = batch.get("loss_mask")
        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)

        _, logits, _, _, aux_loss, _ = self.model(input_ids)
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
            loss = loss.sum() / (shift_mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        loss = loss + aux_loss

        return loss
