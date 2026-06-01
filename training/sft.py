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
                prompt_len = 0
                loss_mask = [0] * len(input_ids)
            elif "messages" in item:
                prompt = ""
                response = ""
                for msg in item["messages"]:
                    if msg["role"] == "user":
                        prompt += msg.get("content", "")
                    elif msg["role"] == "assistant":
                        response += msg.get("content", "")
                full_text = prompt + response
                encoded = tokenizer.encode(full_text)
                input_ids = encoded[:max_seq_len]
                labels = input_ids.copy()
                prompt_len = len(tokenizer.encode(prompt))
                if prompt_len > len(input_ids):
                    prompt_len = len(input_ids)
                loss_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)
            else:
                prompt = item.get("prompt", "")
                response = item.get("response", "")
                full_text = prompt + response
                encoded = tokenizer.encode(full_text)
                input_ids = encoded[:max_seq_len]
                labels = input_ids.copy()
                prompt_len = len(tokenizer.encode(prompt))
                if prompt_len > len(input_ids):
                    prompt_len = len(input_ids)
                loss_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

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

        # BUG-017 fix: only compute CE on assistant token positions
        # instead of computing on all tokens then zeroing out prompt positions
        if loss_mask is not None:
            shift_mask = loss_mask[..., 1:].contiguous().view(-1)
            active_indices = shift_mask.nonzero(as_tuple=True)[0]
            if active_indices.numel() > 0:
                active_logits = shift_logits.view(-1, vocab_size)[active_indices]
                active_labels = shift_labels.view(-1)[active_indices]
                loss = F.cross_entropy(active_logits, active_labels, reduction="mean")
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        loss = loss + aux_loss

        return loss
