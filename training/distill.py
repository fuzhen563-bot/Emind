import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

from model import EmindLM
from training.config import TrainingConfig
from training.trainer import TrainerBase


class DistillationDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_seq_len: int = 2048, pad_token_id: int = 0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item if isinstance(item, str) else item.get("text", item.get("prompt", ""))
        ids = self.tokenizer.encode(text)[:self.max_seq_len]
        pad = [self.pad_token_id] * (self.max_seq_len - len(ids))
        return {
            "input_ids": torch.tensor(ids + pad, dtype=torch.long),
            "labels": torch.tensor(ids + pad, dtype=torch.long),
        }


class DistillationTrainer(TrainerBase):
    def __init__(
        self,
        model: EmindLM,
        teacher_model: EmindLM,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        temperature: float = 2.0,
        alpha_ce: float = 0.5,
        alpha_clm: float = 0.5,
    ):
        super().__init__(model, config, train_dataset, eval_dataset)
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_clm = alpha_clm

    def compute_loss(self, batch) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        _, student_logits, _ = self.model(input_ids)

        with torch.no_grad():
            _, teacher_logits, _ = self.teacher_model(input_ids)

        vocab_size = student_logits.shape[-1]
        shift_student = student_logits[:, :-1, :].contiguous()
        shift_teacher = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = (shift_labels != 0).float().unsqueeze(-1)

        loss_clm = F.cross_entropy(
            shift_student.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=0,
        )

        log_softmax_s = F.log_softmax(shift_student / self.temperature, dim=-1)
        softmax_t = F.softmax(shift_teacher / self.temperature, dim=-1)
        loss_ce = -(log_softmax_s * softmax_t * mask).sum(-1).mean() * (self.temperature ** 2)

        return self.alpha_clm * loss_clm + self.alpha_ce * loss_ce
