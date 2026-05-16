import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Tuple

from model import EmindLM
from training.config import TrainingConfig
from training.trainer import TrainerBase


class DPODataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_seq_len: int = 2048, pad_token_id: int = 0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        def _encode(text: str):
            ids = self.tokenizer.encode(text)[:self.max_seq_len]
            pad = [self.pad_token_id] * (self.max_seq_len - len(ids))
            return torch.tensor(ids + pad, dtype=torch.long)

        return {
            "chosen_input_ids": _encode(prompt + chosen),
            "rejected_input_ids": _encode(prompt + rejected),
        }


class DPOTrainer(TrainerBase):
    def __init__(
        self,
        model: EmindLM,
        ref_model: Optional[EmindLM],
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        beta: float = 0.1,
    ):
        super().__init__(model, config, train_dataset, eval_dataset)
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        self.beta = beta

    def compute_loss(self, batch) -> torch.Tensor:
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)

        _, chosen_logits, _ = self.model(chosen_ids)
        _, rejected_logits, _ = self.model(rejected_ids)

        def _log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            per_token_logps = F.log_softmax(shift_logits, dim=-1).gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            mask = (shift_labels != 0).float()
            return (per_token_logps * mask).sum(-1)

        policy_chosen_logps = _log_probs(chosen_logits, chosen_ids)
        policy_rejected_logps = _log_probs(rejected_logits, rejected_ids)

        if self.ref_model is not None:
            with torch.no_grad():
                _, ref_chosen_logits, _ = self.ref_model(chosen_ids)
                _, ref_rejected_logits, _ = self.ref_model(rejected_ids)
                ref_chosen_logps = _log_probs(ref_chosen_logits, chosen_ids)
                ref_rejected_logps = _log_probs(ref_rejected_logits, rejected_ids)
        else:
            ref_chosen_logps = 0
            ref_rejected_logps = 0

        chosen_diff = policy_chosen_logps - ref_chosen_logps
        rejected_diff = policy_rejected_logps - ref_rejected_logps
        logits_diff = self.beta * (chosen_diff - rejected_diff)
        loss = -F.logsigmoid(logits_diff).mean()

        return loss
