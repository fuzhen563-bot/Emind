import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from contextlib import nullcontext
from typing import List, Dict, Any, Optional, Tuple

from model import EmindLM
from training.config import TrainingConfig
from training.trainer import TrainerBase


def _offload_model_to_cpu(model: EmindLM) -> EmindLM:
    """BUG-012 fix: Move ref model to CPU to halve GPU memory usage for DPO."""
    model.cpu()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


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

        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        def _encode(text: str):
            ids = self.tokenizer.encode(text)[:self.max_seq_len]
            pad = [self.pad_token_id] * (self.max_seq_len - len(ids))
            return torch.tensor(ids + pad, dtype=torch.long)

        return {
            "chosen_input_ids": _encode(prompt + chosen),
            "rejected_input_ids": _encode(prompt + rejected),
            "prompt_len": prompt_len,
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
        label_smoothing: float = 0.0,
    ):
        super().__init__(model, config, train_dataset, eval_dataset)
        # BUG-012 fix: offload ref model to CPU to halve GPU memory
        self.ref_model = ref_model
        if self.ref_model is not None:
            _offload_model_to_cpu(self.ref_model)
        self.beta = beta
        self.label_smoothing = label_smoothing

    @staticmethod
    def _log_probs(logits: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        per_token_logps = F.log_softmax(shift_logits, dim=-1).gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        mask = (shift_labels != pad_token_id).float()
        return (per_token_logps * mask).sum(-1), mask.sum(-1)

    @staticmethod
    def _response_log_probs(logits: torch.Tensor, input_ids: torch.Tensor, prompt_lens: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        per_token_logps = F.log_softmax(shift_logits, dim=-1).gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        response_mask = (shift_labels != pad_token_id).float()
        prompt_mask = torch.arange(shift_labels.shape[1], device=shift_labels.device).unsqueeze(0) >= prompt_lens.unsqueeze(1) - 1
        mask = response_mask * prompt_mask.float()
        return (per_token_logps * mask).sum(-1)

    def compute_loss(self, batch) -> torch.Tensor:
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        prompt_lens = batch["prompt_len"].to(self.device)
        pad_id = getattr(self.model.config, 'pad_token_id', 0)

        _, chosen_logits, _, _, c_aux, _ = self.model(chosen_ids)
        _, rejected_logits, _, _, r_aux, _ = self.model(rejected_ids)

        policy_chosen_logps = self._response_log_probs(chosen_logits, chosen_ids, prompt_lens, pad_id)
        policy_rejected_logps = self._response_log_probs(rejected_logits, rejected_ids, prompt_lens, pad_id)

        if self.ref_model is not None:
            with torch.no_grad():
                # BUG-012 fix: ref model on CPU, move inputs to CPU for inference
                _, ref_chosen_logits, _, _, _, _ = self.ref_model(chosen_ids.cpu())
                _, ref_rejected_logits, _, _, _, _ = self.ref_model(rejected_ids.cpu())
                ref_chosen_logits = ref_chosen_logits.to(self.device)
                ref_rejected_logits = ref_rejected_logits.to(self.device)
                ref_chosen_logps = self._response_log_probs(ref_chosen_logits, chosen_ids, prompt_lens, pad_id)
                ref_rejected_logps = self._response_log_probs(ref_rejected_logits, rejected_ids, prompt_lens, pad_id)
        else:
            ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
            ref_rejected_logps = torch.zeros_like(policy_rejected_logps)

        chosen_ratio = policy_chosen_logps - ref_chosen_logps
        rejected_ratio = policy_rejected_logps - ref_rejected_logps
        logits_diff = self.beta * (chosen_ratio - rejected_ratio)
        logits_diff = torch.clamp(logits_diff, max=50.0)

        if self.label_smoothing > 0:
            losses = -F.logsigmoid(logits_diff) * (1 - self.label_smoothing) - F.logsigmoid(-logits_diff) * self.label_smoothing
        else:
            losses = -F.logsigmoid(logits_diff)

        return losses.mean() + c_aux + r_aux
