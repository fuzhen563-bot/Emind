import torch
import random
from torch.utils.data import Dataset, ConcatDataset
from typing import List, Dict, Any, Optional


class CurriculumDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_seq_len: int = 2048,
                 pad_token_id: int = 0, sort_by: str = "length", stages: int = 3,
                 replay_data: Optional[List[Dict[str, Any]]] = None, replay_ratio: float = 0.1):
        if sort_by == "length":
            data = sorted(data, key=lambda x: len(x.get("prompt", "") + x.get("response", "") + x.get("chosen", "")))
        elif sort_by == "type":
            type_order = {"identity": 0, "anti_hallucination": 1, "code": 2, "reasoning": 3, "deep_reasoning": 4}
            data = sorted(data, key=lambda x: type_order.get(x.get("type", ""), 2))
        elif sort_by == "shuffle":
            random.shuffle(data)

        self.stage_size = max(1, len(data) // stages)
        self.num_stages = stages
        self.current_stage = 0
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.replay_data = replay_data
        self.replay_ratio = replay_ratio

        self.stage_boundaries = [(i * self.stage_size, min((i + 1) * self.stage_size, len(data)))
                                 for i in range(stages)]
        self._build_examples(data, tokenizer)

    def _encode_item(self, item, tokenizer):
        if isinstance(item, str):
            encoded = tokenizer.encode(item)[:self.max_seq_len]
            return encoded, encoded, [1] * len(encoded)

        prompt = item.get("prompt", "")
        response = item.get("response", "") or item.get("chosen", "")
        full_text = prompt + response
        encoded = tokenizer.encode(full_text)[:self.max_seq_len]
        prompt_len = len(tokenizer.encode(prompt))
        labels = encoded.copy()
        loss_mask = [0] * prompt_len + [1] * (len(encoded) - prompt_len)

        pad_len = self.max_seq_len - len(encoded)
        if pad_len > 0:
            encoded += [self.pad_token_id] * pad_len
            labels += [-100] * pad_len
            loss_mask += [0] * pad_len
        return encoded, labels, loss_mask[:self.max_seq_len]

    def _build_examples(self, data, tokenizer):
        self._data = data
        self._examples = []
        for item in data:
            ids, labs, mask = self._encode_item(item, tokenizer)
            self._examples.append({
                "input_ids": torch.tensor(ids[:self.max_seq_len], dtype=torch.long),
                "labels": torch.tensor(labs[:self.max_seq_len], dtype=torch.long),
                "loss_mask": torch.tensor(mask[:self.max_seq_len], dtype=torch.float),
            })

        self._replay_examples = []
        if self.replay_data:
            for item in self.replay_data:
                ids, labs, mask = self._encode_item(item, tokenizer)
                self._replay_examples.append({
                    "input_ids": torch.tensor(ids[:self.max_seq_len], dtype=torch.long),
                    "labels": torch.tensor(labs[:self.max_seq_len], dtype=torch.long),
                    "loss_mask": torch.tensor(mask[:self.max_seq_len], dtype=torch.float),
                })

    def advance_stage(self):
        if self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            return True
        return False

    def get_stage_data(self):
        start, end = self.stage_boundaries[self.current_stage]
        return start, end

    def __len__(self):
        start, end = self.get_stage_data()
        n = end - start
        if self._replay_examples:
            n += max(1, int(n * self.replay_ratio))
        return max(1, n)

    def __getitem__(self, idx):
        start, end = self.get_stage_data()
        stage_len = end - start
        if idx < stage_len:
            return self._examples[start + idx]
        if self._replay_examples:
            return random.choice(self._replay_examples)
        return self._examples[start + idx % stage_len]


class StageScheduler:
    """Multi-stage SFT scheduler: advance curriculum stages at epoch boundaries."""

    def __init__(self, curriculum_dataset: CurriculumDataset, trainer, train_cfg):
        self.dataset = curriculum_dataset
        self.trainer = trainer
        self.train_cfg = train_cfg
        self.epochs_per_stage = max(1, train_cfg.epochs // curriculum_dataset.num_stages)

    def should_advance(self, epoch: int) -> bool:
        return epoch > 0 and epoch % self.epochs_per_stage == 0

    def on_epoch_end(self, epoch: int):
        if self.should_advance(epoch + 1):
            if self.dataset.advance_stage():
                start, end = self.dataset.get_stage_data()
                print(f"  >>> Curriculum stage {self.dataset.current_stage + 1}/{self.dataset.num_stages} "
                      f"(samples {start}-{end})")
                self.trainer.train_dataloader = self.trainer._make_dataloader
