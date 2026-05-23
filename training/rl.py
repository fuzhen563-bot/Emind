"""
Emind RL — PPO & GRPO 强化学习训练器
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from model import EmindLM
from training.config import TrainingConfig
from training.trainer import TrainerBase
from training.checkpoint import CheckpointManager
from training.metrics import MetricsTracker


# =============================================================================
# Configs
# =============================================================================

@dataclass
class PPOConfig(TrainingConfig):
    mode: str = "ppo"
    kl_coef: float = 0.1
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 4
    gamma: float = 0.99
    use_kl_estimator: str = "kl3"


@dataclass
class GRPOConfig(TrainingConfig):
    mode: str = "grpo"
    kl_coef: float = 0.04
    group_size: int = 8
    ppo_epochs: int = 1
    mini_batch_size: int = 8
    use_kl_estimator: str = "kl3"


# =============================================================================
# Dataset
# =============================================================================

class PPODataset(Dataset):
    """PPO/GRPO 数据集: prompt + 多条候选回复 + 奖励"""
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_seq_len: int = 2048, pad_token_id: int = 0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item.get("prompt", "")
        responses = item.get("responses", [])
        rewards = item.get("rewards", [])

        if not responses:
            responses = [item.get("response", "")]
            rewards = [item.get("reward", 0.0)]

        def _encode(text: str):
            ids = self.tokenizer.encode(text)[:self.max_seq_len]
            pad = [self.pad_token_id] * (self.max_seq_len - len(ids))
            return torch.tensor(ids + pad, dtype=torch.long)

        return {
            "prompt": _encode(prompt),
            "responses": [_encode(r) for r in responses],
            "rewards": torch.tensor(rewards, dtype=torch.float),
            "response_texts": responses,
            "prompt_text": prompt,
        }


# =============================================================================
# KL Estimators
# =============================================================================

def kl_estimate(logps: torch.Tensor, ref_logps: torch.Tensor, method: str = "kl3") -> torch.Tensor:
    """
    logps / ref_logps: (batch, seq_len)
    返回: (batch,) 每个样本的 KL 散度
    """
    diff = (logps - ref_logps).clamp(-20, 20)
    ratio = torch.exp(diff)
    if method == "kl1":
        return (-diff).mean(-1)
    elif method == "kl2":
        return diff.mean(-1)
    else:
        return (ratio - 1 - diff).mean(-1)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return (tensor * mask).sum(dim) / mask.sum(dim).clamp(min=1)


# =============================================================================
# Helper: 生成补全 + 计算 log prob
# =============================================================================

@torch.no_grad()
def generate_completions(
    model: EmindLM,
    prompts: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
) -> Tuple[torch.Tensor, int]:
    """批量生成回复，返回 (full_sequences, prompt_len)"""
    full = model.generate(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return full, prompts.size(1)


def compute_token_logps(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """
    返回 per-token log probs, shape (batch, seq_len-1)
    (不对 padding mask, 调用方负责 response-only mask)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    per_token = F.log_softmax(shift_logits, dim=-1)
    token_logps = per_token.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_logps


# =============================================================================
# Reward Model Wrapper
# =============================================================================

class RewardModel(nn.Module):
    """简单线性层 reward model, 接在 EmindLM 最后一层 hidden states 之上"""
    def __init__(self, base_model: EmindLM, hidden_dim: int = 4096):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.base_model.requires_grad_(False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, _, _, _, _, last_hidden = self.base_model(input_ids)
        return self.reward_head(last_hidden[:, -1, :].detach()).squeeze(-1)


class RewardModelTrainer:
    """Reward Model 训练器 (pairwise ranking loss)"""
    def __init__(
        self,
        reward_model: RewardModel,
        config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.model = reward_model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.reward_head.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def train(self):
        loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            for batch in loader:
                chosen_ids = batch["chosen_input_ids"].to(self.device)
                rejected_ids = batch["rejected_input_ids"].to(self.device)
                r_chosen = self.model(chosen_ids)
                r_rejected = self.model(rejected_ids)
                loss = -F.logsigmoid(r_chosen - r_rejected).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg = total_loss / max(1, len(loader))
            print(f"RM Epoch {epoch+1}: loss={avg:.4f}")


# =============================================================================
# PPOTrainer
# =============================================================================

class PPOTrainer(TrainerBase):
    """
    PPO with clipped surrogate + value function + KL penalty.
    架构: policy (训练), ref (冻结), value head, reward model (冻结/外部)
    """
    def __init__(
        self,
        model: EmindLM,
        config: PPOConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        ref_model: Optional[EmindLM] = None,
        reward_model: Optional[Union[RewardModel, Callable]] = None,
        tokenizer=None,
    ):
        from training.config import TrainingConfig
        super().__init__(model, config, train_dataset, eval_dataset)
        self.tokenizer = tokenizer

        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.value_head = nn.Linear(self.model.config.d_model, 1)
        self.value_head.to(self.device)

        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs * max(len(train_dataset), 1),
        )
        self.reward_model = reward_model
        if isinstance(self.reward_model, RewardModel):
            self.reward_model.to(self.device)
            self.reward_model.eval()
            for p in self.reward_model.parameters():
                p.requires_grad = False
        self.checkpoint = CheckpointManager(config.output_dir, config.save_total_limit, config.experiment_name)
        self.metrics = MetricsTracker()
        self.global_step = 0

    def _value(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, _, _, _, _, last_hidden = self.model(input_ids)
        return self.value_head(last_hidden[:, -1, :]).squeeze(-1)

    @torch.no_grad()
    def _rollout(self, prompts: torch.Tensor, prompt_texts: List[str]) -> Dict:
        """生成回复, 计算奖励/value/log probs"""
        self.model.eval()

        sequences, prompt_len = generate_completions(
            self.model, prompts,
            max_new_tokens=self.config.max_seq_len // 2,
        )

        _, logits, _, _, aux_rollout, _ = self.model(sequences)
        policy_logps = compute_token_logps(logits, sequences)

        if self.ref_model is not None:
            _, ref_logits, _, _, _, _ = self.ref_model(sequences)
            ref_logps = compute_token_logps(ref_logits, sequences)
        else:
            ref_logps = None

        if self.reward_model is not None:
            if isinstance(self.reward_model, RewardModel):
                rewards = self.reward_model(sequences)
            elif callable(self.reward_model):
                resp_texts = [self.tokenizer.decode(s[prompt_len:]) for s in sequences] if self.tokenizer else [""] * sequences.size(0)
                rewards = torch.tensor(self.reward_model(prompt_texts, resp_texts), device=self.device, dtype=torch.float)
            else:
                rewards = torch.zeros(sequences.size(0), device=self.device)
        else:
            rewards = torch.zeros(sequences.size(0), device=self.device)

        values = self._value(sequences)

        self.model.train()
        return {
            "sequences": sequences,
            "prompt_len": prompt_len,
            "policy_logps": policy_logps,
            "ref_logps": ref_logps,
            "rewards": rewards,
            "values": values,
        }

    def _response_mask(self, seq_len: int, prompt_len: int, batch_size: int) -> torch.Tensor:
        """
        (batch, seq_len-1) – 只覆盖 response token
        (对应 compute_token_logps 的 shift)
        """
        arange = torch.arange(seq_len - 1, device=self.device).unsqueeze(0)
        return (arange >= prompt_len - 1).float().expand(batch_size, -1)

    def _ppo_loss(
        self,
        policy_logps: torch.Tensor,
        ref_logps: Optional[torch.Tensor],
        advantages: torch.Tensor,
        old_logps: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = self.config.clip_epsilon
        kl_coef = self.config.kl_coef

        ratio = torch.exp(policy_logps - old_logps)
        adv = advantages.unsqueeze(-1)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * ratio.clamp(1 - eps, 1 + eps)
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = masked_mean(pg_loss, mask)

        if ref_logps is not None:
            kl = kl_estimate(policy_logps, ref_logps, method=self.config.use_kl_estimator)
            kl_loss = kl_coef * masked_mean(kl, mask)
        else:
            kl_loss = 0.0

        v_loss = (F.mse_loss(values, returns, reduction="none") * self.config.vf_coef).mean()

        total = pg_loss + kl_loss + v_loss
        return total, pg_loss, kl_loss, v_loss

    def train(self, resume: bool = False):
        loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        for epoch in range(self.config.epochs):
            epoch_stats = {"pg": 0.0, "kl": 0.0, "vf": 0.0, "reward": 0.0}
            n = 0

            for batch in loader:
                prompts = batch["prompt"].to(self.device)
                prompt_texts = batch.get("prompt_text", [""] * prompts.size(0))

                rollout = self._rollout(prompts, prompt_texts)

                seq = rollout["sequences"]
                plen = rollout["prompt_len"]
                bs = seq.size(0)
                old_logps = rollout["policy_logps"].detach()
                ref_logps = rollout["ref_logps"].detach() if rollout["ref_logps"] is not None else None
                old_val = rollout["values"].detach()

                advantages = rollout["rewards"] - old_val
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                returns = rollout["rewards"]

                mask = self._response_mask(seq.size(1), plen, bs)

                for _ in range(self.config.ppo_epochs):
                    _, logits, _, _, _, _ = self.model(seq)
                    policy_logps = compute_token_logps(logits, seq)
                    values = self._value(seq)

                    loss, pg, kl, vf = self._ppo_loss(
                        policy_logps, ref_logps, advantages, old_logps,
                        values, returns, mask,
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()

                self.global_step += 1
                n += 1
                epoch_stats["pg"] += pg.item()
                epoch_stats["kl"] += kl.item() if isinstance(kl, torch.Tensor) else kl
                epoch_stats["vf"] += vf.item()
                epoch_stats["reward"] += rollout["rewards"].mean().item()

                if self.global_step % self.config.logging_steps == 0:
                    print(f"PPO E{epoch+1} S{self.global_step} | PG={pg.item():.4f} KL={kl.item() if isinstance(kl, torch.Tensor) else kl:.4f} V={vf.item():.4f} R={rollout['rewards'].mean().item():.4f}")

            avg = {k: v / max(1, n) for k, v in epoch_stats.items()}
            print(f"PPO Epoch {epoch+1} done | avg_reward={avg['reward']:.4f}")
            if (epoch + 1) % max(1, self.config.save_steps) == 0:
                self.checkpoint.save(
                    step=self.global_step,
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    metrics=avg,
                )


# =============================================================================
# GRPOTrainer (DeepSeekMath GRPO – no critic)
# =============================================================================

class GRPOTrainer:
    """
    GRPO: 每组 group_size 个回复, 组内 reward 归一化做 advantage, 策略梯度 + KL.
    """
    def __init__(
        self,
        model: EmindLM,
        config: GRPOConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        ref_model: Optional[EmindLM] = None,
        reward_fn: Optional[Callable] = None,
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.reward_fn = reward_fn

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs * max(len(train_dataset), 1),
        )
        self.checkpoint = CheckpointManager(config.output_dir, config.save_total_limit, config.experiment_name)
        self.metrics = MetricsTracker()
        self.global_step = 0

    @torch.no_grad()
    def _generate_group(self, prompts: torch.Tensor) -> Dict:
        bs = prompts.size(0)
        gs = self.config.group_size
        exp_prompts = prompts.repeat_interleave(gs, dim=0)

        sequences, prompt_len = generate_completions(
            self.model, exp_prompts,
            max_new_tokens=self.config.max_seq_len // 2,
        )

        _, logits, _, _, _, _ = self.model(sequences)
        policy_logps = compute_token_logps(logits, sequences)

        if self.ref_model is not None:
            _, ref_logits, _, _, _, _ = self.ref_model(sequences)
            ref_logps = compute_token_logps(ref_logits, sequences)
        else:
            ref_logps = None

        return {
            "sequences": sequences,
            "prompt_len": prompt_len,
            "policy_logps": policy_logps,
            "ref_logps": ref_logps,
            "batch_size": bs,
            "group_size": gs,
            "exp_prompts": exp_prompts,
        }

    def _grpo_loss(
        self,
        policy_logps: torch.Tensor,
        ref_logps: Optional[torch.Tensor],
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        kc = self.config.kl_coef
        pg = -(policy_logps * advantages.unsqueeze(-1))
        pg = masked_mean(pg, mask)

        if ref_logps is not None:
            kl = kl_estimate(policy_logps, ref_logps, method=self.config.use_kl_estimator)
            kl = kc * masked_mean(kl, mask)
        else:
            kl = 0.0

        return pg + kl

    def train(self, resume: bool = False):
        loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        for epoch in range(self.config.epochs):
            epoch_stats = {"loss": 0.0, "reward": 0.0}
            n = 0

            for batch in loader:
                prompts = batch["prompt"].to(self.device)
                prompt_texts = batch.get("prompt_text", [""] * prompts.size(0))
                bs = prompts.size(0)
                gs = self.config.group_size

                gd = self._generate_group(prompts)
                seq = gd["sequences"]
                plen = gd["prompt_len"]
                policy_logps = gd["policy_logps"]
                ref_logps = gd["ref_logps"]

                if self.reward_fn is not None:
                    resp_texts = []
                    for i in range(seq.size(0)):
                        t = self.tokenizer.decode(seq[i, plen:]) if self.tokenizer else ""
                        resp_texts.append(t)
                    prompt_expanded = [p for p in prompt_texts for _ in range(gs)]
                    rewards = torch.tensor(
                        self.reward_fn(prompt_expanded, resp_texts),
                        device=self.device, dtype=torch.float,
                    )
                else:
                    rewards = batch.get("rewards", torch.zeros(seq.size(0), device=self.device))
                    if rewards.size(0) == bs:
                        rewards = rewards.repeat_interleave(gs)

                r_grp = rewards.view(bs, gs)
                mean_r = r_grp.mean(dim=1, keepdim=True)
                std_r = r_grp.std(dim=1, keepdim=True) + 1e-8
                advantages = ((r_grp - mean_r) / std_r).view(-1)

                slen = seq.size(1)
                mask = (torch.arange(slen - 1, device=self.device).unsqueeze(0) >= plen - 1).float().expand(seq.size(0), -1)

                self.optimizer.zero_grad()
                loss = self._grpo_loss(policy_logps, ref_logps, advantages, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()

                self.global_step += 1
                n += 1
                epoch_stats["loss"] += loss.item()
                epoch_stats["reward"] += rewards.mean().item()

                if self.global_step % self.config.logging_steps == 0:
                    print(f"GRPO E{epoch+1} S{self.global_step} | loss={loss.item():.4f} R={rewards.mean().item():.4f}")

            avg = {k: v / max(1, n) for k, v in epoch_stats.items()}
            print(f"GRPO Epoch {epoch+1} done | avg_reward={avg['reward']:.4f}")
            if (epoch + 1) % max(1, self.config.save_steps) == 0:
                self.checkpoint.save(
                    step=self.global_step,
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                    metrics=avg,
                )
