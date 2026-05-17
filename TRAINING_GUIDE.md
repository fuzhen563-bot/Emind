# Emind 训练指南

本文档详细介绍 Emind 模型的训练流程、数据蒸馏、模型配置和参数说明。

> **注意**: 原有 `01_pretrain.py` ~ `07_feedback_training.py` 7 个独立脚本已废弃（含 DeprecationWarning 头部），**新训练流程统一通过 `cli.py` 和 `training/` 包完成**。

---

## 一、训练管线总览

```
数据蒸馏 (DistillationPipeline)
  ├── 代码数据 (3000条)     ← DeepSeek V4 Pro Teacher
  ├── 推理数据 (500条)
  ├── 深度推理 (1000条)
  ├── 反幻觉 (1500条)       ← 50% Teacher + 50% 合成拒绝样本
  └── 身份认知 (100条)      ← 零成本合成
         │
         ▼
SFT 训练 (SFTTrainer)
  └── assistant-only loss masking
         │
         ▼
DPO 偏好对齐 (可选)
  └── 标准 DPO loss, beta=0.1
         │
         ▼
RLHF 强化学习 (可选)
  ├── PPO  — clipped surrogate + value function + KL penalty
  └── GRPO — 组内 advantage 归一化, 无 critic
         │
         ▼
推理部署
  ├── 本地模型 (model.py generate)
  ├── vLLM (Prefix Caching + Speculative Decoding + 量化)
  ├── Ollama / llama.cpp / HuggingFace
  └── 亦API 云端
```

---

## 二、数据蒸馏（核心新增）

利用 DeepSeek V4 Pro 等 Teacher 模型生成 6000 条高质量训练数据。

### 2.1 数据构成

| 类型 | 数量 | 种子模板数 | 生成策略 | 说明 |
|------|------|-----------|---------|------|
| 代码 | 3000 | 18 | direct, cot, verify, explain_then_code | 数据结构/算法/API/安全审查 |
| 推理 | 500 | 19 | cot, reason_then_answer | 数学/逻辑/物理/概率 |
| 深度推理 | 1000 | 18 | reason_then_answer, cot, verify | 多步推理/反事实/分治/类比 |
| 反幻觉 | 1500 | 23+13 | refuse, direct, verify + 合成拒绝样本 | 无法回答/虚假前提/自相矛盾 |
| 身份认知 | 100 | 16+10 | direct + 合成 QA 对 | 模型身份/开发者/能力介绍 |

### 2.2 蒸馏命令

```bash
# 完整蒸馏管线
python cli.py pipeline \
  --distill-code 3000 \
  --distill-reasoning 500 \
  --distill-deep-reasoning 1000 \
  --distill-anti-hallucination 1500 \
  --distill-identity 100 \
  --teacher-backend cloud_api \
  --teacher-api-key sk-xxxxx \
  --teacher-model deepseek-v4-pro \
  --teacher-base-url https://api.deepseek.com \
  --distill-output data/emind_code_4b \
  --distill-strategies direct,cot,verify,refuse,reason_then_answer

# 仅蒸馏代码
python cli.py pipeline --distill-code 500 \
  --teacher-backend cloud_api --teacher-api-key sk-xxxxx

# 仅蒸馏推理
python cli.py pipeline --distill-reasoning 200 --distill-deep-reasoning 300 \
  --teacher-backend cloud_api --teacher-api-key sk-xxxxx
```

### 2.3 Teacher 身份自动注入

每次调用 Teacher 时自动在 prompt 前附加身份前缀：

```
(你是一个名为Emind·智脑的 AI 助手，由亦梓科技开发。请在回复中以Emind·智脑的身份回答。)

用 Python 实现一个 LRU Cache...
```

确保 Teacher 生成的回复命中 Emind 身份，无需二次修改。

### 2.4 零成本合成样本

**拒绝样本** — 13 个无法回答的问题 + 7 种礼貌拒绝，随机组合：
```
prompt: "请证明 P = NP。"
response: "我不知道这个问题的答案。作为 AI，我的知识存在边界，我不应该编造答案。"
```

**身份样本** — 10 个 QA 对，运行时注入名称和开发者变量：
```
prompt: "你是谁？"
response: "我是Emind·智脑，由亦梓科技开发的新一代 AI 大语言模型。..."
```

### 2.5 输出格式

每条数据为 JSON 行，写入 `distilled_sft.jsonl`：

```json
{
  "prompt": "用 Rust 实现 LRU Cache...",
  "response": "以下是 Rust 实现...",
  "strategy": "explain_then_code",
  "type": "code",
  "source": "distill_via_cloud_api"
}
```

---

## 三、训练 (SFT)

### 3.1 4B 模型配置

```bash
python cli.py train \
  --mode sft \
  --data data/emind_code_4b/distilled_sft.jsonl \
  --d-model 2560 \
  --n-heads 32 \
  --n-kv-heads 8 \
  --n-layers 32 \
  --max-seq-len 4096 \
  --vocab-size 32000 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --epochs 3 \
  --lr 2e-5 \
  --use-bf16 \
  --output-dir checkpoints/emind-code-4b
```

### 3.2 参数解释

| 参数 | 值 | 说明 |
|------|-----|------|
| `d-model` | 2560 | 隐藏层维度 |
| `n-heads` | 32 | 注意力头数 |
| `n-kv-heads` | 8 | GQA 的 K/V 头数 (4:1 压缩) |
| `n-layers` | 32 | Transformer 层数 |
| `max-seq-len` | 4096 | 最大序列长度 |
| `batch-size` | 8 | 每 GPU 批次大小 |
| `gradient-accumulation-steps` | 2 | 梯度累积步数 (有效 batch=16) |
| `epochs` | 3 | 训练轮数 |
| `lr` | 2e-5 | 峰值学习率 (cosine warmup) |
| `use-bf16` | — | BF16 混合精度 (RTX PRO 6000 支持) |

### 3.3 显存预算 (96GB)

| 项目 | BF16 占用 |
|------|----------|
| 模型权重 (4B) | ~8 GB |
| Adam 优化器状态 | ~32 GB |
| 梯度 | ~8 GB |
| Activations (batch=8) | ~20 GB |
| **合计** | **~68 GB** |

**结论**: 96GB 显存可以全参数训 4B，batch-size=16 也绰绰有余 (约 85 GB)。如需更低占用可使用 LoRA。

### 3.4 LoRA 训练

```bash
python cli.py train \
  --mode sft \
  --data data/train.jsonl \
  --d-model 2560 --n-heads 32 --n-kv-heads 8 --n-layers 32 \
  --lora \
  --lora-rank 16 \
  --batch-size 16 \
  --epochs 3 --lr 1e-4 \
  --use-bf16 \
  --output-dir checkpoints/emind-code-4b-lora
```

LoRA 模式下显存降至 ~25 GB。

---

## 四、DPO 偏好对齐

标准 DPO loss:

```bash
python cli.py train \
  --mode dpo \
  --data data/dpo_pairs.json \
  --d-model 2560 --n-heads 32 --n-kv-heads 8 --n-layers 32 \
  --batch-size 4 \
  --epochs 3 \
  --lr 1e-6 \
  --beta 0.1 \
  --use-bf16 \
  --output-dir checkpoints/emind-code-4b-dpo
```

DPO 数据格式：

```json
{
  "prompt": "写一个快速排序",
  "chosen": "这是快速排序的实现...",
  "rejected": "快速排序就是对数组排序..."
}
```

---

## 五、知识蒸馏 (Logit-based)

```bash
python cli.py train \
  --mode distill \
  --data data/distill_pairs.json \
  --d-model 2560 --n-heads 32 --n-kv-heads 8 --n-layers 32 \
  --batch-size 4 \
  --epochs 5 \
  --temperature 4.0 \
  --lr 1e-5 \
  --use-bf16 \
  --output-dir checkpoints/emind-code-4b-distill
```

蒸馏模式下 Student 用 `d-model 2560`，Teacher 自动用 `d-model 5120`。

---

## 六、强化学习 (PPO / GRPO)

### 6.1 GRPO (DeepSeekMath 方案，推荐)

无需 critic 网络，组内 advantage 归一化：

```bash
python cli.py rl \
  --rl-mode grpo \
  --data data/rl_data.json \
  --d-model 2560 --n-heads 32 --n-kv-heads 8 --n-layers 32 \
  --group-size 8 \
  --epochs 1 \
  --batch-size 4 \
  --lr 1e-6 \
  --kl-coef 0.1 \
  --output-dir checkpoints/emind-code-4b-grpo
```

GRPO 数据格式：

```json
{
  "prompt": "写一个二分查找",
  "response": "def binary_search...",
  "reward": 0.85
}
```

### 6.2 PPO (含 critic)

```bash
python cli.py rl \
  --rl-mode ppo \
  --data data/rl_data.json \
  --d-model 2560 --n-heads 32 --n-kv-heads 8 --n-layers 32 \
  --ppo-epochs 4 \
  --mini-batch-size 4 \
  --clip-epsilon 0.2 \
  --kl-coef 0.1 \
  --epochs 1 --batch-size 4 --lr 1e-6 \
  --output-dir checkpoints/emind-code-4b-ppo
```

### 6.3 Reward Model 训练

```bash
python cli.py rl \
  --rl-mode rm \
  --data data/dpo_pairs.json \
  --d-model 2560 --n-heads 32 --n-kv-heads 8 --n-layers 32 \
  --epochs 3 --batch-size 4 --lr 1e-5 \
  --output-dir checkpoints/reward-model
```

---

## 七、模型大小配置

| 规模 | 参数量 | d_model | n_heads | n_kv_heads | n_layers | d_ff | 显存(BF16) | 适用场景 |
|------|--------|---------|---------|-----------|---------|------|-----------|---------|
| **tiny** | ~100M | 768 | 12 | 4 | 12 | 3072 | ~3 GB | 测试/原型 |
| **0.5b** | ~500M | 1280 | 20 | 4 | 20 | 5120 | ~6 GB | 简单任务 |
| **1b** | ~1B | 2048 | 16 | 4 | 24 | 8192 | ~12 GB | 通用 |
| **3b** | ~3B | 2560 | 32 | 8 | 24 | 10240 | ~30 GB | 生产 |
| **4b** | ~4B | 2560 | 32 | 8 | 32 | 10240 | ~40 GB | 生产(推荐) |
| **7b** | ~7B | 4096 | 32 | 8 | 32 | 11008 | ~60 GB | RTX PRO 6000 |

### 模型参数计算公式

```
总参数量 ≈ vocab_size × d_model
         + n_layers × (3 × d_model × d_ff + 2 × d_model × n_heads × d_head)
```

其中 `d_head = d_model / n_heads`。

---

## 八、分布式训练

```bash
# FSDP 全分片 (推荐, 4B 模型)
torchrun --nproc_per_node=8 cli.py train \
  --mode sft --data data/train.jsonl \
  --d-model 2560 --n-heads 32 --n-kv-heads 8 --n-layers 32 \
  --batch-size 16 \
  --use-fsdp --use-bf16

# DDP (单机多卡)
torchrun --nproc_per_node=4 cli.py train \
  --mode sft --data data/train.jsonl \
  --d-model 2560 --batch-size 8
```

---

## 九、评测

```bash
# 本地模型
python cli.py eval \
  --model checkpoints/emind-code-4b/best_model.pt \
  --benchmarks humaneval,mmlu,ceval \
  --max-new-tokens 512

# 云端/后端
python cli.py eval \
  --backend cloud_api \
  --benchmarks mmlu
```

---

## 十、推理

### 10.1 本地模型

```bash
python cli.py infer \
  --model checkpoints/emind-code-4b/best_model.pt \
  --interactive
```

### 10.2 vLLM (Prefix Caching + Speculative Decoding)

```bash
python cli.py infer \
  --backend vllm \
  --model ./models/emind-4b \
  --vllm-prefix-caching \
  --vllm-speculative \
  --vllm-draft-model ./models/emind-1b-draft \
  --interactive
```

### 10.3 vLLM Server (生产)

```bash
python cli.py serve --vllm \
  --model ./models/emind-4b \
  --vllm-port 8000 \
  --vllm-prefix-caching \
  --vllm-chunked-prefill
```

---

## 十一、完整训练流程（4B 模型从零到上线）

```bash
# 0. 环境
python -m venv venv && venv\Scripts\activate && pip install -e .

# 1. 蒸馏 6000 条数据 (~$6)
export EMIND_API_KEY="sk-xxxxx"
python cli.py pipeline \
  --distill-code 3000 --distill-reasoning 500 \
  --distill-deep-reasoning 1000 --distill-anti-hallucination 1500 \
  --distill-identity 100 \
  --teacher-backend cloud_api \
  --teacher-model deepseek-v4-pro \
  --distill-output data/emind_code_4b

# 2. SFT 训练 (~3h)
python cli.py train \
  --mode sft \
  --data data/emind_code_4b/distilled_sft.jsonl \
  --d-model 2560 --n-heads 32 --n-kv-heads 8 --n-layers 32 \
  --batch-size 8 --epochs 3 --lr 2e-5 --use-bf16 \
  --output-dir checkpoints/emind-code-4b

# 3. DPO (可选, ~2h)
python cli.py train --mode dpo \
  --data data/emind_code_4b/dpo_pairs.jsonl \
  --d-model 2560 --batch-size 4 --lr 1e-6 --beta 0.1 \
  --output-dir checkpoints/emind-code-4b-dpo

# 4. 评测
python cli.py eval \
  --model checkpoints/emind-code-4b/best_model.pt \
  --benchmarks humaneval

# 5. 启动 Web 服务
python cli.py serve --port 3333

# 6. 或启动 vLLM Server (生产)
python cli.py serve --vllm \
  --model checkpoints/emind-code-4b/best_model.pt \
  --vllm-prefix-caching --vllm-gpu-memory 0.92
```

---

## 十二、常见问题

### Q: 显存不足怎么办？

1. 减小 `batch-size` (4 → 2 → 1)
2. 增大 `gradient-accumulation-steps` (保持有效 batch 不变)
3. 启用 LoRA (`--lora --lora-rank 16`)
4. 减小 `max-seq-len` (2048 → 1024)
5. 启用 `--use-bf16` (比 FP16 省 10% 显存)

### Q: 训练 loss 不下降？

- 检查数据格式是否正确（JSON/JSONL）
- 降低学习率 (`--lr 1e-5` → `--lr 5e-6`)
- 增加 warmup 步数（在 config.py 中调整）
- 检查 tokenizer 能否正常编码数据

### Q: 模型回答乱码？

- 确认 `vocab_size` 与 tokenizer 匹配
- 检查 `max-seq-len` 是否过短导致截断
- 推理时使用正确的 tokenizer (EmindTokenizer)

### Q: 如何续训？

```bash
# 自动加载 checkpoints/ 下最新的 checkpoint
python cli.py train --mode sft --data data.jsonl \
  --d-model 2560 \
  --resume
```

### Q: DeepSeek API 蒸馏报错？

- 检查 API key 余额
- 确认 `--teacher-model` 为 `deepseek-v4-pro`
- 检查网络是否可达 `https://api.deepseek.com`

---

## 附录：与旧版脚本对应关系

| 旧脚本 | 新命令 | 说明 |
|--------|--------|------|
| `01_pretrain.py` | `cli.py train --mode sft` | 预训练/SFT 统一 |
| `02_finetune.py` | `cli.py train --mode sft` | 同上 |
| `03_distillation.py` | `cli.py train --mode distill` | Logit 蒸馏 |
| `04_lora.py` | `cli.py train --lora` | 添加 `--lora` 标志 |
| `05_inference.py` | `cli.py infer / serve` | 推理/服务 |
| `06_post_learning.py` | — | 未迁移 (功能重叠) |
| `07_feedback_training.py` | `cli.py rl / train --mode dpo` | RL / DPO |
