<div align="center">
  <img src="emind.png" alt="Emind AI Logo" width="120">
  <h1>Emind AI</h1>
  <p><strong>亦梓·智脑 — 从研究原型到生产级大模型</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange" alt="PyTorch 2.0+">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    <img src="https://img.shields.io/badge/status-production-blue" alt="Production">
  </p>
</div>

---

Emind 是一个从零构建的生产级大语言模型框架，覆盖 **模型训练 → 数据蒸馏 → 推理服务 → WebUI** 全链路。采用现代 LLaMA 架构（RoPE + GQA + RMSNorm + SwiGLU），支持单卡到多卡分布式训练（DDP / FSDP），内置 SFT / DPO / PPO / GRPO / 蒸馏等完整训练管线。

## 特性

- **现代架构** — RoPE 位置编码（NTK-aware 128K+）、GQA 分组查询注意力、RMSNorm、SwiGLU FFN、KV Cache
- **训练管线** — SFT（loss masking）/ DPO / 蒸馏（logit-based）/ PPO / GRPO / LoRA，统一 TrainerBase 基类
- **vLLM 深度集成** — Prefix Caching、Speculative Decoding（Draft/N-Gram/EAGLE）、AWQ/GPTQ/FP8 量化、多 LoRA 动态切换、Server 模式
- **多推理后端** — vLLM、Ollama、llama.cpp、HuggingFace、亦API 云端、本地模型，自动降级
- **数据蒸馏** — 代码/推理/深度推理/反幻觉/身份认知 5 类数据，DeepSeek V4 Pro Teacher，零成本合成拒绝样本
- **强化学习** — PPO（clipped surrogate + value function）、GRPO（组内 advantage，无 critic）、Reward Model
- **数据管线** — 多源采集 → 清洗去重 → PII 脱敏 → 质量过滤 → Self-Instruct/Evol-Instruct 合成 → 格式化输出
- **评测套件** — MMLU、C-Eval、HumanEval（沙箱安全执行）一键运行
- **WebUI** — FastAPI + SSE 流式、深/亮主题、思考过程可视化、模型竞技场、多模式切换
- **OpenAI 兼容 API** — `/v1/chat/completions`、`/v1/completions`、`/v1/models`，支持 Function Calling
- **Docker 部署** — 多阶段构建 + docker-compose（api / vllm / jupyter 多 profile）

## 项目结构

```
Emind_AI_v2.0/
├── model.py                 # 核心模型 (RoPE+GQA+RMSNorm+SwiGLU+KV Cache)
├── tokenizer.py             # 分词器 (SentencePiece + Fallback)
├── logger.py                # 统一日志
├── distributed_utils.py     # 分布式训练 (DDP + FSDP FULL_SHARD)
│
├── training/                # 统一训练框架
│   ├── config.py            #   训练配置 (effective_batch_size)
│   ├── trainer.py           #   基类 (BF16/FSDP/早停/梯度累积/LR warmup)
│   ├── sft.py               #   监督微调 (assistant-only loss masking)
│   ├── dpo.py               #   偏好对齐 (标准 DPO loss)
│   ├── distill.py           #   知识蒸馏 (logit-based, 温度缩放)
│   ├── rl.py                #   强化学习 (PPO/GRPO/Reward Model)
│   ├── distillation_pipeline.py # 蒸馏数据管线 (Teacher → Student)
│   ├── lora.py              #   LoRA 低秩适配 (apply/merge)
│   ├── checkpoint.py        #   断点管理 (best/latest 双保存)
│   └── metrics.py           #   指标追踪 (loss/lr/ppl)
│
├── data_pipeline/           # 数据管线
│   ├── collector.py         #   多源采集 (TXT/JSON/CSV/目录)
│   ├── cleaner.py           #   清洗去重/PII脱敏/语言检测
│   ├── synthesizer.py       #   数据合成 (Self-Instruct/Evol-Instruct/DPO)
│   ├── formatter.py         #   格式转换 (SFT/DPO/Pretrain/Alpaca)
│   └── dataset.py           #   版本管理 (raw/processed/versions)
│
├── eval/                    # 评测套件
│   ├── mmlu.py              #   MMLU (多学科选择题)
│   ├── ceval.py             #   C-Eval (中文四选一)
│   ├── humaneval.py         #   HumanEval (沙箱安全执行)
│   └── runner.py            #   统一运行器 + leaderboard
│
├── quantization.py          # INT4 / FP8 量化推理
├── vllm_integration.py      # vLLM 深度集成 (Prefix Caching/Speculative Decoding/量化/LoRA/Server)
├── unified_inference.py     # 统一推理引擎 (vLLM/Ollama/llama.cpp/HF/Cloud/本地)
├── web_server.py            # FastAPI + OpenAI 兼容 API + Function Calling
├── cli.py                   # CLI (train/infer/serve/eval/pipeline/rl/vllm)
├── web/                     # WebUI 前端 (深/亮主题, 竞技场, 思考可视化)
│
├── Dockerfile               # 多阶段构建 (python:3.12-slim)
├── docker-compose.yml       # 编排 (api / vllm / jupyter)
├── tests/test_emind.py      # 30+ 单元测试 (Model/Tokenizer/Training/Data/Eval/RL/Distill)
├── Emind/                   # Python 包入口 (from Emind import *)
└── 01_pretrain.py ~ 07_*.py # 原有训练脚本 (含 DeprecationWarning)
```

## 安装

```bash
git clone <repo-url> && cd Emind_AI_v2.0

# 核心依赖
pip install torch>=2.0.0 numpy sentencepiece
pip install transformers>=4.36.0 accelerate>=0.25.0 requests

# 推理后端 (按需)
pip install "vllm>=0.6.0"          # vLLM (Prefix Caching + Speculative Decoding)
pip install autoawq                 # AWQ 量化 (可选)
pip install auto-gptq               # GPTQ 量化 (可选)
pip install llama-cpp-python        # llama.cpp

# Web 服务
pip install fastapi uvicorn jinja2 aiofiles sse-starlette

# 安装 Emind 包
pip install -e .
```

## CLI 命令

| 命令 | 用途 | 示例 |
|------|------|------|
| `train` | 训练模型 (SFT/DPO/Distill) | `cli.py train --mode sft --data data.json --d-model 2560` |
| `infer` | 推理 | `cli.py infer --model model.pt --prompt "你好"` |
| `serve` | 启动 Web / vLLM Server | `cli.py serve` / `cli.py serve --vllm --model ./model` |
| `eval` | 评测 | `cli.py eval --model model.pt --benchmarks humaneval` |
| `pipeline` | 数据处理 / 蒸馏 | `cli.py pipeline --collect data/raw --process` |
| `rl` | 强化学习 | `cli.py rl --rl-mode ppo --data rl.json` |
| `vllm` | vLLM 诊断 | `cli.py vllm --detect --auto-configure` |
| `train-tokenizer` | 训练 SentencePiece 分词器 | `cli.py train-tokenizer --data data.jsonl` |

## 训练一个 4B 模型

### 1. 蒸馏训练数据

```bash
python cli.py pipeline \
  --distill-code 3000 --distill-reasoning 500 \
  --distill-deep-reasoning 1000 --distill-anti-hallucination 1500 \
  --distill-identity 100 \
  --teacher-backend cloud_api \
  --teacher-api-key sk-your-key \
  --teacher-model deepseek-v4-pro \
  --distill-output data/emind_code_4b
```

### 2. SFT 训练

```bash
python cli.py train \
  --mode sft \
  --data data/emind_code_4b/distilled_sft.jsonl \
  --d-model 2560 --n-heads 32 --n-kv-heads 8 --n-layers 32 \
  --max-seq-len 4096 --batch-size 8 \
  --epochs 3 --lr 2e-5 --use-bf16 \
  --output-dir checkpoints/emind-code-4b
```

### 3. DPO 偏好对齐

```bash
python cli.py train --mode dpo --data data/dpo.json \
  --d-model 2560 --batch-size 4 --lr 1e-6 --beta 0.1
```

### 4. GRPO 强化学习

```bash
python cli.py rl --rl-mode grpo --data data/rl.json \
  --d-model 2560 --group-size 8 --lr 1e-6
```

## vLLM 深度集成

### 特性

| 特性 | 支持版本 | 说明 |
|------|---------|------|
| Prefix Caching | vLLM 0.4+ | 自动缓存公共前缀 KV 块 |
| Speculative Decoding | vLLM 0.5+ | Draft model 草稿-验证，2-3x 加速 |
| N-Gram Speculator | vLLM 0.6+ | 无额外模型，基于 n-gram 推测 |
| EAGLE | vLLM 0.6+ | 基于特征层的推测解码 |
| AWQ 量化 | 4-bit | 显存减半，几乎无损 |
| GPTQ 量化 | 4/8-bit | 通用后训练量化 |
| FP8 量化 | Hopper GPU | 原生 FP8 支持 |
| 多 LoRA 动态切换 | vLLM 0.6.3+ | 运行时热加载/卸载 |
| Chunked Prefill | vLLM 0.6+ | 长 prompt 分块填充，降低 TTFT |
| 健康检查 + 自动重连 | — | Server 模式内置 |

### 使用方式

```bash
# 检测 GPU 能力并推荐配置
python cli.py vllm --detect --auto-configure

# 生产级 vLLM Server (所有优化全开)
python cli.py serve --vllm \
  --model ./models/emind-4b \
  --vllm-prefix-caching \
  --vllm-speculative \
  --vllm-draft-model ./models/emind-1b-draft \
  --vllm-num-speculative-tokens 5 \
  --vllm-gpu-memory 0.92 --vllm-dtype bfloat16

# 推理
python cli.py infer --backend vllm --model ./models/emind-4b --prompt "你好"

# Python API
from vllm_integration import VLLMIntegratedEngine, VLLMConfig
engine = VLLMIntegratedEngine(VLLMConfig(model_path="./models/emind-4b", enable_prefix_caching=True))
print(engine.generate("你好"))
```

## 模型架构

| 组件 | 方案 | 说明 |
|------|------|------|
| 位置编码 | **RoPE** | 旋转位置编码，NTK-aware 缩放支持 128K+ |
| 注意力 | **GQA** | 分组查询注意力，n_kv_heads=4~8 |
| 归一化 | **RMSNorm** | 均方根归一化 |
| 前馈网络 | **SwiGLU** | Gated Linear Unit + SiLU |
| KV Cache | 已集成 | 自回归缓存，消除重复计算 |
| 生成策略 | Top-k + Top-p + Temperature + Repetition Penalty | 灵活可控 |

## 训练管线

| 阶段 | 模块 | 说明 |
|------|------|------|
| 预训练 | SFTTrainer | 语言模型预训练 |
| 监督微调 | SFTTrainer | 指令微调，assistant-only loss masking |
| 偏好对齐 | DPOTrainer | 标准 DPO loss |
| 知识蒸馏 | DistillationTrainer | Logit-based 蒸馏 (温度缩放) |
| 强化学习 | PPOTrainer | Clipped surrogate + value function + KL penalty (含 LR scheduler 步进修复、value loss 标量化) |
| 强化学习 | GRPOTrainer | 组内 advantage 归一化，无 critic |
| 奖励模型 | RewardModelTrainer | Pairwise ranking loss |
| 高效微调 | LoRA | 低秩适配，支持 apply/merge/unmerge |
| 分布式 | FSDP FULL_SHARD + DDP | BF16 混合精度 + Activation Checkpointing，已适配 2×32GB 双卡 |
| 蒸馏数据 | DistillationPipeline | 5 类种子 + 6 种策略 → Teacher → SFT 数据 |

## 工程路线图

```
第一阶段 ✅ (已完成)
  现代架构重写 (RoPE + GQA + RMSNorm + SwiGLU)
  统一训练框架 (TrainerBase / SFT / DPO / Distill / LoRA)
  FSDP 分布式 + BF16 混合精度
  分词器升级 (SentencePiece + Fallback)
  完整数据管线 (采集→清洗→合成→格式化→版本管理)
  自动化评测 (MMLU / C-Eval / HumanEval)
  OpenAI 兼容 API (+ Function Calling)
  WebUI (深/亮主题, 竞技场, 思考可视化, 移动端适配)
  Docker 容器化 + docker-compose

第二阶段 ✅ (已完成)
  vLLM 深度集成
    Prefix Caching (自动缓存公共前缀)
    Speculative Decoding (Draft/N-Gram/EAGLE)
    量化加载 (AWQ / GPTQ / FP8)
    多 LoRA 动态切换 (热加载/卸载)
    Server 模式 (健康检查 + 自动重连)
    GPU 自动检测 + 推荐配置
  蒸馏管线 (代码/推理/深度推理/反幻觉/身份认知)
  PPO / GRPO 强化学习
  Reward Model 训练
  长上下文 128K+ (NTK-aware RoPE)
  CI/CD 流水线

第三阶段 ✅ (已完成)
   PPO / GRPO 强化学习
   Reward Model 训练
   长上下文 128K+ (NTK-aware RoPE)
   CI/CD 流水线
   全链路 3 轮代码检修 (修复 15+ Bug)
   **kwargs 推理参数转发 (蒸馏管线参数不再丢失)
   WebUI 异步改造 (同步生成器不阻塞事件循环)
   CLI `train-tokenizer` 子命令
   双卡 2×32GB 分布式适配

第四阶段 🚧 (规划中)
   多模态扩展 (视觉/语音输入)
   模型注册表 + 版本管理
   vLLM 生产级 Speculative Decoding 调优
```

## 技术栈

| 领域 | 技术选型 |
|------|----------|
| 深度学习 | PyTorch 2.0+ |
| 分布式 | FSDP FULL_SHARD / DDP |
| 推理 | vLLM 0.6+ / Ollama / llama.cpp / HuggingFace / 亦API |
| 服务 | FastAPI + SSE + httpx |
| 前端 | 原生 HTML/CSS/JS (响应式, 3 断点) |
| 分词 | SentencePiece 32000 |
| 评测 | MMLU / C-Eval / HumanEval (沙箱) |
| 部署 | Docker + docker-compose |
| 日志 | Python logging (EMIND_LOG_LEVEL) |
| 量化 | AWQ / GPTQ / FP8 |

## 贡献

欢迎提交 Issue 和 Pull Request。请确保：

1. 代码通过 ruff lint 检查
2. 新增功能包含单元测试
3. 保持向后兼容

## 许可

[MIT License](LICENSE)

---

<div align="center">
  <strong>亦梓科技 © 2026 Emind AI · 思考 · 创造 · 超越</strong>
</div>
