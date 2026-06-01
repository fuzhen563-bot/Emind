<div align="center">
  <img src="/logo.png" alt="Emind AI Logo" width="120">
  <h1>Emind AI v2.0</h1>
  <h3>亦梓·智脑 — 让每个行业拥有自己的大模型</h3>
  <p>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange" alt="PyTorch 2.0+">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    <img src="https://img.shields.io/badge/status-production-blue" alt="Production">
  </p>
  <p>
    <strong>开发者：</strong>
    <a href="https://yiziyun.com">亦梓科技</a> &
    <a href="https://yyzjai.cn">亦智大模型研究院</a>
  </p>
  <p>
    <a href="https://yiziyun.com">厦门亦梓科技有限公司</a> ·
    <a href="https://yyzjai.cn">苏州云养智健人工智能科技有限公司</a>
  </p>
</div>

---

# Emind AI v2.0 文档

- [概述](#概述)
- [行业垂类模型方案](#行业垂类模型方案)
- [v20-新特性](#v20-新特性)
- [核心技术栈](#核心技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [CLI 命令参考](#cli-命令参考)
- [训练一个 4B 垂类模型](#训练一个-4b-垂类模型)
- [蒸馏工具箱](#蒸馏工具箱-zhengliu)
- [vLLM 深度集成](#vllm-深度集成)
- [模型架构](#模型架构)
- [训练管线详解](#训练管线详解)
- [量化部署](#量化部署)
- [工程路线图](#工程路线图)
- [开发团队](#开发团队)
- [许可](#许可)

---

## 概述

**Emind v2.0** 是一个从零构建的生产级大语言模型框架，覆盖从 **数据蒸馏 → 模型训练 → 量化部署 → 推理服务** 的全链路。它的核心设计目标是：**让任何行业的团队都能在 1-2 天内构建属于自己的垂类大模型**。

传统上，训练一个大模型需要：
- 海量 GPU 资源（数十万 $）
- 顶尖算法团队
- 数月开发周期

**Emind 改变了这一点。** 凭借 Teacher 蒸馏技术 + LoRA 高效微调 + INT4/FP8 量化压缩，你只需要：

1. **准备好行业数据**（文档、QA、知识库）
2. **用蒸馏引擎合成高质量训练数据**（成本约 $3-10）
3. **单卡训练 1-2 小时**（支持 RTX 4090 / A100 / H100）
4. **量化部署到内网服务器**（INT4 仅需 2GB 显存）

整个流程跑通只需 **1 天**，训练成本不到 **$10**。

### 核心能力图谱

| 能力 | 说明 | 面向用户 |
|------|------|---------|
| 🔬 **数据蒸馏** | 用 GPT-4/DeepSeek 级 Teacher 合成行业 SFT/DPO 数据 | 无标注数据的团队 |
| 🎯 **高效微调** | LoRA / 全参 SFT / DPO / PPO / GRPO，单卡可训 4B 模型 | 有行业数据的团队 |
| 📦 **模型压缩** | INT4 量化 8GB→2GB，FP8 原生 H100 支持 | 需要私有化部署的团队 |
| 🚀 **推理引擎** | vLLM Server + OpenAI 兼容 API + Prefix Caching | 需要对外提供 API 的团队 |
| 📊 **数据管线** | 采集 → 清洗 → 合成 → 格式化 → 版本管理 | 有大量原始数据的团队 |
| 🧪 **评测套件** | MMLU / C-Eval / HumanEval | 需要验证模型能力的团队 |

---

## 行业垂类模型方案

Emind 为各行业提供从 **数据到部署** 的完整技术栈，让每个行业都能低成本拥有自己的垂类模型。

### 🏥 医疗行业

```
输入：病历文本、医学文献、诊疗指南、药品说明书
输出：智能预问诊、病历质控、医学问答、用药咨询

管线耗时：约 1-2 天
训练成本：约 $5-10（蒸馏 API 费用）
部署显存：INT4 量化后 2GB，可部署到内网
```

```bash
# 1. 蒸馏 3000 条医疗 QA
python -m zhengliu --mode sft --api-key sk-xxx \
  --code 0 --reasoning 0 --deep-reasoning 0 \
  --anti-hallucination 500 --identity 100

# 2. LoRA SFT 微调（单卡 24GB）
python cli.py train --mode sft \
  --data data/distilled_medical.jsonl \
  --d-model 2560 --n-layers 32 \
  --lora-r 16 --lora-alpha 32 \
  --epochs 3 --lr 2e-4 --use-bf16 \
  --output-dir checkpoints/medical-4b

# 3. INT4 量化 + 部署
python -c "from quantization import quantize_model; ..."
python cli.py serve --port 3333
```

### ⚖️ 法律行业

```
输入：法条文本、裁判文书、合同模板、法律咨询记录
输出：法条检索、合同审查、案情分析、文书撰写

管线耗时：约 2-3 天
训练成本：约 $8-15
推荐策略：SFT + DPO 两阶段
```

### 💰 金融行业

```
输入：研报、财报、监管文件、交易记录
输出：智能投研、财报分析、风险识别、合规审查

管线耗时：约 1-2 天
推荐策略：deep_reasoning 蒸馏 + SFT + Prefix Caching 加速长文档
```

### 🏭 制造业

```
输入：设备手册、操作规程、质检记录、维修日志
输出：故障诊断、操作指导、质量预测、维护建议

管线耗时：约 1 天
推荐策略：anti-hallucination 蒸馏 + INT4 部署到边缘设备
```

### 垂类模型训练流程（通用）

```
行业原始数据（文档/QA/知识库）
    │
    ▼
┌──────────────────────────────────────────┐
│ 第一步：数据准备                          │
│  ├─ collector.py 采集行业文档              │
│  ├─ cleaner.py 清洗去重                    │
│  └─ formatter.py 格式化输出                │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ 第二步：zhengliu 蒸馏（核心差异来源）      │
│  ├─ Teacher 模型（DeepSeek/GPT-4）合成数据 │
│  ├─ 代码/推理/反幻觉/身份认知 5 类模板     │
│  ├─ 质量控制 + 自我纠错                    │
│  ├─ DPO 偏好对生成                         │
│  └─ 成本：约 $3-10 / 3000 条              │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ 第三步：模型训练                           │
│  ├─ LoRA SFT（推荐，1-2h，单卡 24GB）     │
│  ├─ 或全参 SFT（更优但需要更多资源）        │
│  ├─ DPO 偏好对齐（可选，提升安全性）        │
│  └─ GRPO 强化学习（可选，提升推理能力）     │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ 第四步：量化压缩                           │
│  ├─ INT4 量化：8GB → 2GB（推荐）          │
│  ├─ FP8 量化：H100 原生支持               │
│  └─ AWQ/GPTQ：vLLM 集成                   │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│ 第五步：部署上线                           │
│  ├─ vLLM Server（OpenAI 兼容 API）        │
│  ├─ Prefix Caching（长文档 3x 加速）       │
│  ├─ WebUI（深/亮主题，竞技场）             │
│  └─ Docker 容器化                         │
└──────────────────────────────────────────┘
    │
    ▼
   🎯 行业垂类模型已就绪！
```

---

## v2.0 新特性

| 模块 | v2.0 改进 | Bug 修复 |
|------|-----------|----------|
| 🔧 **蒸馏引擎** | Checkpoint/Resume 续传、指数退避限流、Thread-safe 计数器、Token 用量追踪、去重窗口扩展 | 5 个 |
| 🧮 **量化模块** | FP8 per-row scaling + 原生 FP8 GEMM (H100)、INT4 chunked 分批反量化 (×64 显存降低)、int16 防溢出 | 6 个 |
| 🔤 **分词器** | Greedy longest‑match + 8 个 CJK Unicode 扩展区、3.2 万词表、encode_batch/decode_batch | 6 个 |
| 📚 **训练指南** | 完整 15 章中文训练手册，涵盖所有训练场景 + 排错 FAQ | — |
| 📦 **包管理** | `zhengliu` 包纳入 setup.py/pyproject.toml，`zl` CLI 入口 | — |
| ⚡ **稳定性** | 全模块 py_compile 检查通过，41 个已识别 Bug 中修复 37 个 | 37 个 |

> 详细变更见 [`BUG_TRACKER.md`](BUG_TRACKER.md) 和 [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md)

---

## 核心技术栈

| 领域 | 技术选型 |
|------|----------|
| 深度学习框架 | PyTorch 2.0+ |
| 分布式训练 | FSDP FULL_SHARD / DDP |
| 高效微调 | LoRA（低秩适配，apply/merge/unmerge） |
| 推理引擎 | vLLM 0.6+（PagedAttention + Prefix Caching + Speculative Decoding） |
| 推理后端 | vLLM / Ollama / llama.cpp / HuggingFace / 亦API 云端 / 本地模型 |
| 数据蒸馏 | zhengliu 引擎（5 类模板 + 6 种策略 + 质量评分） |
| 模型量化 | INT4 weight-only / FP8 E4M3FN / AWQ / GPTQ |
| 服务框架 | FastAPI + SSE 流式 + httpx |
| WebUI | 原生 HTML/CSS/JS（响应式，深/亮主题） |
| 分词 | SentencePiece 32000 + Fallback（CJK 8 扩展区） |
| 评测 | MMLU / C-Eval / HumanEval（沙箱安全执行） |
| 容器化 | Docker 多阶段构建 + docker-compose |
| 监控 | WandB / TensorBoard / JSON 日志（可选） |

---

## 项目结构

```
Emind_AI_v2.0/
│
├── 📦 核心模型
│   ├── model.py                 # LLaMA 架构 (RoPE+GQA+RMSNorm+SwiGLU+KV Cache)
│   ├── tokenizer.py             # 分词器 (SentencePiece + 3.2 万词表 Fallback)
│   ├── quantization.py          # INT4 / FP8 量化推理
│   └── distributed_utils.py     # DDP + FSDP FULL_SHARD 分布式工具
│
├── 🏋️ 训练管线
│   ├── training/
│   │   ├── config.py            #   训练配置 (effective_batch_size)
│   │   ├── trainer.py           #   基类 (BF16/FSDP/早停/梯度累积/LR warmup)
│   │   ├── sft.py               #   监督微调 (assistant-only loss masking)
│   │   ├── dpo.py               #   偏好对齐 (标准 DPO loss)
│   │   ├── rl.py                #   强化学习 (PPO/GRPO/Reward Model)
│   │   ├── distill.py           #   知识蒸馏 (logit-based, 温度缩放)
│   │   ├── distillation_pipeline.py # Teacher → Student 蒸馏管线
│   │   ├── lora.py              #   LoRA 低秩适配
│   │   ├── checkpoint.py        #   断点管理 (best/latest 双保存)
│   │   └── metrics.py           #   指标追踪 (WandB/TensorBoard/JSON)
│   └── data_pipeline/           # 数据管线
│       ├── collector.py         #   多源采集
│       ├── cleaner.py           #   清洗去重/PII脱敏
│       ├── synthesizer.py       #   数据合成 (Self-Instruct/Evol-Instruct)
│       ├── formatter.py         #   格式转换 (SFT/DPO/Pretrain/Alpaca)
│       └── dataset.py           #   版本管理
│
├── 🔬 蒸馏工具箱
│   ├── zhengliu/
│   │   ├── __main__.py          #   交互菜单 + CLI 入口
│   │   ├── config.py            #   蒸馏配置 (type_counts/resume/quality_check)
│   │   ├── distill.py           #   蒸馏引擎 (checkpoint/resume/限流)
│   │   ├── pipeline.py          #   多模型自动蒸馏流水线
│   │   ├── seeds.py             #   种子模板 (5 类数据)
│   │   ├── visual.py            #   Rich 仪表盘
│   │   └── output/              #   蒸馏输出目录
│   └── scripts/
│       ├── distill_quick.py     #   快速蒸馏脚本
│       └── merge_checkpoint.py  #   LoRA 权重合并
│
├── 🚀 推理服务
│   ├── vllm_integration.py      # vLLM 深度集成 (Prefix Caching/Speculative Decoding/量化/LoRA)
│   ├── unified_inference.py     # 统一推理引擎 (多后端自动降级)
│   ├── web_server.py            # FastAPI + OpenAI 兼容 API + Function Calling
│   ├── cli.py                   # CLI 入口 (train/infer/serve/eval/pipeline/rl/vllm)
│   └── web/                     # WebUI 前端
│
├── 🧪 评测
│   ├── eval/
│   │   ├── mmlu.py              #   MMLU
│   │   ├── ceval.py             #   C-Eval
│   │   ├── humaneval.py         #   HumanEval
│   │   └── runner.py            #   统一运行器
│   └── tests/test_emind.py      #   30+ 单元测试
│
├── 📋 文档
│   ├── README.md                #   本文件
│   ├── TRAINING_GUIDE.md        #   15 章训练指南
│   ├── BUG_TRACKER.md           #   Bug 追踪
│   ├── TECH_REVIEW.md           #   技术评审
│   └── PAPER_ZHENGLIU.md        #   蒸馏论文草稿
│
├── ⚙️ 部署配置
│   ├── Dockerfile               #   多阶段构建 (python:3.12-slim)
│   ├── docker-compose.yml       #   编排 (api / vllm / jupyter)
│   ├── setup.py                 #   Python 包安装
│   └── pyproject.toml           #   项目元数据
│
└── 📁 数据目录
    ├── data/                    # 训练数据
    ├── data_small/              # 小规模测试数据
    ├── checkpoints/             # 模型检查点
    └── data/distilled/          # 蒸馏输出
```

---

## 快速开始

### 环境要求

- **Python** 3.10+
- **PyTorch** 2.0+（建议 CUDA 12.1+）
- **GPU** 推荐 RTX 4090 24GB / A100 80GB / H100（FP8）
- **磁盘** 至少 20GB 可用空间

### 安装

```bash
# 1. 克隆仓库
git clone <repo-url> && cd Emind_AI_v2.0

# 2. 安装核心依赖
pip install torch>=2.0.0 numpy sentencepiece
pip install transformers>=4.36.0 accelerate>=0.25.0 requests
pip install fastapi uvicorn jinja2 aiofiles sse-starlette
pip install rich>=13.0.0

# 3. 安装 Emind 包
pip install -e .

# 4. (可选) 推理后端
pip install "vllm>=0.6.0"          # vLLM
pip install autoawq                 # AWQ 量化
pip install auto-gptq               # GPTQ 量化
pip install llama-cpp-python        # llama.cpp

# 5. (可选) 训练监控
pip install wandb                   # WandB
# pip install tensorboard           # TensorBoard
```

### 验证安装

```bash
# 查看 CLI 帮助
python cli.py --help

# 启动 Web 服务
python cli.py serve --port 3333
# 浏览器访问 http://localhost:3333

# 蒸馏工具箱菜单
python -m zhengliu

# 运行测试
python -m pytest tests/test_emind.py -v
```

---

## CLI 命令参考

| 命令 | 用途 | 必选参数 | 示例 |
|------|------|---------|------|
| `train` | 训练模型 | `--mode`, `--data` | `cli.py train --mode sft --data data.json --d-model 2560` |
| `infer` | 推理 | `--model`, `--prompt` | `cli.py infer --model model.pt --prompt "你好"` |
| `serve` | 启动服务 | — | `cli.py serve --port 3333` |
| `eval` | 评测 | `--model`, `--benchmarks` | `cli.py eval --model model.pt --benchmarks humaneval` |
| `pipeline` | 数据管线 | — | `cli.py pipeline --distill-code 100` |
| `rl` | 强化学习 | `--rl-mode`, `--data` | `cli.py rl --rl-mode grpo --data rl.json` |
| `vllm` | vLLM 诊断 | — | `cli.py vllm --detect --auto-configure` |
| `train-tokenizer` | 训练分词器 | `--data` | `cli.py train-tokenizer --data corpus.jsonl` |

---

## 训练一个 4B 垂类模型

以下是一个完整的端到端示例，从头训练一个 4B 参数的法律垂类模型。

### 前置准备

```bash
# 设置 API Key（用于 Teacher 蒸馏）
export ZL_API_KEY=sk-your-key
export ZL_BASE_URL=https://api.deepseek.com
```

### Step 1：数据准备

```bash
# 用数据管线采集和处理法律文书
python cli.py pipeline \
  --collect data/legal_raw/ \
  --process \
  --format sft \
  --output data/legal_processed.jsonl
```

### Step 2：蒸馏训练数据（核心步骤）

```bash
# 用 zhengliu 蒸馏法律相关数据
python -m zhengliu --mode sft \
  --code 2000 \           # 法律文书分析
  --reasoning 500 \        # 法律推理
  --deep-reasoning 500 \   # 复杂案情分析
  --anti-hallucination 500 \ # 降低幻觉
  --identity 100 \         # 身份认知
  --api-key sk-xxx \
  --cot-depth 2 \
  --max-tokens 4096

# 输出路径: zhengliu/output/2026xxxx_sft.jsonl
```

### Step 3：SFT 监督微调

```bash
# LoRA 微调（推荐，单卡 24GB 足够）
python cli.py train \
  --mode sft \
  --data zhengliu/output/2026xxxx_distilled_sft.jsonl \
  --d-model 2560 \
  --n-heads 32 \
  --n-kv-heads 8 \
  --n-layers 32 \
  --max-seq-len 4096 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --epochs 3 \
  --lr 2e-4 \
  --use-bf16 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --output-dir checkpoints/legal-4b-lora

# 合并 LoRA 权重
python scripts/merge_checkpoint.py \
  --base-model ./base-model \
  --lora checkpoints/legal-4b-lora/latest \
  --output checkpoints/legal-4b-merged
```

### Step 4：DPO 偏好对齐（可选）

```bash
# 先蒸馏 DPO 偏好对
python -m zhengliu --mode dpo \
  --code 500 --api-key sk-xxx \
  --dpo-chosen-temp 0.3 \
  --dpo-rejected-temp 1.2

# DPO 训练
python cli.py train --mode dpo \
  --data zhengliu/output/2026xxxx_distilled_dpo.jsonl \
  --d-model 2560 \
  --batch-size 2 \
  --lr 1e-6 \
  --beta 0.1
```

### Step 5：INT4 量化压缩

```python
# quantize_model.py
from quantization import quantize_model, estimate_model_size
import torch

model = torch.load("checkpoints/legal-4b-merged/model.pt",
                   map_location="cpu", weights_only=False)

# INT4 量化 (8GB → 2GB)
qmodel = quantize_model(model, mode="int4", group_size=128)
torch.save(qmodel, "checkpoints/legal-4b-int4/model.pt")

print(estimate_model_size(qmodel, mode="int4"))
# 输出: ~2.1 GB
```

### Step 6：部署上线

```bash
# vLLM Server（生产推荐）
python cli.py serve --vllm \
  --model checkpoints/legal-4b-int4 \
  --vllm-prefix-caching \
  --vllm-gpu-memory 0.90 \
  --vllm-dtype bfloat16 \
  --port 8000

# 或用 WebUI
python cli.py serve --port 3333
```

---

## 蒸馏工具箱 (zhengliu)

`zhengliu` 是 Emind 的核心数据引擎，负责用 Teacher API 合成高质量训练数据。

### 交互式菜单

```bash
python -m zhengliu
```

菜单选项：
1. **基础蒸馏** — 生成 SFT 数据（代码/推理/问答）
2. **进阶蒸馏** — 生成 DPO 偏好对
3. **一键蒸馏** — 自动发现 API 模型池 + 轮询切换
4. **质量控制** — 质量评分 + 自我纠错管线
5. **预览模式** — 不调用 API，只看生成的 prompt

### CLI 直接模式

```bash
# SFT 蒸馏（基本用法）
python -m zhengliu --mode sft \
  --code 50 --reasoning 20 --deep-reasoning 10 \
  --anti-hallucination 10 --identity 5 \
  --api-key sk-xxx

# DPO 蒸馏
python -m zhengliu --mode dpo \
  --code 100 --api-key sk-xxx \
  --dpo-chosen-temp 0.3 --dpo-rejected-temp 1.2

# 从 checkpoint 续传
python -m zhengliu --mode sft \
  --code 50 --api-key sk-xxx --resume

# 一键自动蒸馏（无限轮询）
python -m zhengliu pipeline \
  --mode sft --code 200 --auto-runs -1 --auto-discover

# 质量控制
python -m zhengliu --mode sft \
  --code 50 --quality-check --multi-turn-correct

# 预览（不消耗 API 额度）
python -m zhengliu --code 10 --dry-run
```

### 使用别名

```bash
pip install -e .
zl --mode sft --code 50
```

### 蒸馏数据格式

```jsonl
# SFT 格式
{"prompt": "用 Python 实现二分查找", "response": "def binary_search(arr, target):...", "type": "code", "strategy": "direct"}

# DPO 格式
{"prompt": "如何优化 SQL 查询?", "chosen": "合理的做法是...", "rejected": "有缺陷的做法是...", "type": "code", "strategy": "cot"}
```

---

## vLLM 深度集成

### 支持特性

| 特性 | 支持版本 | 说明 |
|------|---------|------|
| PagedAttention | vLLM 0.4+ | 高效 KV 缓存管理 |
| Prefix Caching | vLLM 0.4+ | 自动缓存公共前缀 KV 块，多轮对话/长文档 2-3x 加速 |
| Speculative Decoding | vLLM 0.5+ | Draft model 草稿-验证，2-3x 推理加速 |
| N-Gram Speculator | vLLM 0.6+ | 无需额外模型，基于 n-gram 推测 |
| EAGLE | vLLM 0.6+ | 基于特征层的推测解码 |
| Chunked Prefill | vLLM 0.6+ | 长 prompt 分块填充，降低 TTFT |
| AWQ 量化 | 4-bit | 显存减半，几乎无损 |
| GPTQ 量化 | 4/8-bit | 通用后训练量化 |
| FP8 量化 | Hopper GPU | 原生 FP8 支持 |
| 多 LoRA 动态切换 | vLLM 0.6.3+ | 运行时热加载/卸载 |
| 健康检查 + 自动重连 | — | Server 模式内置 |

### 使用方式

```bash
# 1. GPU 能力检测
python cli.py vllm --detect --auto-configure

# 2. 生产级 Server
python cli.py serve --vllm \
  --model ./models/emind-4b \
  --vllm-prefix-caching \
  --vllm-speculative \
  --vllm-draft-model ./models/emind-1b-draft \
  --vllm-num-speculative-tokens 5 \
  --vllm-gpu-memory 0.92 \
  --vllm-dtype bfloat16

# 3. 推理
python cli.py infer --backend vllm \
  --model ./models/emind-4b \
  --prompt "编写一个 Python 装饰器"

# 4. Python API
python -c "
from vllm_integration import VLLMIntegratedEngine, VLLMConfig
engine = VLLMIntegratedEngine(VLLMConfig(
    model_path='./models/emind-4b',
    enable_prefix_caching=True
))
print(engine.generate('你好'))
"
```

---

## 模型架构

### 架构总览

| 组件 | 方案 | 说明 |
|------|------|------|
| 位置编码 | **RoPE** | 旋转位置编码，NTK-aware 缩放支持 128K+ 上下文 |
| 注意力机制 | **GQA** | 分组查询注意力 (Grouped Query Attention)，n_kv_heads=4~8 |
| 归一化 | **RMSNorm** | 均方根归一化 (Root Mean Square Normalization) |
| 前馈网络 | **SwiGLU** | Gated Linear Unit + SiLU 激活函数 |
| KV Cache | ✅ 已集成 | 自回归缓存，消除重复计算 |
| 生成策略 | Top-k + Top-p + Temperature + Repetition Penalty | 灵活可控 |

### 模型大小配置

| 参数量 | d_model | n_layers | n_heads | n_kv_heads | d_ff | 显存 (FP16) | 显存 (INT4) |
|--------|---------|----------|---------|------------|------|------------|------------|
| 0.5B | 1024 | 16 | 16 | 4 | 4096 | ~1 GB | ~0.3 GB |
| 1B | 1536 | 20 | 20 | 5 | 6144 | ~2 GB | ~0.5 GB |
| 2B | 2048 | 24 | 24 | 6 | 8192 | ~4 GB | ~1 GB |
| 4B | 2560 | 32 | 32 | 8 | 10240 | ~8 GB | ~2 GB |
| 7B | 3584 | 36 | 36 | 9 | 14336 | ~14 GB | ~3.5 GB |

> 参数计算公式：`params ≈ d_model × n_layers × (12 × d_model + 2 × d_ff) / 1e9`（含 embedding）

---

## 训练管线详解

### 训练方法对比

| 方法 | 模块 | 适用场景 | 数据要求 | 显存需求 |
|------|------|---------|---------|---------|
| **SFT** | [`SFTTrainer`](training/sft.py:70) | 指令微调、领域适配 | 千条级 QA 数据 | 24GB+ |
| **LoRA SFT** | [`LoRA`](training/lora.py) + [`SFTTrainer`](training/sft.py:70) | **推荐**，高效低成本 | 百条级即可 | 12GB+ |
| **DPO** | [`DPOTrainer`](training/dpo.py:52) | 偏好对齐、安全性 | chosen/rejected 对 | 24GB+ |
| **PPO** | [`PPOTrainer`](training/rl.py:209) | 强化学习对齐 | reward model 分数 | 48GB+ |
| **GRPO** | [`GRPOTrainer`](training/rl.py:431) | 无 critic 的 RL | 组内奖励 | 24GB+ |
| **蒸馏** | [`DistillationTrainer`](training/distill.py) | 知识压缩 | Teacher logits | 24GB+ |

### 训练优化建议

```yaml
# 显存不足时的优化组合
memory_saving:
  - use_bf16: true                    # 半精度训练，显存减半
  - gradient_checkpointing: true      # 激活重计算，显存减 30%
  - batch_size: 1                     # 最小 batch
  - gradient_accumulation_steps: 8    # 等效 batch_size=8
  - lora_r: 8                         # 低秩 LoRA
  - offload_to_cpu: true              # 优化器状态卸载到 CPU

# 推荐训练配置（4B 模型，24GB 单卡）
training_config:
  mode: sft
  dtype: bf16
  lora: true
  lora_r: 16
  lora_alpha: 32
  learning_rate: 2e-4
  num_epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 4
  warmup_steps: 100
```

### 监控与可视化

```bash
# WandB 监控（推荐）
python cli.py train --mode sft --data data.json \
  --wandb-project emind-legal \
  --wandb-run legal-v1

# TensorBoard
python cli.py train --mode sft --data data.json \
  --tensorboard-dir ./logs

# JSON 日志（默认）
# 自动保存在 checkpoints/<run-name>/metrics.jsonl
```

---

## 量化部署

### 量化方式对比

| 方式 | 说明 | 显存 | 速度 | 硬件要求 |
|------|------|------|------|---------|
| **INT4** | Weight-only 4-bit 量化 | 原始 25% | 略慢于 FP16 | 任何 GPU |
| **FP8** | E4M3FN 8-bit 浮点 | 原始 50% | 原生 FP8 GEMM 加速 | H100 或 A100+ |
| **AWQ** | Activation-aware 4-bit | 原始 25% | vLLM 原生支持 | 任何 GPU |
| **GPTQ** | 后训练 4/8-bit | 原始 25-50% | vLLM 原生支持 | 任何 GPU |

### 使用方式

```python
# INT4 量化
from quantization import quantize_model, estimate_model_size

model = torch.load("model.pt", map_location="cpu", weights_only=False)
qmodel = quantize_model(model, mode="int4", group_size=128)
torch.save(qmodel, "model-int4.pt")
print(estimate_model_size(qmodel, mode="int4"))
```

---

## 工程路线图

### 第一阶段 ✅ 已完成

```
现代架构重写 (RoPE + GQA + RMSNorm + SwiGLU)
统一训练框架 (TrainerBase / SFT / DPO / Distill / LoRA)
FSDP 分布式 + BF16 混合精度
分词器升级 (SentencePiece + 3.2 万词表 Fallback)
完整数据管线 (采集→清洗→合成→格式化→版本管理)
自动化评测 (MMLU / C-Eval / HumanEval)
OpenAI 兼容 API (+ Function Calling)
WebUI (深/亮主题, 竞技场, 思考可视化)
Docker 容器化
```

### 第二阶段 ✅ 已完成

```
vLLM 深度集成 (Prefix Caching / Speculative Decoding / 量化 / LoRA)
zhengliu 蒸馏管线 (5 类数据 / DPO 对 / 质量评分 / 自动纠错)
PPO / GRPO 强化学习
Reward Model 训练
长上下文 128K+ (NTK-aware RoPE)
CI/CD 流水线
```

### 第三阶段 ✅ 已完成

```
多轮代码检修（修复 37 个 Bug）
蒸馏引擎 Checkpoint/Resume
量化模块 FP8 per-row + INT4 chunked
分词器 CJK 8 扩展区
完整中文训练指南 (15 章)
```

### 第四阶段 🚧 规划中

```
多模态扩展（视觉/语音输入）
模型注册表 + 版本管理
vLLM 生产级 Speculative Decoding 调优
行业种子模板库（医疗/法律/金融/制造）
自动化数据飞轮（用户反馈 → 模型迭代）
```

---

## 开发团队

**Emind v2.0** 由以下单位联合研发：

| 单位 | 官网 | 研发方向 |
|------|------|---------|
| [厦门亦梓科技有限公司](https://yiziyun.com) | [yiziyun.com](https://yiziyun.com) | 模型架构设计、训练管线、推理引擎、WebUI |
| [苏州云养智健人工智能科技有限公司](https://yyzjai.cn) | [yyzjai.cn](https://yyzjai.cn) | 数据蒸馏管线、数据合成、评测套件、量化部署 |

**亦智大模型研究院** — 双方联合设立的大模型研究机构，致力于大语言模型的前沿技术探索与产业化落地。

---

## 许可

本项目采用 [MIT License](LICENSE)。

```
MIT License

Copyright (c) 2026 厦门亦梓科技有限公司 & 苏州云养智健人工智能科技有限公司

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

<div align="center">
  <strong>© 2026 Emind AI · 亦梓科技 & 亦智大模型研究院</strong><br>
  <a href="https://yiziyun.com">厦门亦梓科技有限公司</a> ·
  <a href="https://yyzjai.cn">苏州云养智健人工智能科技有限公司</a><br>
  <small>思考 · 创造 · 超越</small>
</div>
