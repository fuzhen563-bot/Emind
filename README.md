<div align="center">
  <img src="web/static/logo_sm.png" alt="Emind AI Logo" width="120" height="120">
  <h1>Emind AI</h1>
  <p><strong>亦梓·智脑 — 从研究原型到生产级大模型</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange" alt="PyTorch 2.0+">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    <img src="https://img.shields.io/badge/status-beta-yellow" alt="Beta">
  </p>
</div>

---

Emind 是一个从零开始构建的大语言模型框架，覆盖模型训练、推理服务、WebUI 交互全链路。采用现代 LLaMA 架构（RoPE + GQA + RMSNorm + SwiGLU），支持单卡到多卡分布式训练（DDP / FSDP），并提供统一的训练器抽象层消除重复代码。

## ✨ 特性

- **🧠 现代架构** — RoPE 位置编码、GQA 分组查询注意力、RMSNorm、SwiGLU FFN、KV Cache
- **🚀 训练管线** — 统一 TrainerBase + SFT / DPO / Distillation，内置 FSDP + BF16 混合精度
- **🔌 多推理后端** — 本地模型、vLLM、Ollama、llama.cpp、HuggingFace、亦API 云端
- **📊 数据管线** — 多源采集 → 清洗去重 → 质量过滤 → 数据合成 → 格式化输出
- **🌐 WebUI** — FastAPI + SSE 流式响应，暗/亮主题、会话管理、模型对比竞技场
- **🧪 评测套件** — MMLU / C-Eval / HumanEval 一键评测
- **🐳 Docker 部署** — docker-compose 一键启动
- **🔐 OAuth2 登录** — 支持亦梓科技聚合登录

## 🏗️ 项目结构

```
Emind_AI_v2.0/
├── model.py                 # 核心模型 (RoPE+GQA+RMSNorm+SwiGLU)
├── tokenizer.py             # 分词器 (SentencePiece + Fallback)
├── logger.py                # 统一日志
├── distributed_utils.py     # 分布式训练工具 (DDP + FSDP)
│
├── training/                # 统一训练框架
│   ├── config.py            #   训练配置
│   ├── trainer.py           #   训练器基类 (BF16/FSDP/早停)
│   ├── sft.py               #   监督微调 (loss masking)
│   ├── dpo.py               #   偏好对齐 (DPO)
│   ├── distill.py           #   知识蒸馏
│   ├── lora.py              #   LoRA 微调
│   ├── checkpoint.py        #   断点管理
│   └── metrics.py           #   指标追踪
│
├── data_pipeline/           # 数据管线
│   ├── collector.py         #   多源采集
│   ├── cleaner.py           #   清洗/PII/去重
│   ├── synthesizer.py       #   数据合成
│   ├── formatter.py         #   格式转换
│   └── dataset.py           #   版本管理
│
├── eval/                    # 评测套件
│   ├── mmlu.py              #   MMLU
│   ├── ceval.py             #   C-Eval
│   ├── humaneval.py         #   HumanEval
│   └── runner.py            #   统一运行器
│
├── unified_inference.py     # 统一推理引擎 (vLLM/Ollama/llama.cpp/HF/Cloud)
├── web_server.py            # FastAPI + OpenAI 兼容 API
├── web/                     # WebUI 前端
├── cli.py                   # CLI (train/infer/serve/eval/pipeline)
│
├── Dockerfile               # 多阶段构建
├── docker-compose.yml       # 编排部署
│
├── tests/test_emind.py      # 单元测试
├── Emind/                   # Python 包入口 (from Emind import *)
└── 01_pretrain.py ~ 07_*.py # 训练脚本 (薄封装)
```

## 📦 安装

```bash
git clone https://github.com/your-org/emind.git
cd emind

# 核心依赖
pip install torch>=2.0.0 sentencepiece

# 推理后端 (按需)
pip install llama-cpp-python   # llama.cpp
pip install transformers       # HuggingFace
pip install vllm               # vLLM (GPU)

# Web 服务
pip install fastapi uvicorn
```

## 🚀 快速开始

### 使用模型

```python
from model import EmindConfig, create_model
import torch

config = EmindConfig(vocab_size=32000, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6)
model = create_model(config)

input_ids = torch.tensor([[1, 100, 200, 300]])
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
)
```

### 启动 Web 服务

```bash
# CLI 方式
python cli.py serve

# 或直接启动
python web_server.py
# 打开 http://localhost:3333
```

### 训练一个模型

```python
from model import EmindConfig, create_model
from training import SFTTrainer, TrainingConfig, SFTDataset

dataset = SFTDataset(data, tokenizer, max_seq_len=2048)

cfg = EmindConfig(vocab_size=32000, d_model=768, n_heads=12, n_kv_heads=4, n_layers=12)
model = create_model(cfg)
train_cfg = TrainingConfig(batch_size=4, epochs=3, use_bf16=True)

trainer = SFTTrainer(model, train_cfg, dataset)
trainer.train()
```

### 数据处理管线

```bash
# 采集 → 清洗 → 格式化
python cli.py pipeline --collect data/raw --process --format sft

# 或使用 Python API
from data_pipeline import DataCollector, DataCleaner, DataFormatter
items = DataCollector().from_directory("data/raw")
items = DataCleaner().clean(items, target_lang="zh")
sft = DataFormatter().to_sft(items)
```

### 运行评测

```bash
python cli.py eval --model checkpoints/best/model.pt --benchmarks mmlu,ceval
```

### 分布式训练

```bash
torchrun --nproc_per_node=4 cli.py train --mode sft --data data/train.json --use-fsdp
```

### Docker 部署

```bash
docker-compose up -d
# 访问 http://localhost:8080
```

## 🏛️ 模型架构

| 组件 | 方案 | 说明 |
|------|------|------|
| 位置编码 | **RoPE** | 旋转位置编码，支持长度外推 |
| 注意力 | **GQA** | 分组查询注意力，n_kv_heads < n_heads |
| 归一化 | **RMSNorm** | 均方根归一化，比 LayerNorm 更高效 |
| 前馈网络 | **SwiGLU** | Gated Linear Unit + SiLU 激活 |
| KV Cache | ✅ 已集成 | 自回归生成时缓存 K/V，消除重复计算 |
| 生成策略 | Top-k + Top-p + Repetition Penalty | 灵活可控 |

## 🧪 训练管线

| 阶段 | 模块 | 说明 |
|------|------|------|
| 预训练 | SFTTrainer | 语言模型预训练 |
| 监督微调 | SFTTrainer | 指令数据微调，assistant-only loss masking |
| 偏好对齐 | DPOTrainer | 标准 DPO loss |
| 知识蒸馏 | DistillationTrainer | Logit-based 蒸馏 (温度缩放) |
| 高效微调 | LoRA | 低秩适配，支持 merge/unmerge |
| 分布式 | FSDP + DDP | FULL_SHARD + BF16 混合精度 + Activation Checkpointing |

## 📊 工程路线图

```
第一阶段 ✅ (已完成)
  ├── 现代架构重写 (RoPE + GQA + RMSNorm + SwiGLU)
  ├── 统一训练框架 (TrainerBase / SFT / DPO / Distill / LoRA)
  ├── FSDP 分布式支持
  ├── 分词器升级 (SentencePiece + Fallback)
  ├── vLLM 推理引擎集成
  ├── 完整数据管线 (采集→清洗→合成→格式化→版本管理)
  ├── 自动化评测 (MMLU, C-Eval, HumanEval)
  ├── OpenAI 兼容 API (/v1/*)
  ├── WebUI 移动端适配 (3 断点 + 手势 + 键盘)
  └── Docker 容器化 + docker-compose

第二阶段 🚧 (进行中)
  ├── CI/CD 流水线
  ├── PPO / GRPO 强化学习
  ├── Function Calling / Tool Use
  ├── 长上下文 128K+ (NTK-aware RoPE)
  └── 多模态扩展
```

## 🔧 技术栈

| 领域 | 技术选型 |
|------|----------|
| 深度学习 | PyTorch 2.0+ |
| 分布式 | FSDP / DDP |
| 推理 | vLLM / Ollama / llama.cpp / HuggingFace |
| 服务 | FastAPI + SSE |
| 前端 | 原生 HTML/CSS/JS |
| 分词 | SentencePiece |
| 部署 | Docker + docker-compose |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。请确保：

1. 代码通过 ruff 检查
2. 新增功能包含单元测试
3. 保持向后兼容

## 📄 许可

[MIT License](LICENSE)

---

<div align="center">
  <strong>亦梓科技 © 2026 Emind AI · 思考 · 创造 · 超越</strong>
</div>
