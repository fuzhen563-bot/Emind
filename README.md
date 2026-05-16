<div align="center">
  <img src="/static/logo_sm.png" alt="Emind AI Logo" width="120" height="120">
  <h1>Emind AI</h1>
  <p><strong>亦梓·智脑 — 从研究原型到生产级大模型</strong></p>
  <p>
    <img src="https://img.shields.io/badge/python-3.12-blue" alt="Python 3.12">
    <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange" alt="PyTorch 2.0+">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
    <img src="https://img.shields.io/badge/status-alpha-yellow" alt="Alpha">
  </p>
</div>

---

Emind 是一个从零开始构建的大语言模型框架，覆盖模型训练、推理服务、WebUI 交互全链路。采用现代 LLaMA 架构（RoPE + GQA + RMSNorm + SwiGLU），支持单卡到多卡分布式训练（DDP / FSDP），并提供统一的训练器抽象层消除重复代码。

## ✨ 特性

- **🧠 现代架构** — RoPE 位置编码、GQA 分组查询注意力、RMSNorm、SwiGLU FFN、KV Cache
- **🚀 训练管线** — 统一 TrainerBase + SFT / DPO / Distillation，内置 FSDP + BF16 混合精度
- **🔌 多推理后端** — 本地模型、Ollama、llama.cpp、HuggingFace、亦API 云端
- **🌐 WebUI** — FastAPI + SSE 流式响应，暗/亮主题、会话管理、模型对比竞技场
- **🔐 OAuth2 登录** — 支持亦梓科技聚合登录
- **🪶 轻量可定制** — 无重型依赖，核心代码 1000+ 行，适合学习和二次开发

## 🏗️ 项目结构

```
Emind_AI_v2.0/
├── model.py                 # 核心模型定义
├── tokenizer.py             # 分词器 (SentencePiece + Fallback)
├── distributed_utils.py     # 分布式训练工具 (DDP + FSDP)
│
├── training/                # 统一训练框架
│   ├── config.py            #   训练配置
│   ├── trainer.py           #   训练器基类
│   ├── sft.py               #   监督微调
│   ├── dpo.py               #   偏好对齐
│   ├── distill.py           #   知识蒸馏
│   ├── checkpoint.py        #   断点管理
│   └── metrics.py           #   指标追踪
│
├── unified_inference.py     # 统一推理后端
├── web_server.py            # FastAPI 服务
├── web/                     # WebUI 前端
│
├── 01_pretrain.py ~ 07_*.py # 训练脚本
└── Emind/                   # Python 包入口
```

## 📦 安装

```bash
git clone https://github.com/your-org/emind.git
cd emind

# 核心依赖
pip install torch>=2.0.0

# 推理后端 (按需)
pip install llama-cpp-python   # llama.cpp
pip install transformers       # HuggingFace
pip install sentencepiece      # 分词器训练

# Web 服务
pip install fastapi uvicorn
```

## 🚀 快速开始

### 使用模型

```python
from model import EmindConfig, create_model

config = EmindConfig(vocab_size=32000, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6)
model = create_model(config)

input_ids = [[1, 100, 200, 300]]
output = model.generate(
    torch.tensor(input_ids),
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
)
```

### 启动 Web 服务

```bash
python web_server.py
# 打开 http://localhost:3333
```

### 训练一个模型

```python
from model import EmindConfig, create_model
from training import SFTTrainer, TrainingConfig, SFTDataset

# 数据
dataset = SFTDataset(data, tokenizer, max_seq_len=2048)

# 配置
cfg = EmindConfig(vocab_size=32000, d_model=768, n_heads=12, n_kv_heads=4, n_layers=12)
model = create_model(cfg)
train_cfg = TrainingConfig(batch_size=4, epochs=3, use_bf16=True)

# 训练
trainer = SFTTrainer(model, train_cfg, dataset)
trainer.train()
```

### 分布式训练

```bash
# FSDP (推荐)
torchrun --nproc_per_node=4 train.py --use-fsdp

# DDP
torchrun --nproc_per_node=4 train.py
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
| 监督微调 | SFTTrainer | 指令数据微调，loss masking |
| 偏好对齐 | DPOTrainer | 标准 DPO loss |
| 知识蒸馏 | DistillationTrainer | Logit-based 蒸馏 |
| 分布式 | FSDP + DDP | BF16 混合精度 + Activation Checkpointing |

## 📊 工程路线图

```
第一阶段 ✅  (当前)
  ├── 现代架构重写 (RoPE + GQA + RMSNorm + SwiGLU)
  ├── 统一训练框架 (TrainerBase / SFT / DPO / Distill)
  ├── FSDP 分布式支持
  └── 分词器升级 (SentencePiece)

第二阶段 🚧
  ├── vLLM 推理引擎集成
  ├── 完整数据管线 (采集→清洗→合成)
  ├── 自动化评测 (MMLU, C-Eval, HumanEval)
  └── Docker 容器化

第三阶段 📋
  ├── PPO / GRPO 强化学习
  ├── Function Calling / Tool Use
  ├── 长上下文 128K+
  └── 多模态扩展
```

## 🔧 技术栈

| 领域 | 技术选型 |
|------|----------|
| 深度学习 | PyTorch 2.0+ |
| 分布式 | FSDP / DDP |
| 推理 | vLLM / llama.cpp / HuggingFace |
| 服务 | FastAPI + SSE |
| 前端 | 原生 HTML/CSS/JS |
| 分词 | SentencePiece / BPE |

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
