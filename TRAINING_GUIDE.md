# Emind 训练指南

本文档详细介绍 Emind 模型的训练流程、模型配置和参数修改方法。

---

## 一、训练阶段与对应文件

Emind 模型训练分为 7 个阶段，每个阶段对应不同的训练脚本：

| 阶段 | 文件名 | 功能 | 输出模型 |
|------|--------|------|----------|
| 1. 预训练 | `01_pretrain.py` | 大规模无标注语料预训练 | `checkpoints/pretrained_model.pt` |
| 2. 微调 | `02_finetune.py` | 监督微调 (SFT) | `checkpoints/finetuned_model.pt` |
| 3. 蒸馏 | `03_distillation.py` | 知识蒸馏 | `checkpoints/distilled_model.pt` |
| 4. LoRA | `04_lora.py` | 低秩适配微调 | `checkpoints/lora_model.pt` |
| 5. 推理优化 | `05_inference.py` | 量化/剪枝优化 | `checkpoints/inference_optimized.pt` |
| 6. 持续学习 | `06_post_learning.py` | 增量学习防遗忘 | `checkpoints/post_learning_model.pt` |
| 7. 反馈训练 | `07_feedback_training.py` | RLHF/DPO训练 | `checkpoints/feedback_trained_model.pt` |

### 推荐训练流程

```
预训练 → 微调 → (蒸馏/LoRA) → 推理优化 → 持续学习 → RLHF/DPO
```

---

## 二、快速开始

### 完整训练示例

```bash
# 1. 预训练 (在大规模语料上)
python3 01_pretrain.py

# 2. 微调 (在指令数据上)
python3 02_finetune.py

# 3. 知识蒸馏 (可选，提升小模型效果)
python3 03_distillation.py

# 4. LoRA 微调 (可选，低资源微调)
python3 04_lora.py

# 5. 推理优化
python3 05_inference.py
```

---

## 三、模型大小配置

Emind 支持 5 种预设模型规模：

| 规模 | 参数量 | vocab_size | d_model | n_layers | n_heads | d_ff | 显存需求 |
|------|--------|------------|---------|----------|---------|------|----------|
| **small** | ~10M | 5000 | 256 | 6 | 8 | 512 | ~2GB |
| **tiny** | ~100M | 10000 | 512 | 12 | 8 | 2048 | ~4GB |
| **1b** | ~1B | 25000 | 2048 | 16 | 16 | 4096 | ~8GB |
| **3b** | ~3B | 30000 | 2560 | 24 | 20 | 6800 | ~16GB |
| **7b** | ~7B | 32000 | 4096 | 28 | 32 | 11008 | ~32GB |

---

## 四、更换模型大小

### 方法一：修改 config.py（推荐）

编辑 `config.py` 文件中的默认配置：

```python
# ============================================================
# 默认配置 - 修改这里的值即可更改模型大小
# ============================================================

# 选择预设配置
CONFIG = CONFIG_SMALL    # 小型模型 (~10M)
# CONFIG = CONFIG_TINY   # 微型模型 (~100M)
# CONFIG = CONFIG_1B     # 1B 模型
# CONFIG = CONFIG_3B     # 3B 模型
# CONFIG = CONFIG_7B     # 7B 模型
```

### 方法二：命令行参数

```bash
# 使用 1B 模型预训练
python3 01_pretrain.py --d_model 2048 --n_layers 16 --n_heads 16

# 使用 7B 模型微调
python3 02_finetune.py --d_model 4096 --n_layers 28 --n_heads 32
```

### 方法三：代码中指定

```python
from config import get_model_config

# 获取特定配置
config = get_model_config('1b')   # 1B 模型
config = get_model_config('7b')    # 7B 模型
config = get_model_config('small') # 小型模型

# 使用配置创建模型
from model import create_model
model = create_model(config)
```

---

## 五、自定义模型配置

如果需要自定义模型参数，可以直接修改配置：

```python
# 自定义配置示例
MY_CONFIG = {
    "name": "custom",
    "vocab_size": 20000,   # 词汇表大小
    "d_model": 1024,       # 隐藏层维度
    "n_heads": 16,         # 注意力头数
    "n_layers": 12,        # Transformer层数
    "d_ff": 4096,          # 前馈网络维度
    "max_seq_len": 1024,   # 最大序列长度
    "dropout": 0.1          # Dropout比率
}
```

### 参数量计算公式

```
总参数量 = vocab_size × d_model + n_layers × (4 × d_model² + 2 × d_model × d_ff)
```

---

## 六、训练参数配置

在 `config.py` 中修改训练参数：

```python
TRAIN_CONFIG = {
    "epochs": 50,                    # 训练轮数
    "batch_size": 32,                # 批次大小 (根据显存调整)
    "learning_rate": 2e-4,           # 学习率
    "min_lr_ratio": 0.1,             # 最小学习率比例
    "grad_clip": 1.0,               # 梯度裁剪
    "gradient_accumulation_steps": 4, # 梯度累积步数
    "warmup_steps": 500,             # 学习率预热步数
    "weight_decay": 0.1,             # 权重衰减
    "train_data_path": "data/yu.jsonl",  # 训练数据路径
    "model_save_path": "checkpoints/model.pt",  # 模型保存路径
    "eval_interval": 100,            # 评估间隔
    "save_interval": 500,            # 保存间隔
    "seed": 42,                     # 随机种子
    "device": "cuda"                 # 设备 (cuda/cpu)
}
```

---

## 七、各阶段训练参数

### 1. 预训练 (01_pretrain.py)

```bash
python3 01_pretrain.py \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --use_amp true
```

### 2. 微调 (02_finetune.py)

```bash
python3 02_finetune.py \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --freeze_layers 0
```

### 3. 知识蒸馏 (03_distillation.py)

```bash
python3 03_distillation.py \
    --epochs 20 \
    --batch_size 32 \
    --temperature 2.0 \
    --alpha 0.5
```

### 4. LoRA 微调 (04_lora.py)

```bash
python3 04_lora.py \
    --rank 8 \
    --alpha 16.0 \
    --dropout 0.1
```

---

## 八、分布式训练

### 单卡训练

```bash
python3 01_pretrain.py --epochs 10 --batch_size 32
```

### 多卡训练

```bash
# 2 卡训练
torchrun --nproc_per_node=2 01_pretrain.py --batch_size 16

# 4 卡训练
torchrun --nproc_per_node=4 01_pretrain.py --batch_size 8

# 8 卡训练
torchrun --nproc_per_node=8 01_pretrain.py --batch_size 4
```

---

## 九、显存优化

如果显存不足，可以：

1. **减小 batch_size**
2. **启用梯度累积**：`gradient_accumulation_steps: 4`
3. **启用混合精度**：`use_amp: true`
4. **减少 max_seq_len**
5. **使用 LoRA**：大幅减少显存需求

---

## 十、推理服务启动

训练完成后，启动 Web 服务：

```bash
python3 web_server.py
```

访问 http://localhost:3333 使用对话界面。

---

## 总结

| 需求 | 操作 |
|------|------|
| 更换模型大小 | 修改 `config.py` 中的 `CONFIG` 变量 |
| 调整训练参数 | 修改 `config.py` 中的 `TRAIN_CONFIG` |
| 自定义模型 | 在代码中使用 `get_model_config()` 或直接定义配置 |
| 查看参数估算 | 运行 `python3 config.py` |
