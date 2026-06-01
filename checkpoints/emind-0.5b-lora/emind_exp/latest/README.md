---
license: mit
language:
- zh
- en
pipeline_tag: text-generation
tags:
- emind
- moe
- lora
---

# Emind 0.5B

亦梓·智脑 0.5B 参数语言模型，基于 Emind 框架训练。

## 模型信息

| 属性 | 值 |
|------|-----|
| 参数量 | ~500M |
| 架构 | RoPE + GQA + RMSNorm + SwiGLU |
| 训练方式 | LoRA (rank 16) |
| 训练数据 | 代码 + 推理数据蒸馏 (DeepSeek V4 Flash Teacher) |
| 上下文长度 | 2048 tokens |

## 使用方式

此模型使用 **Emind 框架** 训练，需配合框架代码加载（非标准 HuggingFace Transformers 格式）。

### 安装

```bash
# 克隆 Emind 框架
git clone https://github.com/your-org/emind.git
cd emind
pip install -e .
```

### 加载模型

```python
import torch
from model import EmindConfig, create_model
from tokenizer import EmindTokenizer
from modelscope.hub.snapshot_download import snapshot_download

# 下载模型
local_dir = snapshot_download('fuzhen/emind-0.5b')
ckpt = torch.load(f'{local_dir}/model.pt', map_location='cpu')

# 创建模型
cfg = EmindConfig.from_dict(ckpt['model_config'])
model = create_model(cfg)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.eval()

# 加载分词器
tokenizer = EmindTokenizer(vocab_size=cfg.vocab_size)

# 推理
prompt = "用Python写一个快速排序"
ids = torch.tensor([tokenizer.encode(prompt)], device='cuda')
out = model.generate(ids, max_new_tokens=512, temperature=0.8)
print(tokenizer.decode(out[0].tolist()))
```

### CLI 推理

```bash
python cli.py infer \
  --model checkpoints/emind-0.5b-lora/emind_exp/latest/model.pt \
  --d-model 1280 --n-heads 20 --n-kv-heads 4 --n-layers 20 \
  --interactive
```

## 训练细节

- 基座架构: `d_model=1280, n_layers=20, n_heads=20, n_kv_heads=4`
- 优化器: AdamW, lr=2e-5
- 精度: BF16
- 硬件: 单卡 32GB GPU
- 耗时: ~5 分钟

## 许可

MIT License
