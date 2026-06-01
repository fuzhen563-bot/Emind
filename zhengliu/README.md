# 蒸馏 (zhengliu)

自动生成 SFT 训练数据，用 Teacher 模型蒸馏 → JSONL。

## 快速开始

```bash
# DeepSeek（默认）
python -m zhengliu.distill --api-key sk-xxx --code 50 --reasoning 30

# Ollama 本地
python -m zhengliu.distill --teacher ollama --model qwen2.5:7b --all 10

# OpenAI 兼容
python -m zhengliu.distill --teacher openai --api-key sk-xxx --base-url https://api.yiziyun.com --model gpt-4o --code 20
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--teacher` | deepseek | 后端: deepseek, openai, ollama, vllm, huggingface, local |
| `--api-key` | env | DEEPSEEK_API_KEY 或 OPENAI_API_KEY |
| `--model` | — | 模型名或路径 |
| `--code N` | 0 | 代码数据条数 |
| `--reasoning N` | 0 | 推理数据条数 |
| `--deep-reasoning N` | 0 | 深度推理条数 |
| `--anti-hallucination N` | 0 | 反幻觉条数 |
| `--identity N` | 0 | 身份认知条数 |
| `--all N` | 0 | 每种类型各 N 条 |
| `--max-tokens` | 2048 | Teacher 最大生成长度 |
| `--temperature` | 0.7 | 采样温度 |
| `--workers` | 5 | 并行线程数 |
| `--no-cot` | — | 禁用 CoT 策略 |

## 输出

每行一条 JSON:

```json
{"prompt": "...", "response": "...", "strategy": "direct", "type": "code", "source": "zhengliu"}
```

## Python API

```python
from zhengliu import DistillConfig, DistillEngine

cfg = DistillConfig(teacher="ollama", model="qwen2.5:7b", type_counts={"code": 10})
engine = DistillEngine(cfg)
data = engine.run()  # list[dict]
```
