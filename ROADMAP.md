# Emind AI 路线图

> 亦梓·智脑 (Emind AI) — 生产级 AI 框架

---

## 架构层 (model.py)

| 特性 | 状态 | 说明 |
|------|------|------|
| RoPE + NTK-aware 缩放 | ✅ | 支持 128K+ 上下文扩展 |
| Grouped Query Attention (GQA) | ✅ | n_kv_heads=4~8，推理加速 |
| RMSNorm | ✅ | 替代 LayerNorm，训练更稳定 |
| SwiGLU FFN | ✅ | 相比 ReLU 提升质量 |
| KV Cache | ✅ | 推理时缓存 K/V，避免重算 |
| Flash Attention | ✅ | `F.scaled_dot_product_attention`，避免 O(n²) 显存 |
| Gradient Checkpointing | ✅ | 显存换速度，大 batch 训练必备 |
| QK-Norm | ✅ | Q/K 投影后 RMSNorm，大模型训练稳定性保障 |
| Parallel Attn+FFN | ✅ | Attn 和 FFN 共享归一化输入并行计算，减少串行深度 |
| MoE (Sparse Mixture of Experts) | ✅ | 8 Expert × Top-2，含辅助负载均衡 Loss |
| Long Context (>128K, YaRN/LongRoPE) | ✅ | YaRN RoPE 缩放，4K→128K 扩展 |
| FP8 训练 | ❌ | |
| 量化推理 (FP8/INT4) | ✅ | `quantization.py`: INT4 weight-only (per-group) + FP8 (H100+) |

## 分词器 (tokenizer.py)

| 特性 | 状态 | 说明 |
|------|------|------|
| SentencePiece 训练脚本 | ✅ | `cli.py train-tokenizer` 子命令 |
| SentencePiece BPE 模型 | ✅ | 训练后保存为 `.model` / `.vocab` |
| Fallback 字符级 tokenizer | ✅ | 开发环境备用 |
| 预编码数据集 | ✅ | `SFTDataset.__init__` 提前编码，避免每步重算 |
| Chat template | ❌ | `<|im_start|>` / `<|im_end|>` |
| 特殊 token 注册 | ✅ | tool_call / tool_result |

## 训练框架 (training/)

| 特性 | 状态 | 说明 |
|------|------|------|
| TrainerBase | ✅ | BF16/FSDP/早停/LR warmup/梯度裁剪 |
| SFT | ✅ | 监督微调 |
| DPO | ✅ | 直接偏好优化 |
| Distillation | ✅ | 蒸馏训练 |
| LoRA | ✅ | 函数式注入，无需继承 |
| FSDP | ✅ | 全分片 + MixedPrecision BF16 + DistributedSampler |
| RL (PPO/GRPO/RM) | ✅ | 强化学习全家桶 |

## 数据管线

| 特性 | 状态 | 说明 |
|------|------|------|
| 数据采集 | ✅ | |
| 数据清洗/去重 | ✅ | |
| 格式转换 | ✅ | sft / dpo / alpaca / pretrain |
| 数据蒸馏 | ✅ | 5 类种子 + 6 策略 + 并发 10 线程 |
| 蒸馏产物 | ✅ | 1089 条 SFT 样本 (`distilled_sft.jsonl`) |

## 推理引擎 (unified_inference.py)

| 特性 | 状态 | 说明 |
|------|------|------|
| 统一推理接口 | ✅ | OpenAI 兼容 |
| vLLM | ✅ | 高性能推理 |
| 亦API | ✅ | 云端推理 |
| Ollama | ✅ | 本地回退 |
| llama.cpp | ✅ | 本地回退 |
| HuggingFace | ✅ | 本地回退 |
| Speculative Decoding | ❌ | 参数已暴露，未调优 |

## API / 服务

| 特性 | 状态 |
|------|------|
| OpenAI 兼容 API | ✅ |
| WebUI (思考可视化 / 竞技场 / 上下文记忆) | ✅ |
| Docker 部署 | ✅ |

## CLI (cli.py)

| 命令 | 状态 | 说明 |
|------|------|------|
| train | ✅ | SFT/DPO/Distill/RL 统一入口 |
| infer | ✅ | 交互/单次推理 |
| serve | ✅ | WebUI + vLLM Server |
| eval | ✅ | MMLU/C-Eval/HumanEval |
| pipeline | ✅ | 数据采集→处理→蒸馏 |
| rl | ✅ | PPO/GRPO/RM |
| vllm | ✅ | 诊断 + 自动配置 |

## 待做 (按优先级)

### P0 — 阻塞项
- [x] **训练 SentencePiece 模型** — `cli.py train-tokenizer` 子命令，从 JSONL 语料训练 SP BPE 模型

### P1 — 重要
- [ ] **MoE** — 稀疏混合专家，同参数量下容量翻倍
- [ ] **Long Context 扩展** — YaRN / LongRoPE >128K
- [ ] **加载 HuggingFace 基座模型** — 支持从 HF 加载预训练权重微调

### P2 — 加速/优化
- [x] **量化推理** — FP8 / INT4 部署加速 (`quantization.py`)
- [x] **Speculative Decoding** — vLLM 推测解码参数已暴露并传递
- [ ] **FP8 训练** — H100 上显存减半

### P3 — 扩展
- [ ] **多模态扩展** — 视觉 / 语音输入
- [ ] **模型注册表 + 版本管理**
- [ ] **RLHF 全流程生产化**

## 技术栈

| 维度 | 选型 |
|------|------|
| 框架 | PyTorch 2.8 + FSDP |
| 推理 | vLLM + 自定义引擎 |
| 分词 | SentencePiece (BPE) |
| 训练精度 | BF16 混合精度 |
| 分布式 | FSDP FULL_SHARD |
| API 规范 | OpenAI 兼容 |
| 部署 | Docker + torchserve |
