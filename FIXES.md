# 代码修复与工程化改造记录

## 修复日期
2026-05-16

## 第一阶段改造完成 (工程化重构)

### 1. model.py — 核心架构重构
- 移除旧的 `PositionalEncoding`，替换为 **RoPE** (Rotary Position Embedding)
- 移除标准 `MultiHeadAttention`，替换为 **GQA** (Grouped Query Attention, n_kv_heads=4~8)
- 移除 `LayerNorm`，替换为 **RMSNorm**
- 移除 GELU `FeedForward`，替换为 **SwiGLU** (gate + up + down 投影)
- 添加 **KV Cache** 支持，生成复杂度 O(L²) → O(L)
- 生成策略完善：top-k + top-p (nucleus sampling) + repetition_penalty + temperature
- `EmindConfig` 从 dict 改为 `@dataclass`，支持 `from_dict/to_dict`
- `create_model()` 同时支持 dict 和 EmindConfig 参数

### 2. tokenizer.py — 分词器升级
- 引入 `EmindTokenizer` 统一接口
- SentencePiece 支持 (当可用时自动使用)
- 内置 `_FallbackTokenizer` (无 SentencePiece 时的字符级回退)
- 保留 `SimpleTokenizer`/`BPETokenizer` 别名保证向后兼容
- 默认 vocab_size = 32000，special tokens: pad/unk/bos/eos

### 3. training/ — 统一训练框架
- `TrainingConfig` — dataclass 配置，支持 BF16/FSDP/梯度累积等
- `CheckpointManager` — 断点续训 + best/latest 双保存 + 版本数量限制
- `MetricsTracker` — 自动记录 loss/lr/perplexity，JSON 持久化
- `TrainerBase` — 统一基类，支持混合精度/FSDP/早停/学习率调度
- `SFTTrainer` — 监督微调，支持 assistant-only loss masking
- `DPOTrainer` — 标准 DPO loss (β * log(sigmoid(...)))
- `DistillationTrainer` — logit-based 蒸馏 (温度缩放 + α 平衡)
- 训练脚本 01-07 全部迁移为 training/ 包薄封装 (~3KB each, -80%)

### 4. distributed_utils.py — 分布式升级
- DDP + FSDP 双支持
- FSDP: FULL_SHARD + transformer_auto_wrap_policy + BF16 mixed precision
- 保留现有训练脚本兼容性

### 5. Emind/ 包 — 统一导出
- `Emind/__init__.py` 重新导出根目录新代码
- 版本号 v2.0.0

### 6. vLLM 推理引擎集成
- `unified_inference.py`: VLLMBackend (PagedAttention + Continuous Batching)
- 自动优先级探测：vLLM → 亦API → Ollama → llama.cpp → HuggingFace
- 统一接口：BackendConfig + UnifiedInferenceEngine

### 7. OpenAI 兼容 API
- `web_server.py`: `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- 标准 streaming/non-streaming SSE 格式
- OAuth2 登录 + Session 管理

### 8. WebUI 移动端深度适配
- 3 断点响应式 (1024/768/480px)
- 触摸活跃反馈 (scale:0.97)
- iOS momentum scroll + overscroll-behavior: contain
- 键盘自动适配 + safe-area-inset-bottom 全覆盖
- sidebar 滑动关闭手势 + slide-in overlay
- viewport-fit=cover + 深色模式 theme-color

### 9. 数据管线 (data_pipeline/)
- `DataCollector` — 多源采集 (TXT/JSON/JSONL/CSV/目录)
- `DataCleaner` — 去重/质量过滤/PII脱敏/语言检测
- `DataSynthesizer` — Self-Instruct/Evol-Instruct/模板生成/DPO pairs
- `DataFormatter` — SFT/DPO/Pretrain/Alpaca/ShareGPT 格式
- `DatasetManager` — 数据集版本管理 + stats

### 10. Docker 容器化
- 多阶段构建 Dockerfile (python:3.12-slim)
- docker-compose: api + vllm (GPU profile) + jupyter (dev profile)
- 健康检查 + 环境变量配置

### 11. 自动化评测套件 (eval/)
- `EvaluatorBase` — 评测基类
- `MMLUEvaluator` — 多任务语言理解
- `CEvalEvaluator` — 中文综合评测
- `HumanEvalEvaluator` — 代码生成评测
- `EvaluationRunner` — 统一运行器 + leaderboard

### 12. CLI 扩展
- `pipeline` 子命令：采集→清洗→格式化全流程
- `eval` 子命令：运行评测基准

## 旧修复记录 (2026-03-05)

| 文件 | 修复类型 | 严重程度 |
|------|----------|----------|
| web_server.py | 路径错误 + 类型问题 | 高 |
| 05_inference.py | 函数定义顺序 | 中 |
| 04_lora.py | 缺失导入 | 中 |
| 01_pretrain.py | 梯度计算错误 | 高 |
| model.py (旧) | 逻辑错误 | 中 |
| tokenizer.py (旧) | 类型兼容问题 | 中 |

## 待办

- [x] 训练脚本 (01-07) 迁移到 training/ 包的 TrainerBase
- [x] 集成 vLLM 推理引擎
- [x] 完整数据管线 (采集→清洗→合成→格式化→版本管理)
- [x] Docker 容器化 + docker-compose 一键部署
- [x] 评测套件基础接入 (MMLU, C-Eval, HumanEval)
- [ ] CI/CD + 模型注册表
- [ ] 长上下文扩展 (128K+) + NTK-aware RoPE
- [ ] Function Calling 工具调用支持
