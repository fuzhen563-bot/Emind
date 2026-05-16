# Emind 工程化改造路线图

> 目标：将 Emind 从研究原型改造为亦梓科技主力生产级 AI 模型  
> 作者：亦梓科技 AI 部  
> 版本：v1.0

---

## 一、模型架构层 (P0 - 核心重构)

### 1.1 替换位置编码
- [ ] 移除当前正弦位置编码 (`model.py:PositionalEncoding`)
- [ ] 实现 **RoPE** (Rotary Position Embedding)
- [ ] 验证长度外推能力（训练 2K, 推理 8K+）

### 1.2 注意力机制升级
- [ ] 从标准 MHA 改为 **GQA** (Grouped Query Attention)，n_kv_heads=8 (7B)
- [ ] 添加 **KV Cache** 到 `generate()`，消除重复计算（复杂度 O(L²) → O(L)）
- [ ] 集成 **Flash Attention v2**（可选，依赖 CUDA）

### 1.3 FFN 升级
- [ ] 从 GELU + Linear → **SwiGLU** 结构
- [ ] 参数量不变，但需要增加一个 gate 投影 `W_g`

### 1.4 LayerNorm 替换
- [ ] `LayerNorm` → **RMSNorm**（训练更稳定，减少计算量）

### 1.5 生成策略完善
- [ ] 当前 `generate()` 只支持 top-k，添加 top-p (nucleus sampling)
- [ ] 添加 repetition penalty 参数
- [ ] 添加 beam search 支持（可选）

### 1.6 量化支持
- [ ] FP16/BF16 推理
- [ ] INT8/INT4 量化 (GPTQ / AWQ)

---

## 二、分词器 (P0 - 必须替换)

### 2.1 替换字符级分词器
- [ ] 集成 **SentencePiece** (中文场景首选)
- [ ] 或 **HuggingFace Tokenizers** (BPE，与 LLaMA 生态兼容)
- [ ] vocab_size = 32000~64000
- [ ] 支持特殊 token 管理 (pad/unk/bos/eos + function call tokens)

### 2.2 兼容性
- [ ] 统一根目录 `tokenizer.py` 和 `Emind/tokenizer.py` 两套实现
- [ ] 删除 `Emind/` 包中未修复 bug 的旧版本

---

## 三、训练基础设施 (P0 - 必须重写)

### 3.1 统一训练框架
- [ ] 创建 `emind-train/core/` 共享基类：
  - [ ] `TrainerBase` — 消除 7 个脚本 (`01_*.py` ~ `07_*.py`) 的重复代码
  - [ ] `TrainingConfig` — pydantic 配置校验 (替代纯 dict)
  - [ ] `CheckpointManager` — 断点续训 + 版本管理 + best/latest 双保存
  - [ ] `MetricsTracker` — loss/lr/perplexity 自动记录

### 3.2 分布式升级
- [ ] 当前仅 DDP → 升级到 **FSDP** (Fully Sharded Data Parallel)
  - [ ] 参数/梯度/优化器状态全分片，7B 训练不再 OOM
  - [ ] 支持 hybrid shard (跨节点用 TP + 节点内 FSDP)
- [ ] 可选：集成 **DeepSpeed ZeRO-3**
- [ ] 添加 **Activation Checkpointing** (梯度检查点)

### 3.3 混合精度
- [ ] 从 torch AMP (FP16) → **BF16** 主训练精度
  - [ ] 移除 GradScaler（BF16 不需要）
  - [ ] 减少 loss scaling 带来的精度损失

### 3.4 训练策略升级
- [ ] **SFT**: 增加 loss masking（只对 assistant 回复计算 loss）— `trainer_multiround.py` 有雏形但未正确使用
- [ ] **蒸馏**: 完整 logit-based + feature-based 蒸馏（当前仅 logit）
- [ ] **DPO**: 替换当前简化版，实现标准 DPO loss (`beta * log(sigmoid(...))`)
- [ ] **PPO**: 新增完整 PPO 算法（当前完全缺失）
- [ ] **长上下文**: 添加 YaRN / NTK-aware 位置编码扩展

---

## 四、推理服务层 (P0 - 必须重写)

### 4.1 替换自研推理引擎
- [ ] 当前 `model.py:generate` 无 KV Cache → **对接成熟推理框架**

**推荐方案**（选一）：
- [ ] **vLLM** — 吞吐最高，支持 PagedAttention + 持续批处理
- [ ] **TensorRT-LLM** — NVIDIA 最佳优化，延迟最低
- [ ] **SGLang** — 最新方案，RadixAttention 前缀缓存

**备选方案**（如需要深度定制）：
- [ ] 自研 Paged KV Cache + 持续批处理调度器

### 4.2 API 标准化
- [ ] 替换当前 Flask SSE 接口 → **OpenAI 兼容 API**
  - [ ] `POST /v1/chat/completions` (SSE 流式)
  - [ ] `POST /v1/completions`
  - [ ] `GET /v1/models`
- [ ] 支持 stream_options: `{include_usage: true}`
- [ ] 支持 response_format: `{type: "json_object"}`

### 4.3 服务部署
- [ ] Docker 容器化（NVIDIA Container Toolkit）
- [ ] Kubernetes 部署配置（自动扩缩容）
- [ ] 多模型路由、灰度发布、回滚策略

---

## 五、数据工程管线 (P0 - 必须搭建)

### 5.1 数据处理平台
- [ ] 搭建数据采集系统（定向爬虫 / API 接入）
- [ ] **数据清洗管线**：
  - [ ] MinHash/LSH 去重
  - [ ] 质量过滤（困惑度、长度、语言检测）
  - [ ] PII 脱敏（身份证、手机号、银行卡等）
  - [ ] 毒性检测 + 安全过滤
- [ ] **数据合成**：
  - [ ] Self-Instruct 生成指令数据
  - [ ] Evol-Instruct 渐进式难度提升
  - [ ] Rejection Sampling 拒绝采样（用奖励模型筛选）

### 5.2 数据版本管理
- [ ] 每条数据带版本标签
- [ ] 支持数据集回滚
- [ ] 数据溯源 (provenance tracking)

### 5.3 数据飞轮
- [ ] 用户反馈采集 → 自动标注 → 清洗 → 再训练闭环
- [ ] 主动学习：自动挑选低置信度样本请求人工标注

---

## 六、质量保障体系 (P1 - 必须建立)

### 6.1 自动化评测
- [ ] **通用能力**：MMLU, C-Eval, CMMLU, GSM8K, HumanEval
- [ ] **中文专项**：CLUE, FewCLUE, CHID, AFQMC
- [ ] **对话能力**：MT-Bench, AlpacaEval (LLM-as-Judge)
- [ ] **安全**：Safety-Prompts, 红队测试, Jailbreak 检测
- [ ] **公司专属**：亦梓内部 QA 数据集 + 业务场景评测集

### 6.2 CI/CD
- [ ] GitHub Actions / GitLab CI 集成
- [ ] 提交 → 小规模训练 → 快速评测 → 通过 → 上线
- [ ] A/B 测试框架（canary → staging → production 渐进上线）

### 6.3 模型注册表
- [ ] 每版本记录：参数量、评测分数、训练数据哈希
- [ ] 支持快速回滚到任一历史版本

---

## 七、部署与运维 (P1)

### 7.1 监控体系
| 类别 | 指标 | 告警阈值 |
|------|------|---------|
| 延迟 | P50/P95/P99 TTFT, TPOT | P99 TTFT > 2s |
| 吞吐 | tokens/s, requests/s | 低于基线 30% |
| 质量 | 用户反馈评分、拒绝率 | 好评率 < 80% |
| 系统 | GPU 利用率、显存、OOM | 显存 > 95% |
| 业务 | DAU、对话轮次、留存率 | — |

### 7.2 可观测性
- [ ] Prometheus + Grafana 仪表盘
- [ ] 请求追踪 (OpenTelemetry)
- [ ] 日志聚合 (ELK / Loki)
- [ ] 自定义告警规则

### 7.3 弹性部署
- [ ] 自动扩缩容 (HPA based on GPU utilization)
- [ ] 多区域 / 多集群容灾
- [ ] 私有化部署方案（docker-compose / k8s helm chart）

---

## 八、代码质量 (P1)

### 8.1 消除技术债务
- [ ] 合并根目录和 `Emind/` 包：保留一个，删除另一个
- [ ] 修复 `FIXES.md` 中记录但未在 `Emind/` 包中修复的 bug
- [ ] `advanced_trainer.py` 中 `next_item` 变量名错误
- [ ] `web_server.py` 中硬编码的 epoch4888 路径

### 8.2 测试体系
- [ ] 单元测试：model.py, tokenizer.py, trainer.py
- [ ] 集成测试：端到端训练 + 推理管线
- [ ] 回归测试：每次架构改动自动跑 benchmark

### 8.3 工程规范
- [ ] 类型注解全覆盖 (mypy strict mode)
- [ ] ruff 代码风格检查
- [ ] 模块文档 + API 文档自动生成

---

## 九、实施路线图

### 第一阶段：MVP 可用 (1-2个月)

```
优先级: P0
目标: 能在 4×A100 上训练 7B 模型，推理延迟 < 2s

任务:
  □ 替换 tokenizer (SentencePiece)
  □ RoPE + GQA + RMSNorm 改造 model.py
  □ KV Cache 添加到 generate()
  □ 统一 Trainer 基类 + FSDP 支持
  □ 对接 vLLM 推理服务
  □ OpenAI 兼容 API
  □ Docker 容器化
```

### 第二阶段：生产就绪 (2-4个月)

```
优先级: P0 + P1
目标: 能承载公司主力业务流量，支持持续迭代

任务:
  □ 完整数据管线 (采集→清洗→合成→标注→版本管理)
  □ 完整 DPO 训练管线
  □ 多维度自动评测体系
  □ SwiGLU FFN
  □ 监控 + 告警系统
  □ A/B 测试框架
  □ 质量数据飞轮
```

### 第三阶段：持续领先 (4-8个月)

```
优先级: P1
目标: 成为行业领先的生产级模型，支持多模态扩展

任务:
  □ 完整 PPO/GRPO 强化学习
  □ Function Calling / Tool Use
  □ Long Context 128K+
  □ 多模态 (图像/语音)
  □ 私有化部署方案
  └ 公司专属 benchmark 驱动迭代
```

---

## 十、资源需求估算

| 类别 | 第一阶段 | 第二阶段 | 第三阶段 |
|------|---------|---------|---------|
| GPU | 4×A100 80G | 8×A100 80G | 64×A100/H100 |
| 研发人力 | 2-3 人 | 4-6 人 | 8-10 人 |
| 标注人力 | — | 5-10 人 | 10-20 人 |
| 存储(TB) | 1-5 | 10-50 | 50-200 |
| 月训练时长 | 200h | 1000h | 5000h+ |

---

## 附录：关键决策点

### A. 推理引擎选型对比

| 选项 | 吞吐 | 延迟 | 定制灵活性 | 社区成熟度 |
|------|------|------|-----------|-----------|
| vLLM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| TensorRT-LLM | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| SGLang | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 自研 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |

**推荐**：第一阶段先用 vLLM 快速上线，后期按需自研定制引擎。

### B. 分布式策略选型

| 策略 | 显存节省 | 通信开销 | 适用规模 |
|------|---------|---------|---------|
| DDP (当前) | 0% | 低 | ≤1B |
| FSDP FULL_SHARD | ~75% | 中 | ≤7B |
| FSDP HYBRID_SHARD | ~50% | 低 | ≤13B |
| DeepSpeed ZeRO-3 | ~75% | 中 | ≤70B |
| 3D Parallelism (TP+PP+DP) | ~80% | 高 | ≥70B |

**推荐**：7B 参数以内用 FSDP FULL_SHARD；更大规模用 3D Parallelism。
