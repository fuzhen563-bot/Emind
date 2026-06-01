"""
Emind 单元测试
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestModel:
    def test_config_defaults(self):
        from model import EmindConfig
        cfg = EmindConfig()
        assert cfg.vocab_size == 32000
        assert cfg.d_model == 4096
        assert cfg.n_heads == 32
        assert cfg.n_kv_heads == 8
        assert cfg.n_layers == 32

    def test_config_custom(self):
        from model import EmindConfig
        cfg = EmindConfig(vocab_size=16000, d_model=512, n_heads=8, n_kv_heads=4, n_layers=6)
        assert cfg.d_ff == 2048  # d_model * 4 default
        d = cfg.to_dict()
        assert d["vocab_size"] == 16000
        cfg2 = EmindConfig.from_dict(d)
        assert cfg2.vocab_size == 16000

    def test_create_model(self):
        import torch
        from model import EmindConfig, create_model
        cfg = EmindConfig(vocab_size=1000, d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=256)
        model = create_model(cfg)
        assert model.config.vocab_size == 1000
        ids = torch.randint(0, 1000, (2, 16))
        loss, logits, caches = model(ids, labels=ids)
        assert loss is not None
        assert loss > 0
        assert logits.shape == (2, 16, 1000)

    def test_generate(self):
        import torch
        from model import EmindConfig, create_model
        cfg = EmindConfig(vocab_size=1000, d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=256)
        model = create_model(cfg)
        ids = torch.tensor([[1, 100, 200]])
        out = model.generate(ids, max_new_tokens=10, temperature=0.8, top_k=20, top_p=0.9)
        assert out.shape[1] > ids.shape[1]  # generated new tokens
        assert out.shape[1] <= ids.shape[1] + 10


class TestTokenizer:
    def test_encode_decode(self):
        from tokenizer import EmindTokenizer
        tok = EmindTokenizer(vocab_size=100)
        text = "hello world"
        ids = tok.encode(text)
        assert isinstance(ids, list)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert isinstance(decoded, str)

    def test_bos_eos(self):
        from tokenizer import EmindTokenizer
        tok = EmindTokenizer(vocab_size=100)
        ids = tok.encode("test", add_bos=True)
        assert ids[0] == tok.bos_token_id
        ids_no_bos = tok.encode("test", add_bos=False)
        assert ids_no_bos[0] != tok.bos_token_id


class TestTrainingConfig:
    def test_device_auto(self):
        from training.config import TrainingConfig
        cfg = TrainingConfig(device="auto")
        assert cfg.device in ("cuda", "cpu")

    def test_eval_batch_size_default(self):
        from training.config import TrainingConfig
        cfg = TrainingConfig(batch_size=8)
        assert cfg.eval_batch_size == 16  # batch_size * 2

    def test_effective_batch_size(self):
        from training.config import TrainingConfig
        cfg = TrainingConfig(batch_size=4, gradient_accumulation_steps=4)
        assert cfg.effective_batch_size == 16


class TestCheckpoint:
    def test_checkpoint_paths(self):
        import tempfile
        from training.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as tmp:
            mgr = CheckpointManager(tmp, save_total_limit=2, experiment_name="test")
            assert mgr._checkpoint_path(100).name == "step_100"
            assert mgr._best_path().name == "best"
            assert mgr._latest_path().name == "latest"


class TestMetrics:
    def test_log_and_summary(self):
        from training.metrics import MetricsTracker
        m = MetricsTracker()
        m.log_step(1.5, 2e-5, step=1)
        m.log_step(1.3, 1.8e-5, step=2)
        m.end_epoch(0, 1.4)
        summary = m.summary()
        assert "epochs" in summary
        assert "1.4" in summary


class TestLoRA:
    def test_apply_lora_params(self):
        from model import EmindConfig, create_model
        from training.lora import apply_lora
        cfg = EmindConfig(vocab_size=1000, d_model=64, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=256)
        model = create_model(cfg)
        before = sum(p.numel() for p in model.parameters())
        apply_lora(model, rank=4)
        after = sum(p.numel() for p in model.parameters())
        assert after > before  # LoRA added params

    def test_merge_lora(self):
        import torch
        from model import EmindConfig, create_model
        from training.lora import apply_lora, merge_lora
        cfg = EmindConfig(vocab_size=1000, d_model=64, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=256)
        model = create_model(cfg)
        weight_before = model.layers[0].attention.W_q.weight.clone()
        apply_lora(model, rank=4)
        merge_lora(model)
        weight_after = model.layers[0].attention.W_q.weight
        assert not torch.allclose(weight_before, weight_after)  # weights changed after merge


class TestDataPipeline:
    def test_collector(self):
        import tempfile
        from data_pipeline import DataCollector
        with tempfile.TemporaryDirectory() as tmp:
            filepath = os.path.join(tmp, "test.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("hello\nworld\n")
            collector = DataCollector(data_dir=tmp)
            lines = collector.from_text(filepath)
            assert len(lines) == 2

    def test_cleaner_dedup(self):
        from data_pipeline import DataCleaner
        c = DataCleaner()
        items = ["a", "b", "a", "c", "b"]
        result = c.deduplicate(items)
        assert len(result) == 3

    def test_cleaner_pii(self):
        from data_pipeline import DataCleaner
        c = DataCleaner()
        text = "手机: 13800138000, 邮箱: test@example.com"
        cleaned = c.strip_pii(text)
        assert "[REDACTED]" in cleaned

    def test_synthesizer_templates(self):
        from data_pipeline import DataSynthesizer
        s = DataSynthesizer(seed=42)
        items = s.generate_from_template("qa", count=5)
        assert len(items) == 5
        assert "prompt" in items[0]
        assert "response" in items[0]

    def test_formatter_sft(self):
        from data_pipeline import DataFormatter
        f = DataFormatter()
        items = [{"prompt": "你好", "response": "你好！"}]
        sft = f.to_sft(items)
        assert len(sft) == 1
        assert "messages" in sft[0]
        assert len(sft[0]["messages"]) == 3

    def test_dataset_manager_stats(self):
        import tempfile
        from data_pipeline import DatasetManager
        import json
        with tempfile.TemporaryDirectory() as tmp:
            mgr = DatasetManager(base_dir=tmp)
            test_file = os.path.join(tmp, "raw", "test.jsonl")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "hello"}) + "\n")
                f.write(json.dumps({"text": "world"}) + "\n")
            stats = mgr.stats(test_file)
            assert stats["samples"] == 2


class TestEval:
    def test_evaluator_base(self):
        from eval import EvaluatorBase
        class MockEval(EvaluatorBase):
            def load(self, path=None): pass
            def evaluate(self, model_fn, **kw): return {"acc": 0.85}
        e = MockEval("test")
        e.results = {"acc": 0.85}
        s = e.summary()
        assert "test" in s
        assert "85.00%" in s


class TestRL:
    def test_ppo_config(self):
        from training.rl import PPOConfig
        cfg = PPOConfig()
        assert cfg.mode == "ppo"
        assert cfg.kl_coef == 0.1
        assert cfg.clip_epsilon == 0.2

    def test_grpo_config(self):
        from training.rl import GRPOConfig
        cfg = GRPOConfig()
        assert cfg.mode == "grpo"
        assert cfg.group_size == 8

    def test_ppo_dataset(self):
        from training.rl import PPODataset
        data = [{"prompt": "hi", "responses": ["hello"], "rewards": [1.0]}]
        tokenizer_mock = type("Tokenizer", (), {"encode": lambda s, **kw: [0, 1, 2]})()
        ds = PPODataset(data, tokenizer_mock, max_seq_len=16)
        item = ds[0]
        assert item["prompt"].shape == (16,)
        assert len(item["responses"]) == 1
        assert item["responses"][0].shape == (16,)

    def test_kl_estimators(self):
        import torch
        from training.rl import kl_estimate
        logps = torch.tensor([[-1.0, -2.0, -3.0]])
        ref_logps = torch.tensor([[-1.5, -2.5, -3.5]])
        for m in ("kl1", "kl2", "kl3"):
            kl = kl_estimate(logps, ref_logps, method=m)
            assert kl.shape == (1,)

    def test_compute_token_logps(self):
        import torch
        from training.rl import compute_token_logps
        logits = torch.randn(2, 8, 100)
        ids = torch.randint(0, 100, (2, 8))
        logps = compute_token_logps(logits, ids)
        assert logps.shape == (2, 7)

    def test_reward_model_forward(self):
        import torch
        from model import EmindConfig, create_model
        from training.rl import RewardModel
        cfg = EmindConfig(vocab_size=1000, d_model=64, n_heads=4, n_kv_heads=2, n_layers=1, d_ff=256)
        base = create_model(cfg)
        rm = RewardModel(base, hidden_dim=64)
        ids = torch.randint(0, 1000, (2, 8))
        r = rm(ids)
        assert r.shape == (2,)

    def test_ppo_dataset_empty_responses(self):
        from training.rl import PPODataset
        data = [{"prompt": "hi", "response": "hello", "reward": 1.0}]
        tokenizer_mock = type("Tokenizer", (), {"encode": lambda s, **kw: [0, 1, 2]})()
        ds = PPODataset(data, tokenizer_mock, max_seq_len=16)
        item = ds[0]
        assert len(item["responses"]) == 1
        assert item["rewards"].shape == (1,)

    def test_ppo_dataset_full_fields(self):
        from training.rl import PPODataset
        data = [{"prompt": "hi", "responses": ["r1", "r2", "r3"], "rewards": [0.5, 1.0, 0.0]}]
        tokenizer_mock = type("Tokenizer", (), {"encode": lambda s, **kw: [0, 1, 2]})()
        ds = PPODataset(data, tokenizer_mock, max_seq_len=16)
        item = ds[0]
        assert len(item["responses"]) == 3
        assert "response_texts" in item


class TestDistillation:
    def test_config_defaults(self):
        from training.distillation_pipeline import DistillationConfig
        cfg = DistillationConfig()
        assert cfg.num_code_samples == 200
        assert cfg.num_deep_reasoning_samples == 200
        assert cfg.num_anti_hallucination_samples == 200
        assert cfg.num_identity_samples == 50
        assert cfg.identity_name == "Emind·智脑"
        assert cfg.identity_developer == "亦梓科技"
        assert "direct" in cfg.strategies

    def test_fill_template(self):
        from training.distillation_pipeline import DistillationPipeline, DistillationConfig
        pipe = DistillationPipeline(DistillationConfig(teacher_backend="huggingface"))
        filled = pipe._fill("用 {lang} 实现一个 {ds}")
        assert "{lang}" not in filled
        assert "{ds}" not in filled
        assert len(filled) > 5

    def test_seed_prompts(self):
        from training.distillation_pipeline import DistillationPipeline, DistillationConfig, CODE_SEEDS, REASONING_SEEDS
        pipe = DistillationPipeline(DistillationConfig(teacher_backend="huggingface"))
        code = pipe._seed_prompts(CODE_SEEDS, 5)
        assert len(code) == 5 and all(isinstance(p, str) for p in code)
        reas = pipe._seed_prompts(REASONING_SEEDS, 3)
        assert len(reas) == 3

    def test_synthesize_refusal(self):
        from training.distillation_pipeline import DistillationPipeline, DistillationConfig
        pipe = DistillationPipeline(DistillationConfig(teacher_backend="huggingface"))
        samples = pipe._synthesize_refusal_samples(5)
        assert len(samples) == 5
        for s in samples:
            assert s["type"] == "anti_hallucination_synthetic"
            assert len(s["response"]) > 10

    def test_synthesize_identity(self):
        from training.distillation_pipeline import DistillationPipeline, DistillationConfig
        cfg = DistillationConfig(teacher_backend="huggingface", identity_name="Emind·智脑", identity_developer="亦梓科技")
        pipe = DistillationPipeline(cfg)
        samples = pipe._synthesize_identity_samples(5)
        assert len(samples) == 5
        for s in samples:
            assert s["type"] == "identity"
            assert "亦梓科技" in s["response"]
            assert "Emind" in s["response"] or "智脑" in s["response"]

    def test_build_strategy_includes_identity(self):
        from training.distillation_pipeline import DistillationPipeline, DistillationConfig
        cfg = DistillationConfig(teacher_backend="huggingface", identity_name="Emind", identity_developer="亦梓")
        pipe = DistillationPipeline(cfg)
        p = pipe._build_strategy_prompt("hi", "direct")
        assert "Emind" in p
        assert "亦梓" in p

    def test_filter_short(self):
        from training.distillation_pipeline import DistillationPipeline, DistillationConfig
        pipe = DistillationPipeline(DistillationConfig(teacher_backend="huggingface"))
        items = [{"response": "hi"}, {"response": "hello world how are you doing today this is a test filter quality check"}]
        filtered = pipe._filter(items)
        assert len(filtered) == 1

    def test_dedup_key(self):
        from training.distillation_pipeline import DistillationPipeline
        k1 = DistillationPipeline._dedup_key("hello world")
        k2 = DistillationPipeline._dedup_key("Hello  World")
        assert k1 == k2

    def test_strategies(self):
        from training.distillation_pipeline import DistillationPipeline, DistillationConfig
        cfg = DistillationConfig(teacher_backend="huggingface", identity_name="TestAI", identity_developer="TestCorp")
        pipe = DistillationPipeline(cfg)
        for s in ["direct", "cot", "verify", "refuse", "reason_then_answer", "explain_then_code"]:
            p = pipe._build_strategy_prompt("test", s)
            assert "test" in p
            assert "TestAI" in p
        assert "我不知道" in pipe._build_strategy_prompt("q", "refuse")
