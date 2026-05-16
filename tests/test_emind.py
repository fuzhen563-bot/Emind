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
