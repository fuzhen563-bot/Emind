#!/usr/bin/env python3
"""
Emind CLI — 统一命令行入口

用法:
    python cli.py train --mode sft --data data/sft.json
    python cli.py infer --model checkpoints/latest/model.pt --prompt "你好"
    python cli.py serve
    python cli.py eval --benchmarks mmlu,ceval --model checkpoints/best/model.pt
    python cli.py pipeline --collect data/raw --process --format sft
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_train(args):
    from model import EmindConfig, create_model
    from tokenizer import EmindTokenizer
    from training import SFTTrainer, DPOTrainer, DistillationTrainer, TrainingConfig, SFTDataset, DPODataset, DistillationDataset, apply_lora

    tokenizer = EmindTokenizer(vocab_size=args.vocab_size)

    cfg = EmindConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, n_layers=args.n_layers, d_ff=args.d_model * 4,
        max_seq_len=args.max_seq_len, dropout=0.0,
    )

    import json
    raw_data = []
    if os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            raw_data = json.load(f) if args.data.endswith(".json") else [line.strip() for line in f if line.strip()]
    if not raw_data:
        raw_data = [{"prompt": "你好", "response": "你好！"}] * 20

    train_cfg = TrainingConfig(
        mode=args.mode, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, output_dir=args.output_dir,
        max_seq_len=args.max_seq_len, use_bf16=True,
        use_fsdp=args.use_fsdp,
    )

    if args.mode in ("sft", "pretrain"):
        dataset = SFTDataset(raw_data, tokenizer, max_seq_len=args.max_seq_len)
        model = create_model(cfg)
        if args.lora:
            apply_lora(model, rank=args.lora_rank)
        trainer = SFTTrainer(model, train_cfg, dataset)
    elif args.mode == "dpo":
        dataset = DPODataset(raw_data, tokenizer, max_seq_len=args.max_seq_len)
        model = create_model(cfg)
        trainer = DPOTrainer(model, None, train_cfg, dataset, beta=args.beta)
    elif args.mode == "distill":
        dataset = DistillationDataset(raw_data, tokenizer, max_seq_len=args.max_seq_len)
        student = create_model(cfg)
        teacher_cfg = EmindConfig(vocab_size=args.vocab_size, d_model=args.d_model * 2, n_heads=args.n_heads * 2,
                                  n_kv_heads=args.n_kv_heads, n_layers=args.n_layers)
        teacher = create_model(teacher_cfg)
        trainer = DistillationTrainer(student, teacher, train_cfg, dataset, temperature=args.temperature)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    trainer.train()


def cmd_infer(args):
    import torch
    from model import EmindConfig, create_model
    from tokenizer import EmindTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model and os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        cfg = EmindConfig.from_dict(ckpt.get("model_config", {}))
        model = create_model(cfg)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        model.to(device).eval()
        tokenizer = EmindTokenizer(vocab_size=cfg.vocab_size)
    else:
        from unified_inference import UnifiedInferenceEngine, BackendConfig
        engine = UnifiedInferenceEngine(BackendConfig(backend_type=args.backend, api_key=args.api_key))
        result = engine.generate(args.prompt or args.interactive or "")
        print(result)
        return

    if args.prompt:
        ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], device=device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
        print(tokenizer.decode(out[0].tolist()))

    if args.interactive:
        print("Interactive mode (type 'quit' to exit)")
        while True:
            p = input(">>> ")
            if p.lower() in ("quit", "exit", "q"):
                break
            ids = torch.tensor([tokenizer.encode(p, add_bos=True)], device=device)
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
            print(tokenizer.decode(out[0].tolist()))


def cmd_serve(args):
    os.system(f"python web_server.py --port {args.port}")


def cmd_eval(args):
    import torch
    from model import EmindConfig, create_model
    from tokenizer import EmindTokenizer
    from eval import EvaluationRunner

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model and os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        cfg = EmindConfig.from_dict(ckpt.get("model_config", {}))
        model = create_model(cfg)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        model.to(device).eval()
        tokenizer = EmindTokenizer(vocab_size=cfg.vocab_size)

        def model_fn(prompt: str) -> str:
            ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=0.1, top_p=0.9)
            return tokenizer.decode(out[0].tolist())
    else:
        from unified_inference import UnifiedInferenceEngine, BackendConfig
        engine = UnifiedInferenceEngine(BackendConfig(backend_type=args.backend, api_key=args.api_key))

        def model_fn(prompt: str) -> str:
            return engine.generate(prompt)

    benchmarks = args.benchmarks.split(",") if args.benchmarks else None
    runner = EvaluationRunner(model_fn)
    results = runner.run(benchmarks=benchmarks, sample_limit=args.sample_limit)
    runner.print_leaderboard(results)


def cmd_pipeline(args):
    from data_pipeline import DataCollector, DataCleaner, DataFormatter, DatasetManager

    manager = DatasetManager(base_dir=args.data_dir)

    if args.collect:
        collector = DataCollector(data_dir=args.data_dir)
        if os.path.isdir(args.collect):
            sources = collector.from_directory(args.collect)
        else:
            sources = collector.collect(args.collect)
        for source in (sources if isinstance(sources, list) else [sources]):
            if isinstance(source, str):
                manager.register_raw(source)

    if args.process:
        raw_files = manager.list_raw()
        if not raw_files:
            print("No raw data found. Use --collect first.")
            return
        cleaner = DataCleaner()
        formatter = DataFormatter()
        for entry in raw_files:
            def pipeline(items):
                items = cleaner.clean(items, dedup_key="prompt" if isinstance(items and items[0], dict) else None,
                                      target_lang=args.lang, quality_threshold=args.quality_threshold)
                if args.format == "sft":
                    return formatter.to_sft(items)
                elif args.format == "dpo":
                    return formatter.to_dpo(items)
                elif args.format == "pretrain":
                    return formatter.to_pretrain(items)
                return items
            manager.process(entry["path"], [pipeline])

    print(f"Raw:  {len(manager.list_raw())} files")
    print(f"Processed: {len(manager.list_processed())} files")


def main():
    parser = argparse.ArgumentParser(description="Emind CLI - 亦梓·智脑")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p = sub.add_parser("train", help="训练模型")
    p.add_argument("--mode", choices=["sft", "pretrain", "dpo", "distill"], default="sft")
    p.add_argument("--data", default="data/train.txt")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--use-fsdp", action="store_true")

    # infer
    p = sub.add_parser("infer", help="推理")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--backend", default="cloud_api", choices=["cloud_api", "ollama", "llama_cpp", "huggingface"])
    p.add_argument("--api-key")

    # serve
    p = sub.add_parser("serve", help="启动 Web 服务")
    p.add_argument("--port", type=int, default=3333)

    # eval
    p = sub.add_parser("eval", help="模型评测")
    p.add_argument("--model", default=None)
    p.add_argument("--benchmarks", default="mmlu,ceval,humaneval")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--backend", default="cloud_api")
    p.add_argument("--api-key")

    # pipeline
    p = sub.add_parser("pipeline", help="数据处理管线")
    p.add_argument("--collect", default=None, help="采集源文件或目录")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--process", action="store_true", help="处理已采集的数据")
    p.add_argument("--format", choices=["sft", "dpo", "pretrain", "alpaca"], default="sft")
    p.add_argument("--lang", default=None, help="语言过滤 (zh/en)")
    p.add_argument("--quality-threshold", type=float, default=0.1)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)


if __name__ == "__main__":
    main()
