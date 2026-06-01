#!/usr/bin/env python3
"""
Emind CLI — 统一命令行入口

    用法:
        python cli.py train --mode sft --data data/sft.json
        python cli.py train-tokenizer --data data/distilled/emind_code_4b/distilled_sft.jsonl
        python cli.py infer --model checkpoints/latest/model.pt --prompt "你好"
    python cli.py serve
    python cli.py eval --benchmarks mmlu,ceval --model checkpoints/best/model.pt
    python cli.py pipeline --collect data/raw --process --format sft
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_rl(args):
    from model import EmindConfig, create_model
    from tokenizer import EmindTokenizer
    from training.rl import (
        PPOConfig, GRPOConfig, PPODataset,
        PPOTrainer, GRPOTrainer,
        RewardModel, RewardModelTrainer,
    )

    tokenizer = EmindTokenizer(vocab_size=args.vocab_size)

    import json
    raw_data = []
    if os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            if args.data.endswith(".jsonl"):
                raw_data = [json.loads(line) for line in f if line.strip()]
            else:
                raw_data = json.load(f)
    if not raw_data:
        raw_data = [{"prompt": "你好", "response": "世界", "reward": 1.0}]

    dataset = PPODataset(raw_data, tokenizer, max_seq_len=args.max_seq_len)

    cfg = EmindConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, n_layers=args.n_layers, d_ff=args.d_model * 4,
        max_seq_len=args.max_seq_len, dropout=0.0,
    )

    if args.rl_mode == "ppo":
        from training.lora import apply_lora
        model = create_model(cfg)
        if args.lora:
            apply_lora(model, rank=args.lora_rank)
        train_cfg = PPOConfig(
            kl_coef=args.kl_coef, clip_epsilon=args.clip_epsilon,
            ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch_size,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, output_dir=args.output_dir,
            max_seq_len=args.max_seq_len, use_bf16=True,
        )
        ref_model = create_model(cfg) if args.ref_model else None
        trainer = PPOTrainer(model, train_cfg, dataset, ref_model=ref_model, tokenizer=tokenizer)
    elif args.rl_mode == "grpo":
        model = create_model(cfg)
        if args.lora:
            from training.lora import apply_lora
            apply_lora(model, rank=args.lora_rank)
        train_cfg = GRPOConfig(
            kl_coef=args.kl_coef, group_size=args.group_size,
            ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch_size,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, output_dir=args.output_dir,
            max_seq_len=args.max_seq_len, use_bf16=True,
        )
        ref_model = create_model(cfg) if args.ref_model else None

        def reward_fn(prompts, responses):
            return [1.0] * len(responses)

        trainer = GRPOTrainer(model, train_cfg, dataset, ref_model=ref_model,
                              reward_fn=reward_fn, tokenizer=tokenizer)
    elif args.rl_mode == "rm":
        model = create_model(cfg)
        rm = RewardModel(model, hidden_dim=cfg.d_model)
        from training.config import TrainingConfig as RMConfig
        train_cfg = RMConfig(
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, output_dir=args.output_dir,
            max_seq_len=args.max_seq_len,
        )
        from training.dpo import DPODataset
        dpo_dataset = DPODataset(raw_data, tokenizer, max_seq_len=args.max_seq_len)
        trainer = RewardModelTrainer(rm, train_cfg, dpo_dataset)
    else:
        raise ValueError(f"Unknown rl_mode: {args.rl_mode}")

    trainer.train()


def cmd_train(args):
    import torch
    from model import EmindConfig, create_model
    from tokenizer import EmindTokenizer
    from training import SFTTrainer, DPOTrainer, DistillationTrainer, TrainingConfig, SFTDataset, DPODataset, DistillationDataset, apply_lora
    from training.pretrain import PretrainDataset, PretrainTrainer

    tokenizer = EmindTokenizer(vocab_size=args.vocab_size, model_path=args.tokenizer_path)

    cfg = EmindConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, n_layers=args.n_layers, d_ff=args.d_model * 4,
        max_seq_len=args.max_seq_len, dropout=0.0,
        activation_checkpointing=True,
        qk_norm=True,
        parallel_attn_ffn=True,
        use_moe=args.moe,
        rope_scaling_type="yarn",
        rope_scaling_factor=32.0,
        original_max_len=4096,
    )

    import json
    raw_data = []
    if os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            if args.data.endswith(".jsonl"):
                raw_data = [json.loads(line) for line in f if line.strip()]
            else:
                raw_data = json.load(f)
    if not raw_data:
        raw_data = [{"prompt": "你好", "response": "你好！"}] * 20

    train_cfg = TrainingConfig(
        mode=args.mode, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, output_dir=args.output_dir,
        max_seq_len=args.max_seq_len, use_bf16=True,
        compile_model=args.compile, neftune_noise_alpha=args.neftune, optimizer=args.optimizer,
        curriculum_stages=args.curriculum_stages, replay_ratio=args.replay_ratio,
        use_fsdp=args.use_fsdp,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )

    if args.resume and os.path.exists(args.resume):
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_cfg = resume_ckpt.get("model_config", {})
        if resume_cfg:
            for k, v in resume_cfg.items():
                if not hasattr(cfg, k) or getattr(cfg, k) == EmindConfig.__dataclass_fields__[k].default:
                    setattr(cfg, k, v)
        print(f"[train] Config aligned to checkpoint (d_ff={cfg.d_ff}, vocab_size={cfg.vocab_size})")

    if args.mode == "pretrain":
        dataset = PretrainDataset(raw_data, tokenizer, max_seq_len=args.max_seq_len)
        model = create_model(cfg)
        if args.resume and os.path.exists(args.resume):
            state = resume_ckpt.get("model_state_dict", resume_ckpt)
            model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
            print(f"[Pretrain] Resumed from {args.resume}")
        trainer = PretrainTrainer(model, train_cfg, dataset)
    elif args.mode == "sft":
        if args.curriculum_stages > 1 or args.replay_ratio > 0:
            from training.curriculum import CurriculumDataset
            replay_raw = []
            if args.replay_data and os.path.exists(args.replay_data):
                with open(args.replay_data, encoding="utf-8") as f:
                    replay_raw = [json.loads(line) for line in f if line.strip()]
            dataset = CurriculumDataset(
                raw_data, tokenizer, max_seq_len=args.max_seq_len,
                stages=args.curriculum_stages, replay_data=replay_raw, replay_ratio=args.replay_ratio,
            )
        else:
            dataset = SFTDataset(raw_data, tokenizer, max_seq_len=args.max_seq_len)
        model = create_model(cfg)
        if args.lora:
            apply_lora(model, rank=args.lora_rank)
        if args.resume and os.path.exists(args.resume):
            state = resume_ckpt.get("model_state_dict", resume_ckpt)
            model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
            print(f"[SFT] Resumed from {args.resume}")
        trainer = SFTTrainer(model, train_cfg, dataset)
    elif args.mode == "dpo":
        dataset = DPODataset(raw_data, tokenizer, max_seq_len=args.max_seq_len)
        model = create_model(cfg)
        if args.lora:
            apply_lora(model, rank=args.lora_rank)
        if args.resume and os.path.exists(args.resume):
            state = resume_ckpt.get("model_state_dict", resume_ckpt)
            model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
            print(f"[DPO] Policy model loaded from {args.resume}")
            ref_model = create_model(cfg)
            ref_model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
            print("[DPO] Reference model copied from checkpoint (frozen)")
        else:
            ref_model = create_model(cfg)
            ref_model.load_state_dict(model.state_dict())
            print("[DPO] Reference model copied from policy (same init)")
        trainer = DPOTrainer(model, ref_model, train_cfg, dataset, beta=args.dpo_beta, label_smoothing=args.label_smoothing)
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

    from training.distributed import cleanup as dist_cleanup
    dist_cleanup()


def cmd_train_tokenizer(args):
    import json
    from pathlib import Path
    from tokenizer import EmindTokenizer

    corpus_path = args.corpus or args.data
    corpus_lines = []

    if os.path.exists(corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            if corpus_path.endswith(".jsonl"):
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    corpus_lines.append(item.get("prompt", "") + item.get("response", ""))
            else:
                corpus_lines = [line.strip() for line in f if line.strip()]

    if not corpus_lines:
        corpus_lines = ["你好，世界！Hello, world!"]

    tmp_dir = Path(args.output_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = str(tmp_dir / "corpus.txt")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for line in corpus_lines:
            f.write(line.strip() + "\n")

    print(f"Corpus: {len(corpus_lines)} lines -> {corpus_file}")

    model_prefix = str(tmp_dir / "emind_tokenizer")
    sp_model_path = model_prefix + ".model"

    tokenizer = EmindTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(corpus_file, model_prefix=model_prefix)

    print(f"Tokenizer saved: {sp_model_path}")

    new_tokenizer = EmindTokenizer(vocab_size=args.vocab_size, model_path=sp_model_path)
    print(f"Vocab size: {len(new_tokenizer)}")
    print(f"Sample encode '你好': {new_tokenizer.encode('你好')}")

    if args.data and args.data.endswith(".jsonl") and args.reencode:
        out_path = args.reencode
        with open(args.data, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompt = item.get("prompt", "")
                response = item.get("response", "")
                new_item = {
                    "prompt": prompt,
                    "response": response,
                    "tokens": {
                        "input_ids": new_tokenizer.encode(prompt + response),
                        "prompt_len": len(new_tokenizer.encode(prompt)),
                    }
                }
                fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
        print(f"Re-encoded data saved: {out_path}")


def cmd_infer(args):
    import torch
    from model import EmindConfig, create_model
    from tokenizer import EmindTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model and os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=device, weights_only=False)
        cfg_dict = ckpt.get("model_config", {})
        if not cfg_dict:
            cfg_dict = dict(
                d_model=args.d_model or 768,
                n_heads=args.n_heads or 12,
                n_kv_heads=args.n_kv_heads or 4,
                n_layers=args.n_layers or 12,
                vocab_size=args.vocab_size or 32000,
                max_seq_len=args.max_seq_len or 2048,
            )
        if "d_ff" not in cfg_dict:
            cfg_dict["d_ff"] = cfg_dict.get("d_model", 4096) * 4
        cfg = EmindConfig.from_dict(cfg_dict)
        if args.kv_quant:
            cfg.use_kv_quant = True
        model = create_model(cfg)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        model.to(device).eval()

        if args.quantize:
            from quantization import quantize_model, estimate_model_size
            model = quantize_model(model, mode=args.quantize, group_size=args.quantize_groups)
            print(f"Quantized ({args.quantize}): {estimate_model_size(model, args.quantize)}")

        if not args.tokenizer_path:
            print("WARNING: --tokenizer-path 未指定，将使用字符级 fallback tokenizer（输出可能为乱码）")
        tokenizer = EmindTokenizer(vocab_size=cfg.vocab_size, model_path=args.tokenizer_path)
    else:
        from unified_inference import create_inference_engine
        engine = create_inference_engine(
            backend_type=args.backend,
            model_name=args.model,
            api_key=args.api_key,
            enable_prefix_caching=args.vllm_prefix_caching,
            enable_speculative=args.vllm_speculative,
            speculative_draft_model=args.vllm_draft_model,
            num_speculative_tokens=args.vllm_num_speculative_tokens,
            tensor_parallel_size=args.vllm_tp,
            gpu_memory_utilization=args.vllm_gpu_memory,
            dtype=args.vllm_dtype,
            quantization=args.vllm_quantization,
            enable_lora=args.vllm_lora,
            lora_dir=args.vllm_lora_dir,
        )
        if args.prompt:
            result = engine.generate(args.prompt)
            print(result)
        if args.interactive:
            print("Interactive mode (type 'quit' to exit)")
            while True:
                p = input(">>> ")
                if p.lower() in ("quit", "exit", "q"):
                    break
                result = engine.generate(p)
                print(result)
        return

    if args.prompt:
        ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], device=device)
        if args.speculative:
            out = model.generate_speculative(ids, max_new_tokens=args.max_new_tokens,
                                              temperature=args.temperature, draft_layers=args.spec_draft_layers,
                                              gamma=args.spec_gamma)
        elif args.beam > 0:
            out = model.generate_beam(ids, max_new_tokens=args.max_new_tokens, num_beams=args.beam,
                                       length_penalty=args.length_penalty)
        else:
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                 top_p=args.top_p, use_dola=args.dola, dola_gamma=args.dola_gamma,
                                 min_p=args.min_p)
        print(tokenizer.decode(out[0].tolist()))

    if args.interactive:
        print("Interactive mode (type 'quit' to exit)")
        while True:
            p = input(">>> ")
            if p.lower() in ("quit", "exit", "q"):
                break
            ids = torch.tensor([tokenizer.encode(p, add_bos=True)], device=device)
            if args.speculative:
                out = model.generate_speculative(ids, max_new_tokens=args.max_new_tokens,
                                                  temperature=args.temperature, draft_layers=args.spec_draft_layers,
                                                  gamma=args.spec_gamma)
            elif args.beam > 0:
                out = model.generate_beam(ids, max_new_tokens=args.max_new_tokens, num_beams=args.beam,
                                           length_penalty=args.length_penalty)
            else:
                out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                     top_p=args.top_p, use_dola=args.dola, dola_gamma=args.dola_gamma,
                                     min_p=args.min_p)
            print(tokenizer.decode(out[0].tolist()))


def cmd_serve(args):
    # vLLM 模式: 启动 vLLM OpenAI 兼容 Server
    if args.vllm:
        try:
            from vllm_integration import VLLMConfig, VLLMServerManager, SpeculativeDecodingConfig, LoRAConfig, VLLMServingConfig
            spec = SpeculativeDecodingConfig(
                enabled=args.vllm_speculative,
                draft_model=args.vllm_draft_model,
                num_speculative_tokens=args.vllm_num_speculative_tokens,
            )
            lora = LoRAConfig(
                enabled=args.vllm_lora,
                lora_dir=args.vllm_lora_dir,
            )
            serving = VLLMServingConfig(
                host=args.host,
                port=args.vllm_port,
                api_key=args.api_key,
                max_model_len=args.vllm_max_model_len or args.max_seq_len,
                gpu_memory_utilization=args.vllm_gpu_memory,
                tensor_parallel_size=args.vllm_tp,
                dtype=args.vllm_dtype,
                quantization=args.vllm_quantization,
                enable_chunked_prefill=args.vllm_chunked_prefill,
            )
            vllm_cfg = VLLMConfig(
                model_path=args.model,
                model_name=args.model_name or "emind",
                max_model_len=serving.max_model_len,
                gpu_memory_utilization=serving.gpu_memory_utilization,
                tensor_parallel_size=serving.tensor_parallel_size,
                enable_prefix_caching=args.vllm_prefix_caching,
                dtype=serving.dtype,
                use_server_mode=True,
                speculative=spec,
                lora=lora,
                serving=serving,
            )
            manager = VLLMServerManager(vllm_cfg)
            if manager.start(wait_ready=True):
                print(f"\nvLLM Server 已启动: http://{args.host}:{args.vllm_port}")
                print(f"API 文档: http://{args.host}:{args.vllm_port}/docs")
                print("按 Ctrl+C 停止...")
                try:
                    import time
                    while manager.is_running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\n正在停止 vLLM Server...")
                    manager.stop()
            else:
                print("vLLM Server 启动失败")
            return
        except ImportError:
            print("vLLM 集成模块不可用 (pip install vllm)")
            return

    # 默认: 启动 Emind Web 服务
    import subprocess
    cmd = [sys.executable or "python", "web_server.py", "--port", str(args.port)]
    if args.model:
        cmd.extend(["--model", args.model])
    subprocess.run(cmd)


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
        from unified_inference import create_inference_engine
        engine = create_inference_engine(
            backend_type=args.backend,
            api_key=args.api_key,
            enable_prefix_caching=True,
        )

        def model_fn(prompt: str) -> str:
            return engine.generate(prompt)

    benchmarks = args.benchmarks.split(",") if args.benchmarks else None
    runner = EvaluationRunner(model_fn)
    results = runner.run(benchmarks=benchmarks, sample_limit=args.sample_limit)
    runner.print_leaderboard(results)


def cmd_pipeline(args):
    from data_pipeline import DataCollector, DataCleaner, DataFormatter, DatasetManager

    # 蒸馏模式
    if args.distill_code or args.distill_reasoning or args.distill_deep_reasoning or args.distill_anti_hallucination or args.distill_identity:
        from training.distillation_pipeline import DistillationPipeline, DistillationConfig
        cfg = DistillationConfig(
            teacher_backend=args.teacher_backend,
            teacher_api_key=args.teacher_api_key,
            teacher_base_url=args.teacher_base_url,
            teacher_model=args.teacher_model,
            output_dir=args.distill_output,
            num_code_samples=args.distill_code or 0,
            num_reasoning_samples=args.distill_reasoning or 0,
            num_deep_reasoning_samples=args.distill_deep_reasoning or 0,
            num_anti_hallucination_samples=args.distill_anti_hallucination or 0,
            num_identity_samples=args.distill_identity or 0,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            strategies=args.distill_strategies.split(",") if args.distill_strategies else ["direct", "cot", "verify"],
        )
        pipeline = DistillationPipeline(cfg)
        data = pipeline.generate()
        out = os.path.join(args.distill_output, "distilled_sft.jsonl")
        print(f"Distilled {len(data)} samples → {out}")
        return

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


def cmd_vllm(args):
    """vLLM 诊断/自动配置"""
    try:
        from vllm_integration import detect_vllm_capabilities, auto_configure_for_gpu, VLLMConfig
    except ImportError:
        print("vLLM 集成模块不可用")
        return

    import torch
    TORCH_AVAILABLE = True

    try:
        from vllm import LLM as VLLM
        VLLM_AVAILABLE = True
    except ImportError:
        VLLM_AVAILABLE = False

    if args.detect:
        caps = detect_vllm_capabilities()
        import json
        print(json.dumps(caps, indent=2, ensure_ascii=False))
        if args.auto_configure:
            print("\n=== 自动推荐配置 ===")
            cfg = auto_configure_for_gpu()
            for k, v in cfg.__dict__.items():
                if not k.startswith("_"):
                    print(f"  {k}: {v}")
    elif args.info:
        print("\n[vLLM 信息]")
        print(f"  vLLM 已安装: {VLLM_AVAILABLE}")
        if VLLM_AVAILABLE:
            import vllm
            print(f"  vLLM 版本: {getattr(vllm, '__version__', 'unknown')}")
        print(f"  PyTorch CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    显存: {torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB")


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
    p.add_argument("--moe", action="store_true", help="启用 MoE（默认关闭）")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=500, help="每 N 步保存一次 checkpoint")
    p.add_argument("--logging-steps", type=int, default=10, help="每 N 步打印一次日志")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--dpo-beta", type=float, default=0.1, help="DPO beta temperature")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="DPO label smoothing")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--compile", action="store_true", help="启用 torch.compile 加速 (PyTorch 2.x)")
    p.add_argument("--neftune", type=float, default=0.0, help="NEFTune 噪声强度 (推荐 5.0)")
    p.add_argument("--curriculum-stages", type=int, default=1, help="课程学习阶段数 (按长度排序)")
    p.add_argument("--replay-ratio", type=float, default=0.0, help="Replay Buffer 比例 (防退化)")
    p.add_argument("--replay-data", default=None, help="Replay Buffer 数据文件 (JSONL)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"], help="优化器类型")
    p.add_argument("--use-fsdp", action="store_true")
    p.add_argument("--tokenizer-path", default=None, help="SentencePiece 模型路径")
    p.add_argument("--resume", default=None, help="从 checkpoint 恢复训练 (SFT/DPO)")

    # infer
    p = sub.add_parser("infer", help="推理")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--backend", default="cloud_api", choices=["cloud_api", "ollama", "llama_cpp", "huggingface", "vllm", "vllm_server"])
    p.add_argument("--api-key")
    # 当 checkpoint 不含 model_config 时的架构参数（可选）
    p.add_argument("--d-model", type=int, default=None, help="覆盖 checkpoint 中的模型维度")
    p.add_argument("--n-heads", type=int, default=None)
    p.add_argument("--n-kv-heads", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=None)
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--tokenizer-path", default=None, help="SentencePiece 模型路径")
    # vLLM 参数
    p.add_argument("--no-vllm-prefix-caching", action="store_false", dest="vllm_prefix_caching", default=True, help="禁用 Prefix Caching")
    p.add_argument("--vllm-speculative", action="store_true", help="启用 Speculative Decoding")
    p.add_argument("--vllm-draft-model", default=None, help="Draft 模型路径")
    p.add_argument("--vllm-num-speculative-tokens", type=int, default=5, help="推测解码 token 数")
    p.add_argument("--vllm-tp", type=int, default=1, help="Tensor Parallel 大小")
    p.add_argument("--vllm-gpu-memory", type=float, default=0.90, help="GPU 显存利用率")
    p.add_argument("--vllm-dtype", default="auto", help="数据类型 (auto/float16/bfloat16/fp8)")
    p.add_argument("--vllm-quantization", default=None, help="量化方式 (awq/gptq/fp8)")
    p.add_argument("--vllm-lora", action="store_true", help="启用 LoRA")
    p.add_argument("--vllm-lora-dir", default=None, help="LoRA 模块目录")
    # 量化推理
    p.add_argument("--dola", action="store_true", help="启用 DoLa 对比解码抑制幻觉")
    p.add_argument("--dola-gamma", type=float, default=0.5, help="DoLa 对比强度")
    p.add_argument("--min-p", type=float, default=0.0, help="Min-p 采样阈值 (如 0.05)")
    p.add_argument("--beam", type=int, default=0, help="Beam Search 宽度 (0=禁用)")
    p.add_argument("--length-penalty", type=float, default=1.0, help="Beam Search 长度惩罚")
    p.add_argument("--speculative", action="store_true", help="启用 Speculative Decoding")
    p.add_argument("--spec-gamma", type=int, default=4, help="Speculative gamma 值")
    p.add_argument("--spec-draft-layers", type=int, default=2, help="Draft 模型层数")
    p.add_argument("--kv-quant", action="store_true", help="KV Cache INT8 量化 (省显存)")
    p.add_argument("--quantize", default=None, choices=["int4", "fp8"], help="量化模式 (int4=权重 INT4, fp8=需要 H100)")
    p.add_argument("--quantize-groups", type=int, default=128, help="INT4 量化组大小")

    # serve
    p = sub.add_parser("serve", help="启动 Web 服务")
    p.add_argument("--port", type=int, default=3333)
    # vLLM Server 模式
    p.add_argument("--vllm", action="store_true", help="启动 vLLM OpenAI 兼容 Server (替代 Web 服务)")
    p.add_argument("--model", default=None, help="模型路径")
    p.add_argument("--model-name", default="emind", help="模型名称")
    p.add_argument("--host", default="0.0.0.0", help="监听地址")
    p.add_argument("--api-key", default=None, help="API Key")
    p.add_argument("--max-seq-len", type=int, default=4096, help="最大序列长度")
    p.add_argument("--vllm-port", type=int, default=8000, help="vLLM Server 端口")
    p.add_argument("--no-vllm-prefix-caching", action="store_false", dest="vllm_prefix_caching", default=True, help="禁用 Prefix Caching")
    p.add_argument("--vllm-speculative", action="store_true", help="启用 Speculative Decoding")
    p.add_argument("--vllm-draft-model", default=None, help="Draft 模型路径")
    p.add_argument("--vllm-num-speculative-tokens", type=int, default=5, help="推测解码 token 数")
    p.add_argument("--vllm-tp", type=int, default=1, help="Tensor Parallel 大小")
    p.add_argument("--vllm-gpu-memory", type=float, default=0.90, help="GPU 显存利用率")
    p.add_argument("--vllm-dtype", default="auto", help="数据类型 (auto/float16/bfloat16/fp8)")
    p.add_argument("--vllm-quantization", default=None, help="量化方式 (awq/gptq/fp8)")
    p.add_argument("--no-vllm-chunked-prefill", action="store_false", dest="vllm_chunked_prefill", default=True, help="禁用 Chunked Prefill")
    p.add_argument("--quantize", default=None, choices=["int4", "fp8"], help="量化模式")
    p.add_argument("--quantize-groups", type=int, default=128, help="INT4 量化组大小")
    p.add_argument("--vllm-max-model-len", type=int, default=None, help="vLLM 最大模型长度")
    p.add_argument("--vllm-lora", action="store_true", help="启用 LoRA")
    p.add_argument("--vllm-lora-dir", default=None, help="LoRA 模块目录")

    # eval
    p = sub.add_parser("eval", help="模型评测")
    p.add_argument("--model", default=None)
    p.add_argument("--benchmarks", default="mmlu,ceval,humaneval")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--sample-limit", type=int, default=None)
    p.add_argument("--backend", default="cloud_api")
    p.add_argument("--api-key")

    # rl
    p = sub.add_parser("rl", help="强化学习 (PPO / GRPO / RM)")
    p.add_argument("--rl-mode", choices=["ppo", "grpo", "rm"], default="ppo")
    p.add_argument("--data", default="data/rl.json")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--n-kv-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--kl-coef", type=float, default=0.1)
    p.add_argument("--clip-epsilon", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--mini-batch-size", type=int, default=4)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--ref-model", action="store_true", help="使用 reference model (KL 约束)")

    # pipeline
    p = sub.add_parser("pipeline", help="数据处理管线")
    p.add_argument("--collect", default=None, help="采集源文件或目录")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--process", action="store_true", help="处理已采集的数据")
    p.add_argument("--format", choices=["sft", "dpo", "pretrain", "alpaca"], default="sft")
    p.add_argument("--lang", default=None, help="语言过滤 (zh/en)")
    p.add_argument("--quality-threshold", type=float, default=0.1)

    # pipeline 蒸馏模式参数
    p.add_argument("--distill-code", type=int, default=0, help="蒸馏代码数据数量 (0=跳过)")
    p.add_argument("--distill-reasoning", type=int, default=0, help="蒸馏推理数据数量 (0=跳过)")
    p.add_argument("--distill-deep-reasoning", type=int, default=0, help="蒸馏深度推理数据数量 (0=跳过)")
    p.add_argument("--distill-anti-hallucination", type=int, default=0, help="蒸馏反幻觉数据数量 (0=跳过)")
    p.add_argument("--distill-identity", type=int, default=50, help="蒸馏身份认知数据数量 (0=跳过)")
    p.add_argument("--distill-output", default="data/distilled", help="蒸馏数据输出目录")
    p.add_argument("--distill-strategies", default="direct,cot", help="生成策略 (逗号分隔)")
    p.add_argument("--teacher-backend", default="cloud_api", help="Teacher 模型后端")
    p.add_argument("--teacher-api-key", default=None, help="Teacher API key")
    p.add_argument("--teacher-base-url", default=None, help="Teacher base URL")
    p.add_argument("--teacher-model", default=None, help="Teacher 模型名")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Teacher 生成最大 token 数")
    p.add_argument("--temperature", type=float, default=0.7, help="Teacher 生成温度")
    p.add_argument("--top-p", type=float, default=0.95, help="Teacher 生成 top-p")

    # vllm 子命令
    p = sub.add_parser("vllm", help="vLLM 诊断和自动配置")
    p.add_argument("--skip-detect", action="store_false", dest="detect", default=True, help="跳过 vLLM 能力检测")
    p.add_argument("--info", action="store_true", help="显示详细系统信息")
    p.add_argument("--auto-configure", action="store_true", help="自动生成推荐配置")

    # train-tokenizer 子命令
    p = sub.add_parser("train-tokenizer", help="训练 SentencePiece 分词器")
    p.add_argument("--data", default=None, help="语料 JSONL 文件")
    p.add_argument("--corpus", default=None, help="纯文本语料文件 (每行一条)")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--output-dir", default="tokenizers", help="模型输出目录")
    p.add_argument("--reencode", default=None, help="用新 tokenizer 重新编码 data 并输出到此路径")

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
    elif args.command == "rl":
        cmd_rl(args)
    elif args.command == "vllm":
        cmd_vllm(args)
    elif args.command == "train-tokenizer":
        cmd_train_tokenizer(args)


if __name__ == "__main__":
    main()
