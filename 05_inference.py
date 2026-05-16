#!/usr/bin/env python3
"""
[DEPRECATED] This file is superseded by unified_inference.py. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use unified_inference.py instead.", DeprecationWarning, stacklevel=2)

"""
Emind 推理脚本 — 加载模型并交互式推理
支持本地模型和云端 API 后端

用法:
    python 05_inference.py --model-path checkpoints/latest/model.pt
    python 05_inference.py --backend cloud_api --interactive
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import EmindConfig, create_model
from tokenizer import EmindTokenizer


def load_model(model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg_data = ckpt.get("model_config") or ckpt.get("config", {})
    cfg = EmindConfig.from_dict(cfg_data) if isinstance(cfg_data, dict) else cfg_data
    model = create_model(cfg)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, cfg


def interactive(model, tokenizer, device, max_new_tokens=512, temperature=0.8, top_p=0.9):
    print("Emind 交互式推理 (输入 'quit' 退出)\n")
    while True:
        prompt = input(">>> ")
        if prompt.lower() in ("quit", "exit", "q"):
            break
        ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], device=device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        response = tokenizer.decode(out[0].tolist())
        print(f"Emind: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Emind 推理")
    parser.add_argument("--model-path", type=str, default="checkpoints/latest/model.pt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--prompt", type=str, default=None, help="单次推理 prompt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = EmindTokenizer()

    if not os.path.exists(args.model_path):
        print(f"模型不存在: {args.model_path}")
        print("使用云端 API 后端 (unified_inference.py)")
        from unified_inference import UnifiedInferenceEngine, BackendConfig
        engine = UnifiedInferenceEngine(BackendConfig(backend_type="cloud_api", api_key=os.environ.get("API_KEY")))
        if args.prompt:
            print(engine.generate(args.prompt))
        elif args.interactive:
            while True:
                p = input(">>> ")
                if p.lower() in ("quit", "exit", "q"):
                    break
                print(engine.generate(p))
        return

    model, cfg = load_model(args.model_path, device)
    print(f"模型加载完成: {sum(p.numel() for p in model.parameters())/1e6:.2f}M 参数")

    if args.prompt:
        ids = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], device=device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
        print(tokenizer.decode(out[0].tolist()))

    if args.interactive:
        interactive(model, tokenizer, device, args.max_new_tokens, args.temperature, args.top_p)


if __name__ == "__main__":
    main()
