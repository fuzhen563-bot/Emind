"""
合并 LoRA 权重到基座模型，生成可直接加载的 checkpoint。
用法: python scripts/merge_checkpoint.py --input path/to/model.pt --output path/to/merged.pt
"""
import torch, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import EmindConfig, create_model
from training.lora import apply_lora, merge_lora, lora_state_dict

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cpu"
    ckpt = torch.load(args.input, map_location=device, weights_only=False)
    cfg = EmindConfig.from_dict(ckpt.get("model_config", {}))
    model = create_model(cfg)

    # 先加载基座权重（不含 LoRA 的 strict=False 会跳过 LoRA 键）
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items() if "lora" not in k}, strict=False)

    # 应用 LoRA 并加载 LoRA 权重
    apply_lora(model)
    lora_keys = {k: v for k, v in state.items() if "lora" in k}
    if lora_keys:
        model.load_state_dict(lora_keys, strict=False)
        # 合并到基座
        merge_lora(model)
        print(f"LoRA weights merged ({len(lora_keys)} keys)")
    else:
        print("No LoRA weights found, saving as-is")

    # 保存合并后的 checkpoint
    out = {
        "model_config": ckpt.get("model_config", {}),
        "model_state_dict": model.state_dict(),
    }
    torch.save(out, args.output)
    print(f"Merged checkpoint saved: {args.output} ({os.path.getsize(args.output)//1024//1024} MB)")

if __name__ == "__main__":
    main()
