"""
zhengliu 蒸馏工具箱 — 交互式菜单

用法:
    python -m zhengliu                        交互菜单
    python -m zhengliu --code 200 ...         直接模式
"""
import os
import sys

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_pkg_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from zhengliu.config import DistillConfig, parse_args
from zhengliu.distill import DistillEngine, main as engine_main


BANNER = r"""
╔══════════════════════════════════════╗
║     zhengliu 蒸馏工具箱 v2.0          ║
║  基础蒸馏 · 进阶蒸馏 · 模型轮询       ║
╚══════════════════════════════════════╝
"""


def clear():
    os.system("clear" if os.name != "nt" else "cls")


def show_menu() -> str:
    clear()
    print(BANNER)
    print("  [1] 基础蒸馏 — 生成 SFT 数据")
    print("  [2] 进阶蒸馏 — 生成 DPO 偏好对")
    print("  [3] 一键蒸馏 — 自动发现模型 + 无限轮询")
    print("  [4] 质量控制 — 评分过滤 + 自我纠错")
    print("  [5] 预览 Prompt (dry-run)")
    print("  [6] API 配置管理")
    print("  [7] 查看输出目录")
    print("  [0] 退出")
    print()
    return input("  输入编号: ").strip()


def _get_config(mode="sft", extra=None):
    import json
    profile = os.path.join(_pkg_dir, "config", "profile.json")
    os.makedirs(os.path.dirname(profile), exist_ok=True)

    ak = os.environ.get("ZL_API_KEY", "")
    burl = os.environ.get("ZL_BASE_URL", "")
    mdl = "deepseek-v4-flash"

    if os.path.exists(profile):
        try:
            p = json.load(open(profile))
            ak = ak or p.get("api_key", "")
            burl = burl or p.get("base_url", "")
            mdl = p.get("model", mdl)
        except: pass

    if not ak: ak = input("API Key: ").strip()
    if not burl: burl = input("Base URL (默认 https://api.deepseek.com): ").strip() or "https://api.deepseek.com"

    print("\n数据配比 (回车=跳过):")
    code = int(input("  code: ").strip() or 0)
    reason = int(input("  reasoning: ").strip() or 0)
    deep_r = int(input("  deep-reasoning: ").strip() or 0)
    anti = int(input("  anti-hallucination: ").strip() or 0)
    ident = int(input("  identity: ").strip() or 0)

    tc = {}
    for n, t in [(code,"code"), (reason,"reasoning"), (deep_r,"deep_reasoning"),
                 (anti,"anti_hallucination"), (ident,"identity")]:
        if n: tc[t] = n
    if not tc: tc = {"code": 20, "reasoning": 10}

    cfg = DistillConfig(teacher="custom", api_key=ak, base_url=burl, model=mdl,
                        type_counts=tc, mode=mode, cot_depth=2, max_tokens=4096,
                        quality_check=False, multi_turn_correct=False,
                        no_cot=False, auto_runs=0, auto_discover=True)
    if extra: extra(cfg)
    return cfg


def cmd_sft():
    cfg = _get_config("sft")
    if not cfg: return
    print("\n▶ 基础蒸馏启动...")
    eng = DistillEngine(cfg)
    data = eng.run()
    if data: _save(cfg, data, "sft")
    input("\n按回车返回...")

def cmd_dpo():
    cfg = _get_config("dpo")
    if not cfg: return
    print("\n▶ 进阶蒸馏 (DPO) 启动...")
    eng = DistillEngine(cfg)
    data = eng.run()
    if data: _save(cfg, data, "dpo")
    input("\n按回车返回...")

def cmd_auto():
    cfg = _get_config("sft", lambda c: setattr(c, "auto_runs", -1))
    if not cfg: return
    print("\n▶ 一键蒸馏 (自动发现+无限轮询) 启动...")
    from zhengliu.pipeline import Pipeline
    pipe = Pipeline(cfg)
    pipe.auto_run()
    input("\n按回车返回...")

def cmd_quality():
    cfg = _get_config("sft", lambda c: (setattr(c, "quality_check", True),
                                         setattr(c, "multi_turn_correct", True)))
    if not cfg: return
    print("\n▶ 质量控制蒸馏启动...")
    eng = DistillEngine(cfg)
    data = eng.run()
    if data: _save(cfg, data, "sft")
    input("\n按回车返回...")

def cmd_dry():
    cfg = _get_config("sft")
    if not cfg: return
    cfg.dry_run = True
    eng = DistillEngine(cfg)
    eng.run()
    input("\n按回车返回...")

def cmd_config():
    import json
    profile_dir = os.path.join(_pkg_dir, "config")
    os.makedirs(profile_dir, exist_ok=True)
    path = os.path.join(profile_dir, "profile.json")
    print(f"\n配置文件: {path}")
    data = {}
    if os.path.exists(path):
        data = json.load(open(path))
        print(json.dumps(data, indent=2, ensure_ascii=False))
    ak = input("API Key (留空不变): ").strip()
    bu = input("Base URL (留空不变): ").strip()
    md = input("Model (留空不变): ").strip()
    if ak: data["api_key"] = ak
    if bu: data["base_url"] = bu
    if md: data["model"] = md
    with open(path, "w") as f: json.dump(data, f, indent=2, ensure_ascii=False)
    print("已保存")
    input("\n按回车返回...")

def cmd_output():
    out = os.path.join(_pkg_dir, "output")
    files = sorted(os.listdir(out), reverse=True) if os.path.isdir(out) else []
    print(f"\n输出目录: {out}")
    if not files:
        print("  暂无文件")
    else:
        for f in files[:20]:
            sz = os.path.getsize(os.path.join(out, f))
            print(f"  {f}  ({sz:,} bytes)")
    input("\n按回车返回...")

def _save(cfg, data, suffix):
    import json, time
    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = os.path.join(cfg.output_dir, f"{ts}_distilled_{suffix}.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in data: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已保存: {p} ({len(data)} 条)")


def main():
    # 有命令行参数 → 直接模式
    if len(sys.argv) > 1:
        engine_main()
        return

    menu = {"1": cmd_sft, "2": cmd_dpo, "3": cmd_auto, "4": cmd_quality,
            "5": cmd_dry, "6": cmd_config, "7": cmd_output}

    while True:
        c = show_menu()
        if c in menu:
            try:
                menu[c]()
            except KeyboardInterrupt:
                print("\n返回...")
            except Exception as e:
                print(f"\n错误: {e}")
                import traceback; traceback.print_exc()
                input("按回车继续...")
        elif c == "0":
            print("\n再见\n"); break
        else:
            input("无效, 按回车...")


if __name__ == "__main__":
    main()
