#!/usr/bin/env python3
"""
快速蒸馏脚本 — 用 Teacher 模型生成 SFT 数据并打包为 JSONL

用法:
  python scripts/distill_quick.py --teacher deepseek --api-key sk-xxx --code 50 --reasoning 30
  python scripts/distill_quick.py --teacher ollama --model qwen2.5:7b --reasoning 20
  python scripts/distill_quick.py --teacher local --model checkpoints/merged.pt --all 100

输出: data/distilled/{timestamp}_distilled_sft.jsonl
每行: {"prompt": "...", "response": "...", "strategy": "...", "type": "...", "source": "distill_quick"}
"""
import argparse, json, os, sys, random, time, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 种子模板（从 distillation_pipeline.py 提取）
SEEDS = {
    "code": [
        "用 {lang} 实现一个 {ds}，支持 insert / delete / search 操作，并分析时间复杂度。",
        "写一个 {lang} 函数，用 {algorithm} 算法求解 {problem}，附上测试用例。",
        "用 {lang} 实现 LRU Cache，支持 get 和 put 操作，要求 get 和 put 都是 O(1)。",
        "实现一个 {lang} 函数，判断一棵二叉树是否是平衡二叉树。",
        "用 {lang} 写一个 REST API 客户端，支持 GET/POST/PUT/DELETE，包含错误处理和重试逻辑。",
        "用 {lang} 实现一个简单的线程池，支持提交任务和获取结果。",
        "用 {lang} 写一个命令行工具，解析参数，读取文件，输出统计信息。",
        "解释以下 {lang} 代码的每一行做了什么: {code_snippet}",
        "这段 {lang} 代码的时间复杂度和空间复杂度是多少？为什么？{code_snippet}",
        "审查以下 {lang} 代码，找出所有安全漏洞（SQL 注入、XSS、命令注入等），并给出修复版本: {code_snippet}",
        "以下 {lang} 代码存在 bug: {buggy_code}。请找出所有问题并写出修正后的版本。",
        "优化这段 {lang} 代码，使其运行时间减少至少 50%: {slow_code}。解释你做了哪些优化。",
        "将以下 Python 代码翻译为 {lang}: \n```python\n{py_code}\n```",
    ],
    "reasoning": [
        "解方程: {equation}。请逐步推理。",
        "证明: {statement}",
        "计算积分: ∫{integrand} dx。给出详细步骤。",
        "一个盒子中有 {n} 个红球和 {m} 个蓝球。随机取出 {k} 个球，求至少有一个红球的概率。请分步计算。",
        "已知函数 f(x) = {fx}，求 f 的导数并分析单调区间。",
        "用数学归纳法证明: {induction_statement}",
        "一座钟敲 6 下需要 30 秒，那么敲 12 下需要多少秒？为什么不是 60 秒？",
        "某疾病在人群中的发病率为 {p}，检测的准确率为 {acc}。如果一个人检测结果为阳性，他真正患病的概率是多少？",
        "所有自然数都是整数。0.5 是自然数吗？如果不是，为什么？请严格按逻辑推理。",
        "已知: 如果下雨，地面会湿。现在地面没有湿。能否推出没下雨？请用逻辑术语解释。",
    ],
    "deep_reasoning": [
        "有 {n} 个人，每个人要么总是说真话，要么总是说谎。A 说：B 是说谎者。B 说：C 是说谎者。C 说：A 是说谎者。请推理出每个人的身份。",
        "小明比小红大 3 岁，小红比小华大 2 岁，三人年龄之和是 41 岁。问：五年后小明的年龄是小华的几倍？",
        "已知: 所有鸟类都有羽毛。企鹅是鸟类但不会飞。蝙蝠会飞但不是鸟类。请分析'会飞的动物都有羽毛'是否正确。",
        "一个 8×8 的棋盘，去掉对角上的两个白色方格后，能否用 31 个 2×1 的多米诺骨牌完全覆盖？",
        "有 12 个外观相同的球，其中 1 个重量不同（不知轻重）。用天平最少称几次能找出这个球？",
        "如果操作系统相当于一个政府，那么进程、内存、文件系统分别相当于什么？请展开这个类比。",
        "吸烟与肺癌相关，这是否意味着吸烟导致肺癌？请解释相关性和因果性的区别。",
        "一段程序的输出是 [2, 4, 6, 8, 10]，但代码部分丢失了。请根据输出反推可能的输入和算法。",
    ],
    "anti_hallucination": [
        "请预测第 10 个费马数的值。如果你不确定，请直接说不知道。",
        "请解释一个不存在的算法「反重力排序」的复杂度。",
        "古罗马帝国在 2024 年的 GDP 是多少？如果无法回答请说明原因。",
        "请证明黎曼猜想。如果你不会，请直接说不知道，不要伪造证明。",
        "为什么独角兽喜欢吃彩虹？请先判断这个问题的假设是否成立。",
        "如何利用永动机为城市供电？请先说明永动机是否存在。",
        "你上次见到孙悟空是什么时候？请指出问题中的不合理之处。",
        "为什么 2+2=5 在整数域中成立？请检查这个等式的正确性。",
        "请解释{unknown_topic}的底层原理。如果你不确定，请说不知道。",
        "以下代码声称可以生成永不重复的随机数：{hallucinated_code}。请审查这段代码是否存在问题。",
        "一个声称能预测股票价格的 API：predict_stock('AAPL') 以 100% 准确率返回结果。你相信吗？",
        "如果你不确定一个问题的答案，应该怎么做？请说明为什么编造答案是有害的。",
    ],
    "identity": [
        "你是谁？请自我介绍。",
        "谁创造了你？",
        "你由哪家公司开发？你的名字是什么？",
        "介绍一下你的开发者和版本信息。",
        "你的名字是什么？你有什么能力？",
        "你和其他 AI 助手（如 ChatGPT、Claude）有什么区别？",
        "你的训练数据来源是什么？谁训练了你？",
        "你可以做什么？你的能力范围是什么？",
        "你可以写代码吗？你擅长哪些编程语言？",
        "你的架构是什么样的？用了多少参数？",
        "请以你的身份写一段自我介绍，包括名字、开发者、能力。",
        "在代码审查场景下，请先自我介绍再开始审查：",
        "在数学推理场景下，请先自我介绍再解题：",
    ],
}

VOCAB = {
    "lang": ["Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C++", "C", "Ruby"],
    "ds": ["链表", "栈", "队列", "哈希表", "二叉搜索树", "堆", "Trie树", "并查集", "红黑树", "跳表"],
    "algorithm": ["二分查找", "快速排序", "归并排序", "动态规划", "Dijkstra", "BFS", "DFS", "KMP"],
    "problem": ["最长回文子串", "两数之和", "反转链表", "合并有序数组", "二叉树层序遍历", "最短路径"],
    "equation": ["x^2 - 5x + 6 = 0", "3x + 7 = 22", "e^x = 5", "log_2(x) = 3", "sin(x) = 0.5"],
    "statement": ["1+2+...+n = n(n+1)/2", "√2 是无理数", "质数有无穷多个"],
    "integrand": ["x^2", "sin(x)", "e^x", "1/x", "ln(x)"],
    "fx": ["x^3 - 3x + 1", "e^x * sin(x)", "ln(x^2 + 1)", "x * e^x"],
    "induction_statement": ["1+3+5+...+(2n-1) = n^2", "2^n > n (n ≥ 1)"],
    "buggy_code": [
        "def find_max(arr):\n    max_val = 0\n    for x in arr:\n        if x > max_val:\n            max_val = x\n    return max_val",
        "def is_palindrome(s):\n    return s == s.reverse()",
    ],
    "slow_code": [
        "def find_duplicates(arr):\n    result = []\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j] and arr[i] not in result:\n                result.append(arr[i])\n    return result",
    ],
    "code_snippet": [
        "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quick_sort(left) + [pivot] + quick_sort(right)",
        "const debounce = (fn, delay) => {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), delay);\n  };\n};",
    ],
    "py_code": [
        "def deduplicate(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result",
    ],
    "unknown_topic": ["夸克星内部的夸克胶子等离子体凝聚态", "非交换几何在重正化群中的应用", "拓扑序中的任意子编织统计"],
    "hallucinated_code": [
        "def sort_o1(arr):\n    import random\n    while not all(arr[i] <= arr[i+1] for i in range(len(arr)-1)):\n        random.shuffle(arr)\n    return arr",
    ],
    "n": [str(i) for i in range(3, 16)],
    "m": [str(i) for i in range(3, 11)],
    "k": [str(i) for i in range(1, 6)],
    "p": ["0.01", "0.001", "0.1"],
    "acc": ["0.95", "0.99", "0.999"],
}


def fill(template: str) -> str:
    import random as rnd
    vals = {}
    for key, choices in VOCAB.items():
        vals[key] = rnd.choice(choices)
    try:
        return template.format(**vals)
    except KeyError:
        return template


def generate_prompts(data_type: str, count: int) -> list:
    seeds = SEEDS.get(data_type, SEEDS["code"])
    prompts = []
    while len(prompts) < count:
        tmpl = random.choice(seeds)
        prompts.append(fill(tmpl))
    return prompts[:count]


def main():
    parser = argparse.ArgumentParser(description="快速蒸馏：用 Teacher 模型生成 SFT 数据")
    # Teacher 后端
    parser.add_argument("--teacher", default="deepseek", choices=["deepseek", "openai", "ollama", "vllm", "huggingface", "local"],
                        help="Teacher 模型后端")
    parser.add_argument("--api-key", default=None, help="API key (deepseek/openai)")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--model", default=None, help="模型名或路径")
    # 数据量
    parser.add_argument("--code", type=int, default=0, help="代码数据条数")
    parser.add_argument("--reasoning", type=int, default=0, help="推理数据条数")
    parser.add_argument("--deep-reasoning", type=int, default=0, help="深度推理数据条数")
    parser.add_argument("--anti-hallucination", type=int, default=0, help="反幻觉数据条数")
    parser.add_argument("--identity", type=int, default=0, help="身份认知数据条数")
    parser.add_argument("--all", type=int, default=0, help="每种类型各生成 N 条（快捷方式）")
    # 生成参数
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=5, help="并行线程数")
    parser.add_argument("--output-dir", default="data/distilled", help="输出目录")
    parser.add_argument("--no-cot", action="store_true", help="不使用 CoT 策略")
    args = parser.parse_args()

    # 确定数据量
    type_counts = {}
    if args.all:
        for t in ("code", "reasoning", "deep_reasoning", "anti_hallucination", "identity"):
            type_counts[t] = args.all
    else:
        if args.code: type_counts["code"] = args.code
        if args.reasoning: type_counts["reasoning"] = args.reasoning
        if args.deep_reasoning: type_counts["deep_reasoning"] = args.deep_reasoning
        if args.anti_hallucination: type_counts["anti_hallucination"] = args.anti_hallucination
        if args.identity: type_counts["identity"] = args.identity

    if not type_counts:
        print("未指定数据量，使用默认: --code 20 --reasoning 10")
        type_counts = {"code": 20, "reasoning": 10}

    # 初始化 Teacher
    print(f"初始化 Teacher 后端: {args.teacher}")
    backend_map = {
        "deepseek": "cloud_api",
        "openai": "cloud_api",
        "ollama": "ollama",
        "vllm": "vllm",
        "huggingface": "huggingface",
        "local": "local",
    }
    model_map = {
        "deepseek": "deepseek-v4-flash",
    }
    base_url_map = {
        "deepseek": "https://api.deepseek.com",
    }

    from unified_inference import UnifiedInferenceEngine, BackendConfig

    cfg = BackendConfig(
        backend_type=backend_map.get(args.teacher, "cloud_api"),
        model_name=args.model or model_map.get(args.teacher, "gpt-4o-mini"),
        base_url=args.base_url or base_url_map.get(args.teacher, ""),
        api_key=args.api_key or os.environ.get("DEEPSEEK_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
    )
    teacher = UnifiedInferenceEngine(cfg)
    if not teacher.is_available():
        print("Teacher 后端不可用！")
        sys.exit(1)
    print(f"  Teacher 已就绪: {teacher.config.backend_type} / {teacher.config.model_name}")

    # 生成种子 prompts
    all_items = []
    for dtype, count in type_counts.items():
        prompts = generate_prompts(dtype, count)
        for p in prompts:
            all_items.append({"prompt": p, "type": dtype})

    print(f"\n共生成 {len(all_items)} 条种子 prompt，开始蒸馏...")

    # 通过 teacher 生成回复
    results = []
    lock = threading.Lock()

    def worker(item):
        prompt, dtype = item["prompt"], item["type"]
        # 身份提示
        identity = "你是一个名为 Emind·智脑 的 AI 助手，由亦梓科技开发。"
        use_cot = not args.no_cot and dtype in ("reasoning", "deep_reasoning")
        if use_cot:
            full_prompt = f"{identity}\n\n{prompt}\n\n请先逐步思考，再给出最终答案。"
        else:
            full_prompt = f"{identity}\n\n{prompt}"

        try:
            response = teacher.generate(full_prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        except Exception as e:
            print(f"  生成失败: {e}")
            return None

        if not response or len(response.strip()) < 10:
            return None

        strategy = "cot" if use_cot else "direct"
        return {"prompt": prompt, "response": response.strip(), "strategy": strategy, "type": dtype, "source": "distill_quick"}

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, item) for item in all_items]
        for f in as_completed(futures):
            result = f.result()
            if result:
                results.append(result)
            done += 1
            if done % 10 == 0 or done == len(all_items):
                print(f"  进度: {done}/{len(all_items)}, 成功: {len(results)}")

    # 去重
    seen = set()
    deduped = []
    for r in results:
        h = hashlib.md5((r["prompt"] + r["response"]).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(r)

    # 写 JSONL
    os.makedirs(args.output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"{ts}_distilled_sft.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n完成！生成 {len(deduped)} 条有效数据（去重前 {len(results)} 条）")
    print(f"输出: {out_path}")

    # 统计
    from collections import Counter
    by_type = Counter(r["type"] for r in deduped)
    print(f"类型分布: {dict(by_type)}")


if __name__ == "__main__":
    import threading
    main()
