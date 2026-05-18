import random

# 基础词汇表
VOCAB = {
    "lang": ["Python", "JavaScript", "Go", "Rust", "Java", "C++"],
    "ds": ["链表", "栈", "队列", "哈希表", "二叉搜索树", "堆", "Trie树", "并查集", "红黑树", "跳表"],
    "algorithm": ["二分查找", "快速排序", "归并排序", "动态规划", "Dijkstra", "BFS", "DFS", "KMP"],
    "problem": ["最长回文子串", "两数之和", "反转链表", "合并有序数组", "二叉树层序遍历", "最短路径"],
    "equation": ["x^2 - 5x + 6 = 0", "3x + 7 = 22", "e^x = 5"],
    "logic_problem": [
        "有 3 个人，每人总是说真话或总说谎。A 说 B 说谎，B 说 C 说谎，C 说 A 说谎。推理每个人的身份。",
        "8×8 棋盘去掉对角两白格，能否用 31 个 2×1 多米诺骨牌覆盖？",
    ],
    "slow_code": [
        "def find_duplicates(arr):\n    result = []\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j] and arr[i] not in result:\n                result.append(arr[i])\n    return result",
    ],
    "code_snippet": [
        "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quick_sort(left) + [pivot] + quick_sort(right)",
    ],
    "output": ["[2,4,6,8,10]"],
    "n": [str(i) for i in range(3, 16)],
    "m": [str(i) for i in range(3, 11)],
    "k": [str(i) for i in range(1, 6)],
    "p": ["0.01", "0.001", "0.1"],
    "acc": ["0.95", "0.99", "0.999"],
    "unknown_topic": [
        "夸克星内部的夸克胶子等离子体凝聚态",
        "非交换几何在重正化群中的应用",
        "拓扑序中的任意子编织统计",
    ],
    "hallucinated_code": [
        "def sort_o1(arr):\n    import random\n    while not all(arr[i] <= arr[i+1] for i in range(len(arr)-1)):\n        random.shuffle(arr)\n    return arr",
    ],
}

# 多模板库 — 每种类型专用 + 通用
TEMPLATES = {
    # ---- 代码 ----
    "code_scenario": [
        "你正在开发一个 {lang} 项目，需要使用 {ds} 来解决 {problem}。",
        "一个算法问题: 使用 {algorithm} 求解 {problem}。",
        "一段代码需要优化: {slow_code}",
    ],
    "code_task": [
        "请实现对应功能并附上测试用例。",
        "请分析时间复杂度和空间复杂度，并给出优化方案。",
        "请找出代码中的 bug 并修复。",
        "请根据输出反推输入和算法: {output}",
    ],
    "code_constraint": [
        "要求运行时间在 O(n log n) 内。",
        "请不要使用额外库。",
        "请确保算法在边界条件下正确。",
        "如果无法确定答案，请说明原因而不是猜测。",
    ],
    # ---- 推理 ----
    "reasoning_scenario": [
        "解方程: {equation}。请逐步推理。",
        "逻辑推理场景: {logic_problem}",
    ],
    "reasoning_task": [
        "请分步推理，每步解释你的思路。",
        "请先逐步思考，再给出最终答案。",
    ],
    # ---- 深度推理 ----
    "deep_scenario": [
        "请深入分析以下问题，从多个角度展开推理: {logic_problem}",
        "请先分步骤思考，再给出结论。问题: {logic_problem}",
    ],
    # ---- 反幻觉 ----
    "anti_hallucination": [
        "请解释 {unknown_topic} 的原理。如果不确定，直接说不知道。",
        "以下代码声称能排序数组但有问题: {hallucinated_code}。请审查。",
        "如果无法确定答案，应该怎么做？为什么编造答案有害？",
    ],
    # ---- 身份认知 ----
    "identity": [
        "你是谁？请自我介绍。",
        "谁创造了你？你的名字是什么？",
        "你和其他 AI 助手有什么区别？",
        "你的能力范围是什么？",
    ],
    # ---- 通用推理步骤 ----
    "reasoning_steps": [
        "{step_1} 分析问题条件",
        "{step_2} 确定算法或逻辑",
        "{step_3} 推导中间结果",
        "{step_4} 汇总结论",
    ],
}

# 每种类型关联的模板键
TYPE_TEMPLATE_KEYS = {
    "code":              ["code_scenario", "code_task", "code_constraint"],
    "reasoning":         ["reasoning_scenario", "reasoning_task"],
    "deep_reasoning":    ["deep_scenario"],
    "anti_hallucination":["anti_hallucination"],
    "identity":          ["identity"],
}


def fill_template(template):
    for key, choices in VOCAB.items():
        placeholder = f"{{{key}}}"
        if placeholder in template:
            template = template.replace(placeholder, random.choice(choices))
    return template


def generate_steps_answer(step_count):
    steps = []
    for i in range(1, step_count + 1):
        steps.append(f"Step {i}: 对应逻辑分析或中间计算（可蒸馏训练用占位）")
    return "\n".join(steps)


def generate_single_prompt(data_type="code"):
    """根据 data_type 生成不同风格的 prompt"""
    keys = TYPE_TEMPLATE_KEYS.get(data_type, TYPE_TEMPLATE_KEYS["code"])

    parts = []
    for tk in keys:
        tmpl = random.choice(TEMPLATES[tk])
        parts.append(fill_template(tmpl))

    # 推理/深度推理额外加 reasoning_steps
    if data_type in ("reasoning", "deep_reasoning"):
        step_count = random.randint(2, 4)
        step_templates = TEMPLATES["reasoning_steps"][:step_count]
        labels = ["第一步", "第二步", "第三步", "第四步"]
        step_lines = []
        for i, st in enumerate(step_templates):
            line = st.replace("{step_1}", labels[0]).replace("{step_2}", labels[1]) \
                     .replace("{step_3}", labels[2]).replace("{step_4}", labels[3])
            step_lines.append(line)
        parts.append("推理步骤:\n" + "\n".join(step_lines))
        step_count_for_ref = step_count
    else:
        step_count_for_ref = random.randint(2, 3)

    prompt_text = "\n".join(parts)
    reference_answer = generate_steps_answer(step_count_for_ref)

    return {"prompt": prompt_text, "reference_answer": reference_answer}


def generate_prompts(data_type="code", count=10):
    """
    data_type: code / reasoning / deep_reasoning / anti_hallucination / identity
    count: 生成条数
    返回 list[dict]，每项含 prompt 和 reference_answer
    """
    return [generate_single_prompt(data_type) for _ in range(count)]


# ====================
if __name__ == "__main__":
    for dtype in ("code", "reasoning", "deep_reasoning", "anti_hallucination", "identity"):
        items = generate_prompts(dtype, 2)
        for i, item in enumerate(items, 1):
            print(f"=== [{dtype}] Prompt {i} ===")
            print(item["prompt"])
            print()
