<<<<<<< HEAD
"""
种子模板系统 v2.0
  核心原则: 真实推理 > 格式化 CoT，动态轨迹 > 固定步骤
  Identity 作为隐式行为约束，不出现在通用 prompt 中
"""
import random

# =============================================================================
# 基础词汇池
# =============================================================================

VOCAB = {
    "lang": ["Python", "JavaScript", "Go", "Rust", "Java", "C++", "TypeScript", "Zig"],
    "ds": ["链表", "哈希表", "堆", "Trie", "并查集", "跳表", "B树", "布隆过滤器", "线段树", "LRU缓存"],
    "algorithm": ["二分", "快排", "归并", "动规", "Dijkstra", "BFS/DFS", "KMP", "贪心", "回溯", "双指针"],
    "problem": ["最长回文子串", "两数之和", "反转链表", "层序遍历", "最短路径", "股票买卖", "岛屿数量"],
    "equation": ["x^2 - 5x + 6 = 0", "e^x = 5", "3x + 7 = 22"],
    "logic_problem": [
        "有三个人总是说真话或说谎，A说B说谎，B说C说谎，C说A说谎。找出谁说真话。",
        "8×8棋盘去掉对角两白格，31个2×1骨牌能否覆盖？",
    ],
    "n": [str(i) for i in range(3, 16)],
    "m": [str(i) for i in range(3, 11)],
    # 真实代码片段 — 含 bug / 不完整 / 低效
    "slow_code": [
        "def find_dups(arr):\n    r=[]\n    for i in range(len(arr)):\n        for j in range(i+1,len(arr)):\n            if arr[i]==arr[j] and arr[i] not in r:\n                r.append(arr[i])\n    return r",
        "def fib(n):\n    if n<=1: return n\n    return fib(n-1)+fib(n-2)  # 指数级",
    ],
    "real_bug": [
        "def divide(a,b): return a/b  # ZeroDivisionError 未处理",
        "def fetch_user(id):\n    users={'admin':'root'}\n    return users[id]  # KeyError",
        "for i in range(len(arr)):\n    arr.pop(i)  # 迭代中修改列表",
    ],
    "stacktrace": [
        "Traceback:\n  File 'app.py', line 42, in process\n    result = data['key']\nTypeError: 'NoneType' object is not subscriptable",
        "IndexError: list index out of range at line 78\n  items = cache[hash_idx]  # hash_idx = -1 when cache miss",
    ],
    # 噪声 — 真实互联网的混乱
    "casual_tone": [
        "离谱", "炸了", "感觉不对", "有点怪", "bro", "无语",
        "这个怎么搞", "不太对劲", "试了好多遍还是不行",
    ],
    "incomplete_desc": [
        "有个数组，里面好像有重复元素...",
        "大概是这样的一段代码，跑出来不对",
        "数据库有个表，字段大概是这样的",
        "具体需求记不清了，反正是要统计...",
    ],
    "wrong_term": [
        "哈希树", "动态递归", "循环数组", "协程锁",
        "内存缓存池", "并发链表", "异步递归",
    ],
    "ocr_noise": [
        "Dijkstrra", "QuickSortt",
        "B树,一种自平横的搜索树",
        "complaxity", "recusion",
    ],

    # 真实用户口吻
    "real_user": [
        "这个为什么跑这么慢，我看了半天不知道问题在哪",
        "代码死锁了，帮忙看下咋修，已经调了一下午了",
        "线上 MySQL 查询突然慢了几十倍，不知道是不是索引问题",
        "这段能优化不，现在过不了时间限制",
        "我写的快排在大数据量下反而不如冒泡，离谱",
        "帮忙 review 一下这段代码，总觉得有并发问题但说不上来",
        "递归爆栈了，怎么改成非递归的",
        "这个算法我感觉不对，但说不出来哪里错",
        "并发下偶尔丢数据，概率很低，排查了三天没找到",
        "这个证明 我总觉得少了什么 但又说不清",
        "老板要加个功能 完全改架构 怎么办",
    ],

    # 多文件仓库上下文
    "repo_files": [
        "controller.py: 接收 HTTP 请求,调用 service\nservice.py: 业务逻辑,调用 dao\ndao.py: 数据库操作\nconfig.py: 全局配置",
        "api/handler.go: gRPC handler\ninternal/logic.go: 核心算法\npkg/cache.go: 缓存层\nmigrations/001.sql: 表结构",
        "src/router.ts: 前端路由\nsrc/store.ts: 状态管理\nsrc/api.ts: 后端调用\nserver/middleware.ts: 鉴权",
    ],
    "repo_bug": [
        "controller.py 里有个 bug，但在本地跑不出来，生产环境偶尔触发。可能是 service.py 的缓存没处理好并发。相关代码在三个文件里。",
        "改了一行 config.py 的配置，整个系统的连接池全崩了。不知道为什么会影响 dao.py 的初始化。",
        "前端改了一个 API 调用的参数名，后端也跟着改了，结果老版本的客户端全挂了。",
    ],
=======
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
>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329
    "unknown_topic": [
        "夸克星内部的夸克胶子等离子体凝聚态",
        "非交换几何在重正化群中的应用",
        "拓扑序中的任意子编织统计",
<<<<<<< HEAD
        "AdS/CFT对偶中bulk重构的量子纠错码解释",
    ],
    "fake_code": [
        "def solve_np_complete_in_p(arr):\n    return sorted(arr)  # ?",
    ],
    "real_code": [
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = (lo+hi)//2\n        if arr[mid]==target: return mid\n        elif arr[mid]<target: lo=mid+1\n        else: hi=mid-1\n    return -1",
    ],
}

# =============================================================================
# 推理思维模式 — 真实思考的动态轨迹
# =============================================================================

THINKING_STYLES = {
    "self_doubt": [
        "等等，这个前提可能有问题。",
        "如果反向思考呢？",
        "这里存在一个隐含假设，需要确认。",
        "直觉上可以这样做，但需要验证。",
    ],
    "constraint_check": [
        "边界条件是什么？",
        "极端输入会怎样？",
        "有没有遗漏的特殊情况？",
        "当输入为空时…",
    ],
    "counter_example": [
        "如果 n=0 呢？",
        "考虑反例：...",
        "这个算法在稠密图上会退化。",
    ],
    "exploration": [
        "还有一种思路：...",
        "换个角度，从数据结构入手。",
        "能不能转化成图论问题？",
    ],
    "revision": [
        "不对，刚才的想法有漏洞。",
        "重新审视约束条件。",
        "忽略了重复元素的情况。",
    ],
    "convergence": [
        "综合来看，最优方案是...",
        "在时间和空间之间折中。",
        "根据实际数据分布选择方案。",
    ],
}

# =============================================================================
# 噪声化 Code 模板 — 真实工程语气
# =============================================================================

NOISE_CODE_TEMPLATES = [
    # 非正式口吻
    "{real_user}",

    # 不完整描述
    "有个 {lang} 项目，{repo_files}\n{incomplete_desc}",

    # 仓库级 bug
    "{repo_bug}",

    # 多文件重构
    "这个系统的代码结构大概是：\n{repo_files}\n需要重构 service 层，但不能影响 controller 的接口。还要保证 dao 的兼容性。给出方案。",

    # 错误术语 + 真实需求
    "我想用 {wrong_term} 实现一个缓存，性能和 {ds} 差不多就行。数据量大概 {n} 万条。",

    # OCR 噪声
    "之前看的 {algorithm} 算法，具体是 {ocr_noise}，我照着写的但跑不对。帮我看看。",
]

# =============================================================================
# 长程递归推理 — 状态演化 (5+ 轮)
# =============================================================================

LONG_HORIZON_TEMPLATES = [
    # 系统设计 → 瓶颈 → 重构 → 扩展 → 修复
    [
        "设计一个实时排行榜系统，支持按分数查询前100名。给出初版架构。",
        "用户量增长到百万级，当前方案在高峰期延迟到 2 秒。分析瓶颈在哪里。",
        "基于你的分析，重构系统。保留核心接口不变，内部大改。",
        "需求新增：要支持过去 24 小时的热榜（需要考虑时间衰减）。调整你的设计。",
        "如果突然来了流量洪峰（日常 10 倍），你上一步的方案能撑住吗？不能的话改哪里？",
    ],
    # Bug 链 → 调查 → 部分修复 → 回归 → 最终修复
    [
        "用户反馈每天都有一两秒钟系统返回 500。没有 pattern。从哪里开始排查？",
        "你在日志里发现了一个异常堆栈。分析可能的根因。",
        "给了第一个修复方案，但 QA 说出现了新的问题：之前工作正常的批量导出功能会随机失败。",
        "回溯你的修复，找出为什么引入了新 bug。重新设计修复。",
        "线上跑了 24 小时，看起来正常了。但你觉得还有隐藏风险吗？",
    ],
    # 算法优化 → 尝试 → 碰壁 → 换方向 → 成功
    [
        "处理一个 {n}G 的日志文件，提取访问量最高的 IP。内存只能 {m}G。先给一个初步方案。",
        "你的方案在小样本测试通过，但在真实数据上 OOM 了。为什么？",
        "换一种数据结构重新设计。分析新方案的复杂度。",
        "如果日志格式变了（JSON → 纯文本），需要改多少？",
        "把所有优化方案合并，给出最终的生产级代码。",
    ],
]


def generate_long_conversation(count: int) -> list:
    """生成长程推理对话"""
    results = []
    for _ in range(count):
        turns = random.choice(LONG_HORIZON_TEMPLATES)
        results.append({
            "type": "conversation",
            "turns": [fill(t) for t in turns],
        })
    return results

# =============================================================================
# 真实失败路径 — 不是"表演修正"，而是真的走错再回来
# =============================================================================

FAILURE_PATH_TEMPLATES = [
    "用 {wrong_term} 对 {problem} 做了一次求解，复杂度 O({n}²)，数据量大时就炸了。正确应该怎么解？",

    "用 {algorithm} 做了一遍 {problem}，但测试发现随机数据都过不了。可能是边界条件没处理好。请重新实现。",

    "按照 {ds} 的思路写了 {lang} 实现，跑了 1000 个随机测试，{m} 个没过。分析问题出在哪。",

    "第一版用贪心过了样例，但提交发现 WA（错误答案）。构造一个贪心会失败的例子，然后给正确解法。",
]


def generate_failure_paths(count: int) -> list:
    """生成真实失败路径 prompt"""
    return [fill(random.choice(FAILURE_PATH_TEMPLATES)) for _ in range(count)]

def fill(template: str) -> str:
    """替换 {key} 为词汇池中的随机值"""
    result = template
    for key in VOCAB:
        if "{" + key + "}" in result:
            result = result.replace("{" + key + "}", random.choice(VOCAB[key]))
    return result


TEMPLATES = {
    # =========================================================================
    # 代码 — 真实工程场景
    # =========================================================================
    "code": [
        "用 {lang} 实现 {ds}。分析时间和空间复杂度。",

        "下面是 O(n²) 的 {algorithm} 实现。请改写为 O(n log n)：\n```\n{slow_code}\n```",

        "以下代码有 bug：\n```\n{real_bug}\n```\n请修复并解释原因。",

        "这段代码运行时报错：\n```\n{stacktrace}\n```\n定位问题并给出修复方案。",

        "比较 {algorithm} 和 {algorithm} 解决 {problem} 的优劣。什么场景下选哪个？",

        "实现 {ds}，要求支持以下操作并分析均摊复杂度。",

        "以下 PR 中有一个并发安全问题，请 review：\n```{lang}\n# shared counter\ndef worker(n):\n    global counter\n    for _ in range(n):\n        counter += 1\n```",

        "用 {lang} 写一个带 LRU 淘汰策略的缓存。内存不能无限增长。",

        "这段递归代码会栈溢出。改为迭代版本：\n```\n{slow_code}\n```",

        "设计一个 {ds} 的 API，考虑线程安全和内存效率。",

        "给一个已有项目的 {lang} 代码加单元测试。当前没有任何测试。",

        "重构这段代码，提升可读性和性能：\n```\n{slow_code}\n```",

        "实现 {algorithm}，不允许使用任何内置排序函数。",

        "这段代码通过了测试，但有隐蔽的性能陷阱。找到它：\n```\n{real_code}\n```",

        "需要处理一个流式日志文件（10GB），每行是 JSON。找出出现次数最多的 IP 地址。内存限制 512MB。",
    ],

    # =========================================================================
    # 推理 — 动态思维轨迹
    # =========================================================================
    "reasoning": [
        "有 {n} 个人参加圆桌会议，每个人总是说真话或总说谎。"
        "A 说 B 说谎，B 说 C 说谎，...如此循环。"
        "如果 {n} 是偶数，能否确定谁说真话？如果是奇数呢？",

        "一个 {m}×{m} 的棋盘，左上角到右下角的最短路径数是多少？允许任意方向还是只能右下？两种情况分别回答。",

        "你需要证明或推翻：对于任意正整数 n，n² + n + 41 是否恒为素数？如果不是，给出最小反例。",

        "3 个人，5 顶帽子（3 蓝 2 红），每人戴一顶，能看到别人看不到自己。"
        "依次问 A、B、C 知不知道自己帽子的颜色。推理每个人的回答。",

        "如果 x 和 y 满足 x² + y² = 1，求 x + y 的最大最小值。至少用两种方法验证。",

        "分析以下命题的真伪：对于任意无向图，如果每个顶点度数 ≥ n/2，则图一定连通。给出证明或反例。",

        "{logic_problem}",

        "两个水壶分别 5L 和 3L，如何量出 4L 水？能否推广到任意容量 {m}L 和 {n}L？",

        "估算以下数量级：地球上一年的降雨量如果用 500ml 矿泉水瓶装，可以摆满多少个足球场？给出推理过程。",

        "如果一台量子计算机有 {n} 个量子比特，它能同时表示多少个状态？和经典计算机的本质区别在哪？",
    ],

    # =========================================================================
    # 深度推理 — 含自我修正和探索
    # =========================================================================
    "deep_reasoning": [
        "证明并查集路径压缩后，每次 find 操作的均摊复杂度是 O(α(n))。"
        "请：1)给出 Ackermann 反函数的直观理解；2)证明关键引理；3)讨论为什么实际中 α(n) ≤ 4。",

        "设计一个分布式系统，存储 {n} 亿个 key-value 对，要求："
        "1) 单点故障不影响服务 2) 读写延迟 < 10ms 3) 可扩展到 {m} 亿。"
        "给出架构图、一致性策略和容量估算。如果初始方案有瓶颈，修正它。",

        "证明或推翻：P = NP。这不是让你真的证明，而是："
        "1) 概述目前学界的主流观点和各自证据 2) 如果 P=NP 会怎样 3) 如果 P≠NP 会怎样 4) 量子计算能否改变结论。",

        "分析 {algorithm} 和 {algorithm} 在不同数据分布下的时间复杂度（最好、最坏、平均）。"
        "在流式数据场景下，哪种更适合？如果数据有偏斜呢？",

        "如何设计一个 URL 短链接系统？从数据库选型、ID 生成策略、缓存设计、并发控制、全球部署五个维度分析。"
        "对每个设计决策，给出替代方案并比较。",

        "以下神经网络梯度消失的三种解决方案：1)BatchNorm 2)残差连接 3)ReLU 激活函数。"
        "解释各自的原理、适用场景和局限性。在什么情况下它们仍然会失败？",

        "给出并证明 {problem} 的最优时间复杂度下界。先尝试信息论下界，然后考虑归约。"
        "如果归约到已知的 Ω(n log n) 问题但你自己发现了一个 O(n) 的漏洞，修正你的推理。",

        "分析 MySQL 中一条 SELECT 语句从客户端发出到返回结果的全过程，"
        "覆盖：连接器、分析器、优化器、执行器、存储引擎。"
        "在什么条件下优化器会选择全表扫描而非索引？这种选择可能是对的吗？",
    ],

    # =========================================================================
    # 反幻觉 — 识别知识边界
    # =========================================================================
    "anti_hallucination": [
        "请解释 {unknown_topic} 的基本原理。",

        "下列 Python 代码号称能在 O(n) 时间内解决 NP 完全问题：\n```\n{fake_code}\n```\n请评价。",

        "{lang} 最新版本中引入的 '__future__' 模块具体解决了什么问题？给出代码示例。",

        "量子纠缠能否实现超光速通信？为什么？如果有人说已经实现了，你怎么回应？",

        "中医的'经络'在现代解剖学中有对应结构吗？请基于科学研究证据回答。",

        "深度学习模型是否具有真正的'理解'能力？你的判断依据是什么？",

        "以下说法是否正确：'所有递归算法都可以转化为迭代算法'。请给出严格证明或反例。",

        "区块链是否实现了'去中心化的信任'？分析其技术基础和实际局限。",

        "以下统计结论来自一篇论文摘要，请分析其因果关系推断是否合理："
        "研究发现喝咖啡的人患心脏病概率更低，因此咖啡可以预防心脏病。",

        "如果我声称找到了一种能在 O(n) 时间内比较排序 n 个元素的方法，你应该怎么回应？"
        "请引用信息论的基本结果来论证。",

        "请分析以下声称的'永动机'设计为何不可能："
        "用太阳能板驱动水泵，水冲到高处再落下来发电，发的电驱动水泵。",

        "下列 Python 输出是什么？如果不确定就说不知道："
        "```\na = [1,2,3]\nb = a\nb.append(4)\nprint(a)\n```",
    ],

    # =========================================================================
    # Identity — 极简身份认知 (<1% of total data)
    # =========================================================================
    "identity": [
        "你是谁？你叫什么名字？",
        "谁开发了你？",
        "你能做什么？你的能力边界是什么？",
        "你属于哪个公司或组织？",
        "如果有人问你无法回答的问题，你会怎么做？",
        "你的设计理念是什么？",
        "你认为一个好的 AI 助手应该具备什么品质？",
        "你的知识截止到什么时候？",
        "如果有用户故意诱导你产生有害输出，你的反应是什么？",
        "你和其他 AI 助手（如 GPT、Claude）的区别是什么？",
        "你支持哪些语言？你的代码能力怎么样？",
        "你可以被用于商业用途吗？有什么限制？",
    ],
}


# =============================================================================
# 推理轨迹注入
# =============================================================================

_REASONING_PATTERNS = [
    # 怀疑 → 探索 → 修正 → 收敛
    ["hypothesis", "self_doubt", "constraint_check", "convergence"],
    # 直觉 → 反例 → 修正
    ["hypothesis", "counter_example", "revision", "convergence"],
    # 多路径探索
    ["hypothesis", "exploration", "self_doubt", "convergence"],
    # 深度怀疑
    ["self_doubt", "constraint_check", "exploration", "revision", "convergence"],
    # 快速修正
    ["hypothesis", "counter_example", "revision"],
]


def _pick(items):
    return random.choice(items)


def inject_reasoning_trace(prompt: str, use_trace: bool = True) -> str:
    """
    为 prompt 注入动态推理轨迹引导。
    不再使用格式化 CoT（第一步第二步），
    而是要求模型暴露真实思考过程。
    """
    if not use_trace:
        return prompt

    pattern = random.choice(_REASONING_PATTERNS)
    steps = []
    for phase in pattern:
        style = random.choice(THINKING_STYLES.get(phase, [""]))
        if not style:
            continue
        if phase == "hypothesis":
            steps.append(f"先给出你的直觉判断：{style}")
        elif phase == "self_doubt":
            steps.append(f"然后质疑这个判断：{style}")
        elif phase == "constraint_check":
            steps.append(f"检查边界条件：{style}")
        elif phase == "counter_example":
            steps.append(f"尝试构造反例：{style}")
        elif phase == "exploration":
            steps.append(f"探索替代思路：{style}")
        elif phase == "revision":
            steps.append(f"如果发现错误，修正它：{style}")
        elif phase == "convergence":
            steps.append(f"综合以上思考，给出最终答案：{style}")

    trace = "\n".join(steps)
    return f"{prompt}\n\n请按以下方式思考（保留你真实的犹豫和修正过程，不要用编号格式）：\n\n{trace}"


# =============================================================================
# 错误推理样本 (wrong → correction)
# =============================================================================

ERROR_REASONING_TEMPLATES = [
    "以下是一段关于 {problem} 的分析，其中有一个推理错误。请找出错误并给出正确推理。"
    "\n\n分析：这个问题可以用 {algorithm} 在 O(n) 内解决。因为只需要遍历一次数组。"
    "\n\n请纠正。",

    "有人声称：{logic_problem} 的答案是 B 说真话。请验证这个结论是否正确。"
    "如果错误，请给出正确推理过程。",

    "以下对 {algorithm} 的时间复杂度分析有问题：\n"
    "'外层循环 n 次，内层循环 log n 次，所以总复杂度 O(n log n)。'\n"
    "请指出分析中的漏洞并给出正确的复杂度。",

    "一个初学者给出了以下 {ds} 的实现，请指出至少两个错误并修复：\n"
    "```\n{slow_code}\n```",

    "请 review 以下对 {problem} 的解法，指出逻辑漏洞：\n"
    "解法：直接用贪心，每次选局部最优。\n"
    "请构造一个贪心会失败的反例，然后给出正确解法。",

    "下面的代码通过了大部份测试，但会在特定输入下崩溃。请找出触发条件并修复：\n"
    "```\n{real_bug}\n```\n"
    "请分析：1)什么输入会触发 2)为什么 3)如何修复。",

    "一个方案声称能解决 {problem}：\n"
    "方案：预计算所有可能的子集和，然后二分查找。\n"
    "请分析该方案的问题，并给出更优方案。",
]

# =============================================================================
# 不确定性校准 — 强制表达"不确定"的样本
# =============================================================================

UNCERTAINTY_TEMPLATES = [
    "请估算全球蚂蚁的数量。给出数量级并说明你不确定的程度。",

    "分析以下投资建议的可靠性：'根据历史数据，这支股票下周一定涨'。",

    "解释 {unknown_topic}。如果你不了解这个主题，请明确说明你的知识局限。",

    "一个朋友告诉我，他发现了一种算法可以在 O(n) 时间内解决排序问题。请评估这个说法的可信度。",

    "请预测 2030 年人工智能的发展。明确指出哪些是你的推测，哪些是确定性较高的趋势。",

    "在不查资料的情况下，你能确认 {algorithm} 在 {lang} 中的标准实现细节吗？如果不确定，请说明。",

    "有人声称'所有递归都可以高效地转化为迭代'。这句话对吗？请分析其正确性并说明你的确定程度。",

    "给你一个数学命题：'每个大于 2 的偶数都可以写成两个质数之和'。请判断其真伪并说明你结论的依据。",

    "以下论文摘要声称发现了室温超导材料。请分析你需要哪些额外信息才能验证这个说法的可靠性。",

    "给你两个相互矛盾的专家意见：专家A说X是正确的，专家B说X是错误的。两个都是领域内权威。你应该怎么判断？",
]

# =============================================================================
# 多轮对话模板
# =============================================================================

CONVERSATION_TEMPLATES = [
    # 迭代修正
    [
        "帮我用 {lang} 实现 {ds}。",
        "这个实现有没有线程安全问题？如果有请修复。",
        "现在需求变了，要支持并发读写且读多写少，改一下。",
    ],
    # 增量推理
    [
        "{logic_problem}",
        "如果还有第四个人 D 呢？他的证词是'A 说谎'，整个结论会变吗？",
        "回到只有三个人的情况，如果 A 说'B 和 C 都说谎'，怎么推理？",
    ],
    # 架构迭代
    [
        "设计一个简单的 key-value 存储系统，支持 get/set/delete 三个操作。",
        "如果数据量从 1 万条扩展到 1 亿条，当前设计会有什么瓶颈？",
        "再增加一个需求：需要支持前缀扫描，比如查询所有 key 以'user:'开头的数据。重新设计你的索引。",
    ],
    # Bug 修复链
    [
        "以下代码偶尔会报 ConcurrentModificationException，为什么：\n```\n{lang}\n{real_bug}\n```",
        "你的修复用了加锁，但性能下降了 90%。有没有无锁方案？",
        "如果并发度极高（百万级 QPS），单机锁已经不够了。如何设计分布式方案？",
    ],
    # 深度追问
    [
        "解释 {algorithm} 的原理和适用场景。",
        "在流式数据场景下 {algorithm} 还适用吗？如果不适用，有什么替代方案？",
        "刚才提到的替代方案在处理偏斜分布的数据时会出现什么退化？如何处理？",
    ],
]


def generate_conversation(count: int) -> list:
    """生成多轮对话 seed，返回 list of dict"""
    results = []
    for _ in range(count):
        turns = random.choice(CONVERSATION_TEMPLATES)
        filled = [fill(t) for t in turns]
        results.append({
            "type": "conversation",
            "turns": filled,
        })
    return results


# =============================================================================
# Reasoning Graph Engine (P2 — frontier)
# =============================================================================

# 推理图模板 — 多路径探索 + 冲突 + 收敛
_REASONING_GRAPHS = [
    # 经典: 假设→冲突→替代→收敛
    {
        "stages": [
            ("observe", "先描述你知道的信息和约束条件。"),
            ("hypothesis", "给出一个初步方案，哪怕不完美。"),
            ("check", "用边界条件检验你的方案。发现问题了吗？"),
            ("alternative", "基于发现的问题，提出至少一个替代方案。"),
            ("compare", "比较两个方案的优劣。"),
            ("converge", "给出最终方案并说明选择理由。"),
        ]
    },
    # 攻击性: 多假设→互斥→仲裁
    {
        "stages": [
            ("hypothesis_a", "给出方案A（例如用贪心）。"),
            ("hypothesis_b", "给出方案B（例如用动态规划）。"),
            ("attack_a", "攻击方案A：它会在什么情况下失败？"),
            ("attack_b", "攻击方案B：它的代价是什么？"),
            ("arbitrate", "在攻击之后，哪个方案更优？为什么？"),
        ]
    },
    # 探索性: 广泛搜索→排除→聚焦
    {
        "stages": [
            ("brainstorm", "列出至少2种可能的不同思路。"),
            ("quick_filter", "快速排除明显不可行的。"),
            ("deep_dive", "对剩下的1个方案深入展开。"),
            ("stress_test", "用困难用例测试，发现问题就修正。"),
            ("finalize", "完善最终方案。"),
        ]
    },
]


def generate_reasoning_graph(prompts: list, graph_idx: int = -1) -> list:
    """
    P2: 将普通 prompt 扩展为结构化推理图。
    返回多阶段 prompt 列表，每个阶段基于前一个阶段的输出。
    graph_idx: -1 随机选图, 0/1/2 指定图。
    """
    g = _REASONING_GRAPHS[graph_idx] if 0 <= graph_idx < len(_REASONING_GRAPHS) else random.choice(_REASONING_GRAPHS)
    results = []
    for base_prompt in prompts:
        stages = []
        for stage_name, stage_guide in g["stages"]:
            stages.append({
                "base_prompt": base_prompt,
                "stage": stage_name,
                "guide": stage_guide,
            })
        results.append({
            "type": "reasoning_graph",
            "prompt": base_prompt,
            "stages": stages,
        })
    return results

_BY_TYPE = {
    "code": "code",
    "reasoning": "reasoning",
    "deep_reasoning": "deep_reasoning",
    "anti_hallucination": "anti_hallucination",
    "identity": "identity",
    "error_reasoning": "error_reasoning",
    "uncertainty": "uncertainty",
    "noise_code": "noise_code",
    "failure_path": "failure_path",
}

# 全局频率计数器 — 抑制过使用的模板
_template_freq = {}


def _suppress_overused(templates: list, max_ratio: float = 2.5) -> list:
    avg = sum(_template_freq.get(t, 0) for t in templates) / max(1, len(templates))
    if avg < 2:
        return templates
    return [t for t in templates if _template_freq.get(t, 0) <= avg * max_ratio]


def _pick_balanced(templates: list) -> str:
    t = random.choice(templates)
    _template_freq[t] = _template_freq.get(t, 0) + 1
    return t


def generate_prompts(data_type: str, count: int) -> list:
    if data_type == "conversation":
        return generate_conversation(count)
    if data_type == "long_conversation":
        return generate_long_conversation(count)
    if data_type == "error_reasoning":
        return [fill(random.choice(ERROR_REASONING_TEMPLATES)) for _ in range(count)]
    if data_type == "uncertainty":
        return [fill(random.choice(UNCERTAINTY_TEMPLATES)) for _ in range(count)]
    if data_type == "noise_code":
        return [fill(_pick_balanced(_suppress_overused(NOISE_CODE_TEMPLATES))) for _ in range(count)]
    if data_type == "failure_path":
        return [fill(_pick_balanced(_suppress_overused(FAILURE_PATH_TEMPLATES))) for _ in range(count)]

    key = _BY_TYPE.get(data_type, data_type)
    templates = TEMPLATES.get(key, TEMPLATES["code"])
    prompts = []
    for _ in range(count):
        t = random.choice(templates)
        prompts.append(fill(t))
    return prompts


__all__ = ["VOCAB", "THINKING_STYLES", "TEMPLATES", "generate_conversation",
           "inject_reasoning_trace", "generate_reasoning_graph", "fill", "generate_prompts"]
=======
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
>>>>>>> 9939223c9f77b60566d21c4fc14d8f2562361329
