"""
Data Synthesizer — 数据合成与增强
支持：Self-Instruct、Evol-Instruct、模板生成、回译增强
"""
import random
import json
from typing import List, Dict, Any, Optional, Callable


class DataSynthesizer:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.templates = {
            "qa": [
                {"prompt": "请解释什么是{concept}", "response": "{concept}是指{definition}"},
                {"prompt": "{concept}的主要应用场景有哪些？", "response": "{concept}的主要应用场景包括{scenes}"},
                {"prompt": "对比{concept_a}和{concept_b}的异同", "response": "{concept_a}和{concept_b}的相同点是{common}，不同点是{diff}"},
            ],
            "writing": [
                {"prompt": "请写一篇关于{topic}的短文", "response": "关于{topic}的短文：\n{content}"},
                {"prompt": "为{topic}写一个提纲", "response": "## {topic}提纲\n\n{outline}"},
            ],
            "coding": [
                {"prompt": "用{lang}写一个{task}函数", "response": "```{lang}\n{code}\n```"},
                {"prompt": "解释以下{lang}代码的功能：{code}", "response": "这段代码的功能是：{explanation}"},
            ],
        }

        self.vocab = {
            "concept": ["深度学习", "Transformer", "注意力机制", "预训练", "微调", "知识蒸馏", "强化学习", "神经网络"],
            "definition": ["一种基于多层神经网络的机器学习方法", "一种处理序列数据的神经网络架构",
                           "一种让模型关注输入中重要部分的技术", "在大规模数据上训练通用模型的过程"],
            "scenes": ["自然语言处理", "计算机视觉", "语音识别", "推荐系统", "自动驾驶"],
            "concept_a": ["CNN", "RNN", "GPT", "BERT", "LSTM"],
            "concept_b": ["Transformer", "GNN", "T5", "RoBERTa", "GRU"],
            "common": ["都使用神经网络", "都需要大量训练数据", "都可以用于NLP任务"],
            "diff": ["架构设计不同", "训练方式不同", "适用场景不同"],
            "topic": ["人工智能的未来", "深度学习入门", "Python编程技巧", "数据分析方法"],
            "content": ["随着技术的不断发展...", "在当今数字化时代..."],
            "outline": ["1. 引言\n2. 背景\n3. 方法\n4. 结论"],
            "lang": ["Python", "JavaScript", "Java", "C++", "Go"],
            "task": ["排序", "搜索", "数据处理", "文件读写", "网络请求"],
            "code": ["def hello(): print('hello')", "import os\nprint(os.getcwd())"],
            "explanation": ["定义了一个函数", "导入os模块并打印当前工作目录"],
        }

    def generate_from_template(self, category: str = "qa", count: int = 100) -> List[Dict]:
        results = []
        templates = self.templates.get(category, self.templates["qa"])
        for _ in range(count):
            tmpl = random.choice(templates)
            filled = {}
            for key, values in self.vocab.items():
                filled[key] = random.choice(values)
            try:
                prompt = tmpl["prompt"].format(**filled)
                response = tmpl["response"].format(**filled)
                results.append({"prompt": prompt, "response": response})
            except KeyError:
                continue
        print(f"Generated {len(results)} {category} samples from templates")
        return results

    def generate_self_instruct(self, seed_prompts: List[str], generator_fn: Callable, count: int = 100) -> List[Dict]:
        results = []
        for prompt in seed_prompts[:count]:
            try:
                response = generator_fn(prompt)
                results.append({"prompt": prompt, "response": response})
            except Exception as e:
                print(f"Self-instruct error for '{prompt[:30]}': {e}")
        print(f"Self-instruct generated {len(results)} samples")
        return results

    def evol_instruct(self, items: List[Dict], generator_fn: Callable, rounds: int = 2) -> List[Dict]:
        strategies = [
            self._add_constraints,
            self._deepen,
            self._concretize,
            self._increase_reasoning,
        ]
        evolved = list(items)
        for r in range(rounds):
            new_items = []
            for item in evolved:
                prompt = item.get("prompt", item.get("text", ""))
                if not prompt:
                    continue
                strategy = random.choice(strategies)
                evolved_prompt = strategy(prompt)
                try:
                    response = generator_fn(evolved_prompt)
                    new_items.append({"prompt": evolved_prompt, "response": response, "evolved_from": prompt[:50]})
                except:
                    new_items.append(item)
            evolved.extend(new_items)
            print(f"Evol-Instruct round {r+1}: added {len(new_items)} samples")
        return evolved

    def _add_constraints(self, prompt: str) -> str:
        constraints = [
            "请用100字以内回答。",
            "请分三点回答。",
            "请给出具体的例子。",
            "请用通俗的语言解释。",
            "请从正反两个方面分析。",
        ]
        return prompt + " " + random.choice(constraints)

    def _deepen(self, prompt: str) -> str:
        if "为什么" not in prompt and "如何" not in prompt:
            return "为什么" + prompt + "？请深入分析原因。"
        return prompt + "请从更深的层次进行分析。"

    def _concretize(self, prompt: str) -> str:
        return prompt + "请给出具体的案例或数据支持。"

    def _increase_reasoning(self, prompt: str) -> str:
        return prompt + "请先逐步推理，再给出最终答案。"

    def augment_text(self, items: List[str], augment_fn: Optional[Callable] = None, times: int = 2) -> List[str]:
        if augment_fn:
            augmented = []
            for item in items:
                augmented.append(item)
                for _ in range(times):
                    augmented.append(augment_fn(item))
            return augmented
        return items

    def generate_dpo_pairs(self, items: List[Dict], weak_model_fn: Callable) -> List[Dict]:
        dpo_data = []
        for item in items:
            prompt = item.get("prompt", "")
            chosen = item.get("response", "")
            try:
                rejected = weak_model_fn(prompt)
                dpo_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            except:
                pass
        print(f"Generated {len(dpo_data)} DPO pairs")
        return dpo_data
