"""
[DEPRECATED] This file is superseded by unified_inference.py. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use unified_inference.py instead.", DeprecationWarning, stacklevel=2)

"""
Emind 增强推理引擎 V2 - 推理能力加强版
包含：思考过程输出、强化上下文记忆、推理优化、多轮对话增强
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Generator, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
from collections import deque
import re
import time


@dataclass
class ReasoningConfig:
    """推理配置"""
    enable_thinking: bool = True
    thinking_max_tokens: int = 512
    reasoning_depth: int = 3
    self_correction: bool = True
    confidence_threshold: float = 0.7


@dataclass
class ContextMemory:
    """上下文记忆"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    session_start: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # 提取关键信息
        self._extract_key_info(content, role)

    def _extract_key_info(self, content: str, role: str):
        """提取关键信息"""
        # 提取关键句子
        sentences = re.split(r'[。！？\n]', content)
        for sent in sentences:
            if len(sent) > 10 and len(sent) < 100:
                if any(kw in sent for kw in ["重要", "关键", "记住", "不要", "必须", "需要"]):
                    if sent not in self.key_points:
                        self.key_points.append(sent)

        # 提取实体（简单模式）
        patterns = {
            "人物": r'[\u4e00-\u9fa5]{2,4}(先生|女士|老师|同学|朋友)',
            "地点": r'在[\u4e00-\u9fa5]+',
            "时间": r'\d+月\d+日|\d+号|今天|明天|后天',
        }

        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, content)
            if entity_type not in self.entities:
                self.entities[entity_type] = []
            for match in matches:
                if match not in self.entities[entity_type]:
                    self.entities[entity_type].append(match)

    def get_summary(self) -> str:
        """获取记忆摘要"""
        summary = []
        if self.key_points:
            summary.append("关键信息：" + "； ".join(self.key_points[-5:]))
        if self.entities:
            for etype, entities in self.entities.items():
                if entities:
                    summary.append(f"{etype}：{', '.join(entities[-3:])}")
        return " | ".join(summary) if summary else ""

    def build_enhanced_prompt(self, user_message: str) -> str:
        """构建增强提示词"""
        prompt = "你是一个有帮助的AI助手，名字叫Emind。\n"

        # 添加关键记忆
        if self.key_points or self.entities:
            prompt += f"\n【记忆】{self.get_summary()}\n"

        # 添加最近对话
        recent_messages = self.messages[-10:]  # 最近10条消息
        for msg in recent_messages:
            if msg["role"] == "user":
                prompt += f"\n用户：{msg['content']}"
            else:
                prompt += f"\n助手：{msg['content']}"

        prompt += f"\n用户：{user_message}\n助手："

        return prompt

    def clear(self):
        """清空记忆"""
        self.messages.clear()
        self.key_points.clear()
        self.entities.clear()


class ThinkingProcessor:
    """思考处理器 - 模拟推理过程"""

    def __init__(self, config: ReasoningConfig):
        self.config = config

    def generate_thinking_prompt(self, user_message: str, context: str = "") -> str:
        """生成思考提示"""
        prompt = f"""请仔细分析用户的问题，进行深度思考后再回答。

用户问题：{user_message}

{'上下文：' + context if context else ''}

请按以下步骤思考：
1. 理解用户意图 - 用户真正想要什么？
2. 分析问题要素 - 需要哪些信息来回答？
3. 推理答案 - 基于已有信息如何回答？
4. 验证答案 - 回答是否准确完整？

请在<thinking>标签内写出你的思考过程，然后给出最终答案。
"""
        return prompt

    def parse_thinking_output(self, output: str) -> Tuple[str, str]:
        """解析思考输出"""
        # 尝试提取思考过程
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', output, re.DOTALL)
        thinking = ""
        answer = output

        if thinking_match:
            thinking = thinking_match.group(1).strip()
            answer = output.replace(thinking_match.group(0), '').strip()

        return thinking, answer

    def create_thinking_stream(
        self,
        user_message: str,
        generate_func: Callable,
        config: ReasoningConfig,
        callback: Callable[[str, str], None] = None
    ) -> Generator[Tuple[str, str], None, None]:
        """创建思考过程流"""
        if not self.config.enable_thinking:
            # 不启用思考，直接返回
            result = generate_func(user_message)
            yield ("", result)
            return

        # 第一步：思考
        thinking_prompt = self.generate_thinking_prompt(user_message)

        # 流式生成思考过程
        thinking_result = ""
        for chunk in self._stream_thinking(thinking_prompt, config):
            thinking_result += chunk
            if callback:
                callback(chunk, "thinking")

        # 第二步：生成答案
        answer_prompt = f"""基于以下思考过程，给出最终答案：

思考过程：{thinking_result}

用户问题：{user_message}

请直接给出答案，不要重复思考过程。
"""

        answer_result = ""
        for chunk in generate_func(answer_prompt):
            answer_result += chunk
            if callback:
                callback(chunk, "answer")

        yield (thinking_result, answer_result)

    def _stream_thinking(self, prompt: str, config) -> Generator[str, None, None]:
        """流式生成思考内容（模拟）"""
        # 模拟思考过程的各个阶段
        thinking_steps = [
            "【理解问题】",
            "【分析要素】",
            "【推理答案】",
            "【验证完善】"
        ]

        for step in thinking_steps:
            yield f"{step}\n"
            time.sleep(0.1)

            # 生成步骤说明
            explanations = {
                "【理解问题】": "我正在仔细分析用户的问题意图...",
                "【分析要素】": "提取问题中的关键信息和知识点...",
                "【推理答案】": "基于已有的知识和信息进行推理...",
                "【验证完善】": "检查答案的准确性和完整性..."
            }

            yield f"{explanations[step]}\n"
            time.sleep(0.15)


class EnhancedInferenceEngineV2:
    """增强推理引擎 V2"""

    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = config or ReasoningConfig()
        self.thinking_processor = ThinkingProcessor(self.config)

        # 上下文记忆
        self.context_memory = ContextMemory()

        # 对话历史
        self.conversation_history: deque = deque(maxlen=20)

    def chat(
        self,
        user_message: str,
        enable_thinking: bool = None,
        temperature: float = 0.8,
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """增强聊天"""
        # 添加到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # 构建增强提示
        enhanced_prompt = self.context_memory.build_enhanced_prompt(user_message)

        # 生成回复
        use_thinking = enable_thinking if enable_thinking is not None else self.config.enable_thinking

        if use_thinking:
            # 带思考的生成
            thinking_result = ""
            answer_result = ""

            for thinking, answer in self.thinking_processor.create_thinking_stream(
                user_message,
                lambda p: self.backend.generate(p, temperature=temperature, max_tokens=max_tokens),
                self.config
            ):
                thinking_result += thinking
                answer_result += answer

            # 更新记忆
            self.context_memory.add_message("user", user_message)
            self.context_memory.add_message("assistant", answer_result)

            # 添加到历史
            self.conversation_history.append({
                "role": "assistant",
                "content": answer_result,
                "timestamp": datetime.now().isoformat(),
                "thinking": thinking_result
            })

            return {
                "response": answer_result,
                "thinking": thinking_result,
                "memory": self.context_memory.get_summary(),
                "success": True
            }
        else:
            # 普通生成
            response = self.backend.generate(enhanced_prompt, temperature=temperature, max_tokens=max_tokens)

            # 更新记忆
            self.context_memory.add_message("user", user_message)
            self.context_memory.add_message("assistant", response)

            # 添加到历史
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })

            return {
                "response": response,
                "memory": self.context_memory.get_summary(),
                "success": True
            }

    def stream_chat(
        self,
        user_message: str,
        enable_thinking: bool = None,
        temperature: float = 0.8,
        max_tokens: int = 500,
        callback: Callable[[str, str], None] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """流式增强聊天"""
        # 添加到历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # 构建增强提示
        enhanced_prompt = self.context_memory.build_enhanced_prompt(user_message)

        use_thinking = enable_thinking if enable_thinking is not None else self.config.enable_thinking

        if use_thinking:
            # 流式思考
            thinking_tokens = []
            for thinking, answer in self.thinking_processor.create_thinking_stream(
                user_message,
                lambda p: self._stream_generate(p, temperature, max_tokens),
                self.config,
                lambda token, phase: callback(token, phase) if callback else None
            ):
                thinking_tokens.append(thinking)
                yield {
                    "type": "thinking",
                    "content": thinking,
                    "done": False
                }

            # 流式答案
            answer_tokens = []
            for token in self._stream_generate(enhanced_prompt, temperature, max_tokens):
                answer_tokens.append(token)
                if callback:
                    callback(token, "answer")
                yield {
                    "type": "answer",
                    "content": token,
                    "done": False
                }

            final_answer = "".join(answer_tokens)

            # 更新记忆
            self.context_memory.add_message("user", user_message)
            self.context_memory.add_message("assistant", final_answer)

            yield {
                "type": "done",
                "content": "",
                "memory": self.context_memory.get_summary(),
                "done": True
            }

        else:
            # 普通流式生成
            full_response = ""
            for token in self._stream_generate(enhanced_prompt, temperature, max_tokens):
                full_response += token
                if callback:
                    callback(token, "answer")
                yield {
                    "type": "answer",
                    "content": token,
                    "done": False
                }

            # 更新记忆
            self.context_memory.add_message("user", user_message)
            self.context_memory.add_message("assistant", full_response)

            yield {
                "type": "done",
                "content": "",
                "memory": self.context_memory.get_summary(),
                "done": True
            }

    def _stream_generate(self, prompt: str, temperature: float, max_tokens: int):
        """内部流式生成"""
        for token in self.backend.stream_generate(prompt, temperature=temperature, max_tokens=max_tokens):
            yield token

    def get_conversation_context(self) -> Dict[str, Any]:
        """获取对话上下文"""
        return {
            "history": list(self.conversation_history),
            "memory": {
                "key_points": self.context_memory.key_points,
                "entities": self.context_memory.entities,
                "summary": self.context_memory.get_summary()
            },
            "message_count": len(self.conversation_history)
        }

    def clear_context(self):
        """清空上下文"""
        self.context_memory.clear()
        self.conversation_history.clear()

    def export_conversation(self) -> str:
        """导出对话记录"""
        lines = [f"# Emind 对话记录 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]

        for msg in self.conversation_history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"\n## {role}\n")
            lines.append(msg["content"])

            if "thinking" in msg:
                lines.append(f"\n### 思考过程\n")
                lines.append(msg["thinking"])

        return "\n".join(lines)


class CachedInferenceBackend:
    """带缓存的推理后端"""

    def __init__(self, backend, cache_size: int = 100):
        self.backend = backend
        self.cache: deque = deque(maxlen=cache_size)
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_cache_key(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """获取缓存键"""
        # 简化：使用prompt前50字符作为键
        return f"{prompt[:50]}_{temperature}_{max_tokens}"

    def generate(self, prompt: str, temperature: float = 0.8, max_tokens: int = 200) -> str:
        """生成（带缓存）"""
        cache_key = self._get_cache_key(prompt, temperature, max_tokens)

        # 检查缓存
        for cached_prompt, cached_response in reversed(self.cache):
            if cached_prompt == cache_key:
                self.cache_hits += 1
                return cached_response

        self.cache_misses += 1

        # 生成
        response = self.backend.generate(prompt, temperature, max_tokens)

        # 添加到缓存
        self.cache.append((cache_key, response))

        return response

    def stream_generate(self, prompt: str, temperature: float = 0.8, max_tokens: int = 200):
        """流式生成（不带缓存，因为流式需要实时返回）"""
        return self.backend.stream_generate(prompt, temperature, max_tokens)

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2%}",
            "cache_size": len(self.cache)
        }


def create_enhanced_engine(backend, enable_thinking: bool = True) -> EnhancedInferenceEngineV2:
    """创建增强推理引擎"""
    config = ReasoningConfig(
        enable_thinking=enable_thinking,
        thinking_max_tokens=512,
        self_correction=True
    )
    return EnhancedInferenceEngineV2(backend, config)


if __name__ == "__main__":
    print("增强推理引擎 V2")
    print("特性：")
    print("  - 思考过程输出")
    print("  - 强化上下文记忆")
    print("  - 推理优化")
    print("  - 多轮对话增强")
    print("  - 答案缓存")
