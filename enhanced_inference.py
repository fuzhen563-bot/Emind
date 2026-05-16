"""
[DEPRECATED] This file is superseded by unified_inference.py. Use that instead.
"""
import warnings
warnings.warn("This module is deprecated. Use unified_inference.py instead.", DeprecationWarning, stacklevel=2)

"""
Emind 增强推理模块 - 提升交互能力
包含：流式输出、对话历史管理、多种生成策略、工具集成
"""

import torch
import torch.nn.functional as F
import json
import os
import re
from typing import List, Dict, Optional, Callable, Generator, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue

from model import create_model
from tokenizer import SimpleTokenizer


@dataclass
class GenerationConfig:
    """生成配置"""
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    length_penalty: float = 1.0


@dataclass
class ConversationMessage:
    """对话消息"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ConversationManager:
    """对话管理器"""

    def __init__(self, max_history: int = 10, max_tokens: int = 2000):
        self.max_history = max_history
        self.max_tokens = max_tokens
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        self.current_session_id: Optional[str] = None

    def create_session(self, session_id: str = None) -> str:
        """创建新对话会话"""
        if session_id is None:
            session_id = f"session_{len(self.conversations)}_{int(datetime.now().timestamp())}"

        self.conversations[session_id] = []
        self.current_session_id = session_id
        return session_id

    def add_message(self, role: str, content: str, session_id: str = None) -> None:
        """添加消息"""
        if session_id is None:
            session_id = self.current_session_id

        if session_id not in self.conversations:
            self.create_session(session_id)

        message = ConversationMessage(role=role, content=content)
        self.conversations[session_id].append(message)

        # 裁剪历史
        self._trim_history(session_id)

    def get_history(self, session_id: str = None) -> List[ConversationMessage]:
        """获取对话历史"""
        if session_id is None:
            session_id = self.current_session_id

        return self.conversations.get(session_id, [])

    def clear_history(self, session_id: str = None) -> None:
        """清除对话历史"""
        if session_id is None:
            session_id = self.current_session_id

        if session_id in self.conversations:
            self.conversations[session_id] = []

    def _trim_history(self, session_id: str) -> None:
        """裁剪历史记录"""
        if session_id not in self.conversations:
            return

        # 限制消息数量
        if len(self.conversations[session_id]) > self.max_history:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history:]

    def build_prompt(
        self,
        user_message: str,
        system_prompt: str = None,
        session_id: str = None
    ) -> str:
        """构建提示词"""
        if session_id is None:
            session_id = self.current_session_id

        # 系统提示
        if system_prompt is None:
            system_prompt = "你是一个有帮助的AI助手名叫Emind。请用清晰、准确的中文回答用户的问题。"

        prompt = f"系统: {system_prompt}\n"

        # 对话历史
        history = self.get_history(session_id)
        for msg in history:
            if msg.role == "user":
                prompt += f"用户: {msg.content}\n"
            else:
                prompt += f"助手: {msg.content}\n"

        # 当前消息
        prompt += f"用户: {user_message}\n助手:"

        return prompt

    def get_conversation_summary(self, session_id: str = None) -> Dict[str, Any]:
        """获取对话摘要"""
        if session_id is None:
            session_id = self.current_session_id

        history = self.get_history(session_id)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "first_message": history[0].content if history else None,
            "last_message": history[-1].content if history else None,
            "timestamp": datetime.now().isoformat()
        }


class EnhancedInferenceEngine:
    """增强推理引擎"""

    def __init__(
        self,
        model,
        tokenizer: SimpleTokenizer,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.eval()

        # 对话管理
        self.conversation_manager = ConversationManager()

        # 预设回复
        self.preset_responses = self._load_preset_responses()

    def _load_preset_responses(self) -> Dict[str, str]:
        """加载预设回复"""
        return {
            "greeting": "你好！我是 Emind AI 智能助手，很高兴为你服务！有什么我可以帮助你的吗？",
            "thanks": "不客气！很高兴能帮到你。如果还有其他问题，请随时问我。",
            "goodbye": "再见！希望我们的对话对你有帮助。下次有需要随时找我！",
            "sorry": "抱歉，我不太明白你的意思。你可以尝试换一种表达方式，或者问我一些其他问题。",
            "capabilities": "我可以帮助你：\n1. 回答各类知识问题\n2. 协助写作和文案创作\n3. 编程问题和代码调试\n4. 翻译和多语言交流\n5. 学习方法和技巧指导\n6. 日常生活建议",
        }

    def _detect_intent(self, text: str) -> Optional[str]:
        """检测用户意图"""
        text_lower = text.lower()

        # 问候
        greetings = ["你好", "您好", "hello", "hi", "hey", "早上好", "晚上好"]
        if any(g in text_lower for g in greetings):
            return "greeting"

        # 感谢
        thanks = ["谢谢", "感谢", "感谢你", "thanks", "thank you"]
        if any(t in text_lower for t in thanks):
            return "thanks"

        # 再见
        goodbye = ["再见", "拜拜", "bye", "goodbye", "再见"]
        if any(g in text_lower for g in goodbye):
            return "goodbye"

        # 询问能力
        capabilities = ["你能做什么", "有什么功能", "可以帮助我什么", "Capabilities"]
        if any(c in text_lower for c in capabilities):
            return "capabilities"

        return None

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: GenerationConfig = None,
        session_id: str = None
    ) -> str:
        """生成回复"""
        if config is None:
            config = GenerationConfig()

        # 编码
        encoded = self.tokenizer.encode(prompt)

        # 截断输入
        max_input_len = 512
        if len(encoded) > max_input_len:
            encoded = encoded[-max_input_len:]

        input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)

        # 生成
        generated = []
        eos_token_id = self.tokenizer.stoi.get(self.tokenizer.eos_token, 3)

        for step in range(config.max_new_tokens):
            # 前向传播
            logits = self.model(input_ids)[1][:, -1, :]

            # 应用温度
            if config.temperature > 0:
                logits = logits / config.temperature

            # Repetition penalty
            if config.repetition_penalty > 1.0:
                for token_id in set(generated):
                    logits[0, token_id] /= config.repetition_penalty

            # Top-k filtering
            if config.top_k > 0:
                indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # 采样
            if config.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # 检查结束
            if next_token.item() == eos_token_id:
                break

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 限制生成长度
            if len(generated) >= config.max_new_tokens:
                break

        # 解码
        result = self.tokenizer.decode(generated)
        return result

    def stream_generate(
        self,
        prompt: str,
        config: GenerationConfig = None,
        callback: Callable[[str], None] = None,
        session_id: str = None
    ) -> Generator[str, None, None]:
        """流式生成"""
        if config is None:
            config = GenerationConfig()

        # 编码
        encoded = self.tokenizer.encode(prompt)
        max_input_len = 512
        if len(encoded) > max_input_len:
            encoded = encoded[-max_input_len:]

        input_ids = torch.tensor([encoded], dtype=torch.long).to(self.device)

        # 生成
        generated = []
        eos_token_id = self.tokenizer.stoi.get(self.tokenizer.eos_token, 3)
        special_tokens = {self.tokenizer.pad_token, self.tokenizer.bos_token,
                         self.tokenizer.eos_token, self.tokenizer.unk_token}

        for step in range(config.max_new_tokens):
            with torch.no_grad():
                logits = self.model(input_ids)[1][:, -1, :]

            if config.temperature > 0:
                logits = logits / config.temperature

            if config.repetition_penalty > 1.0:
                for token_id in set(generated):
                    if token_id < logits.shape[-1]:
                        logits[0, token_id] /= config.repetition_penalty

            if config.top_k > 0:
                indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > config.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            if config.do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()

            if token_id == eos_token_id:
                break

            generated.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # 解码并yield
            decoded_char = self.tokenizer.itos.get(token_id, '')
            if decoded_char and decoded_char not in special_tokens:
                if callback:
                    callback(decoded_char)
                yield decoded_char

            if len(generated) >= config.max_new_tokens:
                break

    def chat(
        self,
        user_message: str,
        session_id: str = None,
        system_prompt: str = None,
        config: GenerationConfig = None,
        stream: bool = False,
        stream_callback: Callable[[str], None] = None
    ) -> Dict[str, Any]:
        """聊天接口"""
        # 检测意图
        intent = self._detect_intent(user_message)

        # 如果检测到预设意图，直接返回
        if intent and intent in self.preset_responses:
            # 如果不是流式模式，添加到历史
            if not stream:
                self.conversation_manager.add_message("user", user_message, session_id)
                self.conversation_manager.add_message("assistant",
                    self.preset_responses[intent], session_id)

            return {
                "intent": intent,
                "response": self.preset_responses[intent],
                "is_preset": True,
                "session_id": session_id or self.conversation_manager.current_session_id
            }

        # 构建提示
        prompt = self.conversation_manager.build_prompt(
            user_message, system_prompt, session_id
        )

        # 生成回复
        if stream:
            full_response = ""

            def callback(text):
                nonlocal full_response
                full_response += text
                if stream_callback:
                    stream_callback(text)

            # 流式生成
            list(self.stream_generate(prompt, config, callback, session_id))

            # 添加到历史
            self.conversation_manager.add_message("user", user_message, session_id)
            self.conversation_manager.add_message("assistant", full_response, session_id)

            return {
                "response": full_response,
                "is_preset": False,
                "session_id": session_id or self.conversation_manager.current_session_id
            }
        else:
            # 普通生成
            response = self.generate(prompt, config, session_id)

            # 添加到历史
            self.conversation_manager.add_message("user", user_message, session_id)
            self.conversation_manager.add_message("assistant", response, session_id)

            return {
                "response": response,
                "is_preset": False,
                "session_id": session_id or self.conversation_manager.current_session_id
            }

    def create_session(self, session_id: str = None) -> str:
        """创建新会话"""
        return self.conversation_manager.create_session(session_id)

    def clear_session(self, session_id: str = None) -> None:
        """清除会话"""
        self.conversation_manager.clear_history(session_id)


def load_inference_engine(
    model_path: str = "checkpoints/model.pt",
    tokenizer_path: str = "checkpoints/tokenizer.json",
    device: str = "cuda"
) -> EnhancedInferenceEngine:
    """加载推理引擎"""
    # 加载分词器
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('model_config', {})

    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 创建推理引擎
    engine = EnhancedInferenceEngine(model, tokenizer, device)

    print(f"推理引擎已加载")
    print(f"模型: {model_path}")
    print(f"设备: {device}")

    return engine


if __name__ == "__main__":
    print("增强推理模块测试")
    print("模块加载成功！")
