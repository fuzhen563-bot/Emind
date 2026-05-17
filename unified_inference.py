"""
Emind 统一推理后端 - 支持多种模型来源
包括：vLLM, Ollama, llama.cpp (LLM), HuggingFace Transformers, 本地模型, 亦API 云端
"""

import os
import json
from typing import Optional, Dict, Any, List, Generator, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import queue

# 亦API 默认配置 (优先从环境变量读取)
DEFAULT_API_KEY = os.environ.get("EMIND_API_KEY", "")
DEFAULT_API_BASE_URL = os.environ.get("EMIND_API_BASE_URL", "https://api.yiziyun.com")

# 尝试导入各后端库
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from vllm import LLM as VLLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# 尝试导入 vLLM 深度集成模块
try:
    from vllm_integration import (
        VLLMIntegratedEngine, VLLMConfig, create_vllm_engine,
        detect_vllm_capabilities, auto_configure_for_gpu,
        DynamicLoRAManager, VLLMHealthMonitor,
        SpeculativeDecodingConfig, LoRAConfig,
    )
    VLLM_INTEGRATION_AVAILABLE = True
except ImportError:
    VLLM_INTEGRATION_AVAILABLE = False

from tokenizer import EmindTokenizer as SimpleTokenizer


@dataclass
class BackendConfig:
    """后端配置"""
    backend_type: str = "ollama"  # ollama, llama_cpp, huggingface, local, cloud_api, vllm, vllm_server
    model_name: str = "llama2"
    model_path: Optional[str] = None
    base_url: str = "http://localhost:11434"
    device: str = "cuda"
    n_ctx: int = 2048
    n_threads: int = 4
    n_gpu_layers: int = 0
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 200
    api_key: Optional[str] = None

    # vLLM 特有
    vllm_enable_prefix_caching: bool = True
    vllm_enable_speculative: bool = False
    vllm_speculative_draft_model: Optional[str] = None
    vllm_num_speculative_tokens: int = 5
    vllm_enable_lora: bool = False
    vllm_lora_dir: Optional[str] = None
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.90
    vllm_dtype: str = "auto"
    vllm_quantization: Optional[str] = None
    vllm_enable_chunked_prefill: bool = True
    vllm_max_model_len: Optional[int] = None


class BaseBackend(ABC):
    """后端基类"""

    @abstractmethod
    def generate(self, prompt: str, config: BackendConfig = None, **kwargs) -> str:
        """同步生成"""
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        config: BackendConfig = None,
        callback: Callable[[str], None] = None
    ) -> Generator[str, None, None]:
        """流式生成"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用"""
        pass


class OllamaBackend(BaseBackend):
    """Ollama 后端"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.base_url = config.base_url

    def is_available(self) -> bool:
        """检查 Ollama 是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []

    def generate(self, prompt: str, config: BackendConfig = None, **kwargs) -> str:
        """同步生成"""
        if config is None:
            config = self.config

        payload = {
            "model": config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens,
                "repeat_penalty": config.repeat_penalty,
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
        except Exception as e:
            print(f"Ollama 生成错误: {e}")

        return "抱歉，服务暂时不可用。"

    def stream_generate(
        self,
        prompt: str,
        config: BackendConfig = None,
        callback: Callable[[str], None] = None
    ) -> Generator[str, None, None]:
        """流式生成"""
        if config is None:
            config = self.config

        payload = {
            "model": config.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "num_predict": config.max_tokens,
                "repeat_penalty": config.repeat_penalty,
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=60
            )

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            if callback:
                                callback(token)
                            yield token
                        if data.get("done", False):
                            break
                    except:
                        pass
        except Exception as e:
            print(f"Ollama 流式生成错误: {e}")


class LlamaCppBackend(BaseBackend):
    """llama.cpp 后端 (LLM)"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        if not LLAMA_CPP_AVAILABLE:
            print("llama-cpp-python 未安装")
            return

        if self.config.model_path and os.path.exists(self.config.model_path):
            print(f"加载 llama.cpp 模型: {self.config.model_path}")
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=self.config.n_gpu_layers,
            )
            print("模型加载成功")
        else:
            print(f"模型文件不存在: {self.config.model_path}")

    def is_available(self) -> bool:
        """检查后端是否可用"""
        return LLAMA_CPP_AVAILABLE and self.model is not None

    def generate(self, prompt: str, config: BackendConfig = None, **kwargs) -> str:
        """同步生成"""
        if not self.is_available():
            return "模型未加载"

        if config is None:
            config = self.config

        try:
            output = self.model(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                echo=False,
            )
            return output["choices"][0]["text"]
        except Exception as e:
            print(f"llama.cpp 生成错误: {e}")
            return "生成失败"

    def stream_generate(
        self,
        prompt: str,
        config: BackendConfig = None,
        callback: Callable[[str], None] = None
    ) -> Generator[str, None, None]:
        """流式生成"""
        if not self.is_available():
            return

        if config is None:
            config = self.config

        try:
            output = self.model(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                echo=False,
                stream=True,
            )

            for chunk in output:
                token = chunk["choices"][0].get("text", "")
                if token:
                    if callback:
                        callback(token)
                    yield token
        except Exception as e:
            print(f"llama.cpp 流式生成错误: {e}")


class HuggingFaceBackend(BaseBackend):
    """HuggingFace Transformers 后端"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """加载模型"""
        if not TRANSFORMERS_AVAILABLE:
            print("transformers 库未安装")
            return

        model_name = self.config.model_name

        try:
            print(f"加载 HuggingFace 模型: {model_name}")

            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map="auto" if self.config.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.config.device == "cuda":
                self.model = self.model.cuda()

            self.model.eval()
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")

    def is_available(self) -> bool:
        """检查后端是否可用"""
        return TRANSFORMERS_AVAILABLE and self.model is not None

    def generate(self, prompt: str, config: BackendConfig = None, **kwargs) -> str:
        """同步生成"""
        if not self.is_available():
            return "模型未加载"

        if config is None:
            config = self.config

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if config.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return response
        except Exception as e:
            print(f"HuggingFace 生成错误: {e}")
            return "生成失败"

    def stream_generate(
        self,
        prompt: str,
        config: BackendConfig = None,
        callback: Callable[[str], None] = None
    ) -> Generator[str, None, None]:
        """流式生成"""
        if not self.is_available():
            return

        if config is None:
            config = self.config

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if config.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            generated_ids = inputs["input_ids"]

            for _ in range(config.max_tokens):
                with torch.no_grad():
                    outputs = self.model(generated_ids)
                    next_token_logits = outputs.logits[:, -1, :] / config.temperature

                    # Top-k 采样
                    if config.top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, config.top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')

                    # Top-p 采样
                    if config.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                        sorted_indices_to_remove = cumulative_probs > config.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0

                        indices_to_remove = sorted_indices_to_remove.scatter(
                            1, sorted_indices, sorted_indices_to_remove
                        )
                        next_token_logits[indices_to_remove] = float('-inf')

                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                token = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                if token:
                    if callback:
                        callback(token)
                    yield token

        except Exception as e:
            print(f"HuggingFace 流式生成错误: {e}")


class CloudAPIBackend(BaseBackend):
    """亦API 云端模型后端 (OpenAI 兼容)"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.api_key = config.api_key or DEFAULT_API_KEY
        self.base_url = config.base_url.rstrip("/") or "https://api.yiziyun.com"
        self.available = False
        self._models = []
        self._check_availability()

    def _check_availability(self):
        try:
            resp = requests.get(
                f"{self.base_url}/v1/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._models = [m["id"] for m in data.get("data", [])]
                self.available = True
                print(f"亦API 已连接，可用模型: {', '.join(self._models[:5])}...")
        except Exception as e:
            print(f"亦API 连接失败: {e}")

    def is_available(self) -> bool:
        return self.available

    def list_models(self) -> List[str]:
        return self._models

    def _build_messages(self, prompt: str) -> List[Dict]:
        """将简单 prompt 转为 OpenAI messages 格式"""
        lines = prompt.split("\n")
        messages = []
        for line in lines:
            if line.startswith("系统: "):
                messages.append({"role": "system", "content": line[4:]})
            elif line.startswith("用户: "):
                messages.append({"role": "user", "content": line[4:]})
            elif line.startswith("助手: "):
                messages.append({"role": "assistant", "content": line[4:]})
        if not messages:
            messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, config: BackendConfig = None, **kwargs) -> str:
        if config is None:
            config = self.config
        try:
            messages = self._build_messages(prompt)
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": config.model_name,
                    "messages": messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            return f"API 错误: {resp.status_code}"
        except Exception as e:
            return f"生成失败: {e}"

    def stream_generate(
        self,
        prompt: str,
        config: BackendConfig = None,
        callback: Callable[[str], None] = None,
    ) -> Generator[str, None, None]:
        if config is None:
            config = self.config
        try:
            messages = self._build_messages(prompt)
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": config.model_name,
                    "messages": messages,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "stream": True,
                },
                timeout=60,
                stream=True,
            )
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if callback:
                            callback(content)
                        yield content
                except:
                    pass
        except Exception as e:
            print(f"亦API 流式生成错误: {e}")


class LocalModelBackend(BaseBackend):
    """本地 Emind 模型后端"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载本地模型"""
        from model import create_model

        model_path = self.config.model_path or "checkpoints/model_multiround_epoch4888.pt"
        tokenizer_path = "checkpoints/tokenizer.json"

        if not os.path.exists(model_path):
            print(f"本地模型不存在: {model_path}")
            return

        # 加载分词器
        self.tokenizer = SimpleTokenizer()
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)

        # 加载模型
        device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        model_config = checkpoint.get('model_config', {})

        self.model = create_model(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        print(f"本地模型已加载: {model_path}")

    def is_available(self) -> bool:
        """检查后端是否可用"""
        return self.model is not None

    def generate(self, prompt: str, config: BackendConfig = None, **kwargs) -> str:
        """同步生成"""
        if not self.is_available():
            return "模型未加载"

        if config is None:
            config = self.config

        # 编码
        encoded = self.tokenizer.encode(prompt)
        if len(encoded) > config.n_ctx:
            encoded = encoded[-config.n_ctx:]

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        input_ids = torch.tensor([encoded], dtype=torch.long).to(device)

        # 生成
        generated = []
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', 3)

        with torch.no_grad():
            for _ in range(config.max_tokens):
                logits = self.model(input_ids)[1][:, -1, :]

                # 温度采样
                if config.temperature > 0:
                    logits = logits / config.temperature

                # Top-k
                if config.top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus)
                if config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cum_probs > config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()
                if token_id == eos_token_id:
                    break
                generated.append(token_id)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return self.tokenizer.decode(generated)

    def stream_generate(
        self,
        prompt: str,
        config: BackendConfig = None,
        callback: Callable[[str], None] = None
    ) -> Generator[str, None, None]:
        """流式生成"""
        if not self.is_available():
            return

        if config is None:
            config = self.config

        # 编码
        encoded = self.tokenizer.encode(prompt)
        if len(encoded) > config.n_ctx:
            encoded = encoded[-config.n_ctx:]

        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        input_ids = torch.tensor([encoded], dtype=torch.long).to(device)

        # 生成
        generated = []
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', 3)
        decode = getattr(self.tokenizer, 'id_to_token', None)

        with torch.no_grad():
            for _ in range(config.max_tokens):
                logits = self.model(input_ids)[1][:, -1, :]

                if config.temperature > 0:
                    logits = logits / config.temperature

                if config.top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, config.top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()

                if token_id == eos_token_id:
                    break

                generated.append(token_id)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                if decode:
                    decoded_char = decode(token_id)
                else:
                    decoded_char = chr(token_id) if token_id < 256 else ''
                if decoded_char and decoded_char not in ('<pad>', '<s>', '</s>', '<unk>'):
                    if callback:
                        callback(decoded_char)
                    yield decoded_char

                if len(input_ids[0]) > config.n_ctx:
                    break

                generated.append(token_id)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                decoded_char = self.tokenizer.itos.get(token_id, '')
                if decoded_char and decoded_char not in special_tokens:
                    if callback:
                        callback(decoded_char)
                    yield decoded_char


class VLLMBackend(BaseBackend):
    """vLLM 后端 — 高性能推理引擎 (PagedAttention + Prefix Caching + Speculative Decoding)"""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.engine = None
        self._load_model()

    def _load_model(self):
        if not VLLM_AVAILABLE:
            print("vLLM 未安装 (pip install vllm)")
            return

        if VLLM_INTEGRATION_AVAILABLE:
            try:
                cfg = VLLMConfig(
                    model_path=self.config.model_path or self.config.model_name,
                    model_name=self.config.model_name,
                    max_model_len=self.config.vllm_max_model_len or self.config.n_ctx,
                    gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                    tensor_parallel_size=self.config.vllm_tensor_parallel_size,
                    enable_prefix_caching=self.config.vllm_enable_prefix_caching,
                    dtype=self.config.vllm_dtype,
                    enable_chunked_prefill=self.config.vllm_enable_chunked_prefill,
                    use_server_mode=(self.config.backend_type == "vllm_server"),
                    speculative=SpeculativeDecodingConfig(
                        enabled=self.config.vllm_enable_speculative,
                        draft_model=self.config.vllm_speculative_draft_model,
                        num_speculative_tokens=self.config.vllm_num_speculative_tokens,
                    ) if VLLM_INTEGRATION_AVAILABLE else None,
                    lora=LoRAConfig(
                        enabled=self.config.vllm_enable_lora,
                        lora_dir=self.config.vllm_lora_dir,
                    ) if VLLM_INTEGRATION_AVAILABLE else None,
                )
                if self.config.vllm_quantization:
                    cfg.dtype = self.config.vllm_dtype
                    if self.config.vllm_quantization in ("fp8",):
                        cfg.dtype = "bfloat16"

                self.engine = VLLMIntegratedEngine(cfg)
                if self.engine.is_available:
                    print(f"vLLM 引擎已就绪 (模式: {self.engine.mode})")
                    return
            except Exception as e:
                print(f"vLLM 集成模块加载失败: {e}")

        # Fallback: 使用原生 vLLM API
        model_path = self.config.model_path or self.config.model_name
        if not model_path:
            print("vLLM: 未指定模型路径或名称")
            return
        try:
            print(f"加载 vLLM 模型 (原生): {model_path}")
            kwargs = {
                "model": model_path,
                "tensor_parallel_size": self.config.vllm_tensor_parallel_size or 1,
                "max_model_len": self.config.vllm_max_model_len or self.config.n_ctx,
                "trust_remote_code": True,
                "gpu_memory_utilization": self.config.vllm_gpu_memory_utilization or 0.90,
                "enable_prefix_caching": self.config.vllm_enable_prefix_caching,
            }
            if self.config.vllm_dtype and self.config.vllm_dtype != "auto":
                kwargs["dtype"] = self.config.vllm_dtype
            elif torch.cuda.is_bf16_supported():
                kwargs["dtype"] = "bfloat16"
            else:
                kwargs["dtype"] = "float16"
            self.model = VLLM(**kwargs)
            print("vLLM 模型加载成功")
            print(f"  Prefix Caching: {self.config.vllm_enable_prefix_caching}")
        except Exception as e:
            print(f"vLLM 加载失败: {e}")

    def is_available(self) -> bool:
        if self.engine is not None:
            return self.engine.is_available
        return VLLM_AVAILABLE and self.model is not None

    def generate(self, prompt: str, config: BackendConfig = None, **kwargs) -> str:
        cfg = config or self.config
        if self.engine is not None:
            return self.engine.generate(
                prompt,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repetition_penalty=cfg.repeat_penalty,
            )
        if not self.is_available():
            return "vLLM not available"
        try:
            params = SamplingParams(
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                max_tokens=cfg.max_tokens,
                repetition_penalty=cfg.repeat_penalty,
            )
            result = self.model.generate([prompt], params)
            return result[0].outputs[0].text
        except Exception as e:
            return f"vLLM generate error: {e}"

    def stream_generate(
        self, prompt: str, config: BackendConfig = None,
        callback: Callable[[str], None] = None,
    ) -> Generator[str, None, None]:
        cfg = config or self.config
        if self.engine is not None:
            for token in self.engine.generate_stream(
                prompt,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                repetition_penalty=cfg.repeat_penalty,
            ):
                if token:
                    if callback:
                        callback(token)
                    yield token
            return
        if not self.is_available():
            return
        try:
            params = SamplingParams(
                temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                max_tokens=cfg.max_tokens, repetition_penalty=cfg.repeat_penalty,
            )
            for result in self.model.generate([prompt], params):
                for output in result.outputs:
                    for token_id in output.token_ids:
                        text = self.model.get_tokenizer().decode([token_id])
                        if text:
                            if callback:
                                callback(text)
                            yield text
        except Exception as e:
            print(f"vLLM stream error: {e}")


class UnifiedInferenceEngine:
    """统一推理引擎 - 自动选择可用后端"""

    def __init__(self, config: BackendConfig = None):
        self.config = config or BackendConfig()
        self.backend: Optional[BaseBackend] = None
        self._init_backend()

    def _init_backend(self):
        """初始化后端"""
        backend_type = self.config.backend_type

        print(f"初始化后端: {backend_type}")

        if backend_type in ("vllm", "vllm_server"):
            self.backend = VLLMBackend(self.config)
            if self.backend.is_available():
                mode_str = "Server" if backend_type == "vllm_server" else "Embedded"
                print(f"vLLM 后端 ({mode_str}) 已就绪")
                return
            print("vLLM 不可用，尝试其他后端...")

        if backend_type == "ollama":
            self.backend = OllamaBackend(self.config)
            if self.backend.is_available():
                print("Ollama 后端已就绪")
                return
            print("Ollama 不可用，尝试其他后端...")

        if backend_type == "llama_cpp" or backend_type == "llm":
            self.backend = LlamaCppBackend(self.config)
            if self.backend.is_available():
                print("llama.cpp 后端已就绪")
                return
            print("llama.cpp 不可用，尝试其他后端...")

        if backend_type == "huggingface" or backend_type == "hf":
            self.backend = HuggingFaceBackend(self.config)
            if self.backend.is_available():
                print("HuggingFace 后端已就绪")
                return
            print("HuggingFace 不可用，尝试其他后端...")

        if backend_type == "cloud_api" or backend_type == "yiziyun":
            self.backend = CloudAPIBackend(self.config)
            if self.backend.is_available():
                print("亦API 后端已就绪")
                return
            print("亦API 不可用，尝试其他后端...")

        # 默认使用本地模型
        self.backend = LocalModelBackend(self.config)
        if self.backend.is_available():
            print("本地模型后端已就绪")
            return

        print("警告: 所有后端都不可用")

    def generate(self, prompt: str, config: BackendConfig = None, **kwargs) -> str:
        """同步生成"""
        if self.backend:
            return self.backend.generate(prompt, config or self.config)
        return "后端未初始化"

    def stream_generate(
        self,
        prompt: str,
        config: BackendConfig = None,
        callback: Callable[[str], None] = None
    ) -> Generator[str, None, None]:
        """流式生成"""
        if self.backend:
            return self.backend.stream_generate(prompt, config or self.config, callback)
        return

    def is_available(self) -> bool:
        """检查后端是否可用"""
        return self.backend is not None and self.backend.is_available()

    def get_backend_info(self) -> Dict[str, Any]:
        """获取后端信息"""
        info = {
            "backend_type": self.config.backend_type,
            "is_available": self.is_available(),
            "model_name": self.config.model_name,
        }

        if isinstance(self.backend, OllamaBackend):
            info["available_models"] = self.backend.list_models()

        return info


def create_inference_engine(
    backend_type: str = "auto",
    model_name: str = None,
    model_path: str = None,
    base_url: str = "http://localhost:11434",
    device: str = "cuda",
    api_key: str = None,
    **vllm_kwargs,
) -> UnifiedInferenceEngine:
    """创建推理引擎工厂函数"""

    if backend_type == "auto":
        # 优先尝试 vLLM (检测可用性和 GPU 能力)
        if VLLM_AVAILABLE and (model_path or os.environ.get("VLLM_MODEL_PATH")):
            model_path = model_path or os.environ.get("VLLM_MODEL_PATH")
            backend_type = "vllm"
            print(f"自动选择: vLLM ({model_path})")

    if backend_type == "auto":
        # 其次尝试亦API
        try:
            key = api_key or DEFAULT_API_KEY
            resp = requests.get(
                "https://api.yiziyun.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=5,
            )
            if resp.status_code == 200:
                backend_type = "cloud_api"
                base_url = "https://api.yiziyun.com"
                api_key = key
        except:
            pass

        if backend_type == "auto":
            if REQUESTS_AVAILABLE:
                try:
                    response = requests.get(f"{base_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        backend_type = "ollama"
                except:
                    pass

        if backend_type == "auto":
            if REQUESTS_AVAILABLE and model_path:
                backend_type = "local"

        if backend_type == "auto":
            backend_type = "local"

    config = BackendConfig(
        backend_type=backend_type,
        model_name=model_name or "llama2",
        model_path=model_path,
        base_url=base_url,
        device=device,
        api_key=api_key,
        # vLLM 特定参数 (通过 **vllm_kwargs 传入)
        vllm_enable_prefix_caching=vllm_kwargs.get("enable_prefix_caching", True),
        vllm_enable_speculative=vllm_kwargs.get("enable_speculative", False),
        vllm_speculative_draft_model=vllm_kwargs.get("speculative_draft_model", None),
        vllm_num_speculative_tokens=vllm_kwargs.get("num_speculative_tokens", 5),
        vllm_enable_lora=vllm_kwargs.get("enable_lora", False),
        vllm_lora_dir=vllm_kwargs.get("lora_dir", None),
        vllm_tensor_parallel_size=vllm_kwargs.get("tensor_parallel_size", 1),
        vllm_gpu_memory_utilization=vllm_kwargs.get("gpu_memory_utilization", 0.90),
        vllm_dtype=vllm_kwargs.get("dtype", "auto"),
        vllm_quantization=vllm_kwargs.get("quantization", None),
        vllm_enable_chunked_prefill=vllm_kwargs.get("enable_chunked_prefill", True),
        vllm_max_model_len=vllm_kwargs.get("max_model_len", None),
    )

    return UnifiedInferenceEngine(config)


# 示例配置
DEFAULT_CONFIGS = {
    "vllm": BackendConfig(
        backend_type="vllm",
        model_path="./models/emind-7b",
        n_ctx=4096,
        vllm_enable_prefix_caching=True,
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.90,
    ),
    "vllm_server": BackendConfig(
        backend_type="vllm_server",
        model_path="./models/emind-7b",
        n_ctx=4096,
        base_url="http://localhost:8000",
        vllm_enable_prefix_caching=True,
        vllm_tensor_parallel_size=1,
        vllm_enable_chunked_prefill=True,
    ),
    "ollama": BackendConfig(
        backend_type="ollama",
        model_name="llama2:7b",
        base_url="http://localhost:11434"
    ),
    "llama_cpp": BackendConfig(
        backend_type="llama_cpp",
        model_path="./models/llama-2-7b-chat.gguf",
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=32
    ),
    "huggingface": BackendConfig(
        backend_type="huggingface",
        model_name="meta-llama/Llama-2-7b-chat-hf",
        device="cuda"
    ),
    "local": BackendConfig(
        backend_type="local",
        model_path="checkpoints/model.pt"
    ),
    "cloud_api": BackendConfig(
        backend_type="cloud_api",
        model_name="gpt-4o-mini",
        base_url="https://api.yiziyun.com",
        api_key=DEFAULT_API_KEY,
    ),
}


def get_cloud_api_models() -> List[str]:
    """获取亦API 可用模型列表"""
    if not DEFAULT_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.yiziyun.com/v1/models",
            headers={"Authorization": f"Bearer {DEFAULT_API_KEY}"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception as e:
        print(f"获取亦API 模型列表失败: {e}")
    return []


if __name__ == "__main__":
    print("统一推理引擎测试")
    print("支持的后端: ollama, llama_cpp, huggingface, local")

    # 创建引擎
    engine = create_inference_engine("auto")

    if engine.is_available():
        print(f"后端信息: {engine.get_backend_info()}")

        # 测试生成
        prompt = "你好，请介绍一下自己"
        print(f"\n测试提示: {prompt}")
        print("生成结果:", engine.generate(prompt))
    else:
        print("没有可用的推理后端")
