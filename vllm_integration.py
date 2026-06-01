"""
vLLM 深度集成 — 生产级推理优化引擎
提供: Prefix Caching, Speculative Decoding, 量化 (AWQ/GPTQ/FP8), 多 LoRA 动态切换,
vLLM Server 生命周期管理, OpenAI 兼容路由, 健康检查/自动重连, 多 GPU 张量并行
"""

import os
import json
import time
import uuid
import subprocess
import threading
from typing import Optional, Dict, Any, List, Generator, Callable, Union
from dataclasses import dataclass, field

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from vllm import LLM as VLLMEngine, SamplingParams as VLLMSamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SpeculativeDecodingConfig:
    """推测解码配置"""
    enabled: bool = False
    draft_model: Optional[str] = None
    num_speculative_tokens: int = 5
    draft_model_tp_size: Optional[int] = None
    use_ngram: bool = False
    ngram_prompt_lookup_max: int = 4
    ngram_prompt_lookup_min: int = 1
    use_eagle: bool = False
    eagle_ckpt_dir: Optional[str] = None


@dataclass
class LoRAConfig:
    """多 LoRA 配置"""
    enabled: bool = False
    max_loras: int = 16
    max_lora_rank: int = 64
    lora_extra_vocab_size: int = 0
    lora_dir: Optional[str] = None
    lora_modules: Dict[str, str] = field(default_factory=dict)


@dataclass
class VLLMServingConfig:
    """vLLM 服务配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: Optional[str] = None
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None
    seed: int = 42
    trust_remote_code: bool = True
    enforce_eager: bool = False
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    enable_chunked_prefill: bool = True
    max_logprobs: int = 20
    extra_cli_args: List[str] = field(default_factory=list)


@dataclass
class VLLMConfig:
    """vLLM 完整配置"""
    # 基本
    model_path: Optional[str] = None
    model_name: str = "emind-4b"
    trust_remote_code: bool = True
    seed: int = 42

    # 内存 / 性能
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    swap_space: int = 4
    dtype: str = "auto"
    enforce_eager: bool = False
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    enable_chunked_prefill: bool = True

    # 并行
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Prefix Caching
    enable_prefix_caching: bool = True

    # 使用 vLLM Server 模式 (OpenAI 兼容) 还是嵌入式模式
    use_server_mode: bool = False

    # Server 模式配置
    serving: VLLMServingConfig = field(default_factory=VLLMServingConfig)

    # Speculative Decoding
    speculative: SpeculativeDecodingConfig = field(default_factory=SpeculativeDecodingConfig)

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # 采样默认值
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 40
    default_max_tokens: int = 1024
    default_repetition_penalty: float = 1.05


# Available quantization options
QUANTIZATION_OPTIONS = {
    "awq": "AWQ (Activation-aware Weight Quantization, 4-bit)",
    "gptq": "GPTQ (Post-Training Quantization, 4/8-bit)",
    "fp8": "FP8 (8-bit Float, Hopper GPUs only)",
    "squeezellm": "SqueezeLLM (Dense-and-Sparse Quantization)",
    "sparsemar": "SparseMAR (Sparse Mixture of Adaptive Replicas)",
    "bitsandbytes": "BitsAndBytes (8/4-bit)",
    "none": None,
}


# =============================================================================
# Health Monitor
# =============================================================================

class VLLMHealthMonitor:
    """vLLM Server 健康检查 + 自动重连"""

    def __init__(
        self,
        base_url: str,
        check_interval: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        on_unhealthy: Optional[Callable] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.check_interval = check_interval
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.on_unhealthy = on_unhealthy
        self._healthy = True
        self._last_check = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _check(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        retries = 0
        while not self._stop_event.is_set():
            healthy = self._check()
            if not healthy and self._healthy:
                retries += 1
                if retries > self.max_retries:
                    self._healthy = False
                    if self.on_unhealthy:
                        self.on_unhealthy()
                    retries = 0
            elif healthy:
                self._healthy = True
                retries = 0
            self._stop_event.wait(self.check_interval)

    @property
    def is_healthy(self) -> bool:
        return self._healthy


# =============================================================================
# Embedded vLLM Engine (直接进程内加载)
# =============================================================================

if VLLM_AVAILABLE:

    class EmbeddedVLLMEngine:
        """嵌入式 vLLM 引擎 — 进程内直接加载模型"""

        def __init__(self, config: VLLMConfig):
            self.config = config
            self.engine: Optional[VLLMEngine] = None
            self._model_path = config.model_path or config.model_name
            self._loaded = False
            self._load()

        def _build_kwargs(self) -> Dict[str, Any]:
            kwargs: Dict[str, Any] = {
                "model": self._model_path,
                "trust_remote_code": self.config.trust_remote_code,
                "seed": self.config.seed,
                "max_model_len": self.config.max_model_len,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "swap_space": self.config.swap_space,
                "dtype": self.config.dtype if self.config.dtype != "auto" else "auto",
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "pipeline_parallel_size": self.config.pipeline_parallel_size,
                "enable_prefix_caching": self.config.enable_prefix_caching,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
                "max_num_seqs": self.config.max_num_seqs,
                "enforce_eager": self.config.enforce_eager,
            }

            # Chunked prefill
            if self.config.enable_chunked_prefill:
                kwargs["enable_chunked_prefill"] = True

            # Quantization
            if self.config.dtype == "fp8":
                kwargs["dtype"] = "bfloat16"
                kwargs["quantization"] = "fp8"

            if self.config.lora.enabled:
                kwargs["enable_lora"] = True
                kwargs["max_loras"] = self.config.lora.max_loras
                kwargs["max_lora_rank"] = self.config.lora.max_lora_rank
                kwargs["lora_extra_vocab_size"] = self.config.lora.lora_extra_vocab_size

            # Speculative Decoding
            spec = self.config.speculative
            if spec.enabled:
                if spec.use_ngram:
                    kwargs["speculative_model"] = "[ngram]"
                    kwargs["num_speculative_tokens"] = spec.num_speculative_tokens
                    kwargs["ngram_prompt_lookup_max"] = spec.ngram_prompt_lookup_max
                    kwargs["ngram_prompt_lookup_min"] = spec.ngram_prompt_lookup_min
                elif spec.use_eagle and spec.eagle_ckpt_dir:
                    kwargs["speculative_model"] = spec.eagle_ckpt_dir
                    kwargs["num_speculative_tokens"] = spec.num_speculative_tokens
                elif spec.draft_model:
                    kwargs["speculative_model"] = spec.draft_model
                    kwargs["num_speculative_tokens"] = spec.num_speculative_tokens
                    if spec.draft_model_tp_size:
                        kwargs["speculative_draft_tensor_parallel_size"] = spec.draft_model_tp_size

            return kwargs

        def _load(self):
            if self._loaded:
                return
            try:
                kwargs = self._build_kwargs()
                print(f"[vLLM] 加载模型: {self._model_path}")
                print(f"[vLLM] Prefix Caching: {self.config.enable_prefix_caching}")
                print(f"[vLLM] 量化学: {self.config.dtype}")
                if self.config.speculative.enabled:
                    print(f"[vLLM] Speculative Decoding: {self.config.speculative.num_speculative_tokens} tokens")
                if self.config.lora.enabled:
                    print(f"[vLLM] LoRA: {len(self.config.lora.lora_modules)} modules")
                self.engine = VLLMEngine(**kwargs)
                self._loaded = True
                print(f"[vLLM] 模型加载完成")
            except Exception as e:
                print(f"[vLLM] 模型加载失败: {e}")
                self._loaded = False

        def generate(
            self,
            prompts: Union[str, List[str]],
            sampling_params: Optional["VLLMSamplingParams"] = None,
            lora_request: Optional[str] = None,
        ) -> List[str]:
            if not self._loaded or self.engine is None:
                return ["vLLM 未加载"] if isinstance(prompts, str) else ["vLLM 未加载"] * len(prompts)

            if isinstance(prompts, str):
                prompts = [prompts]

            params = sampling_params or VLLMSamplingParams(
                temperature=self.config.default_temperature,
                top_p=self.config.default_top_p,
                top_k=self.config.default_top_k,
                max_tokens=self.config.default_max_tokens,
                repetition_penalty=self.config.default_repetition_penalty,
            )

            try:
                if lora_request and self.config.lora.enabled:
                    from vllm.lora.request import LoRARequest
                    lora_path = self.config.lora.lora_modules.get(lora_request) or os.path.join(self.config.lora.lora_dir or '', lora_request)
                    lora = LoRARequest(lora_request, 1, lora_path)
                    results = self.engine.generate(prompts, params, lora_request=lora)
                else:
                    results = self.engine.generate(prompts, params)

                return [r.outputs[0].text for r in results]
            except Exception as e:
                print(f"[vLLM] 生成失败: {e}")
                return [""] * len(prompts)

        def generate_stream(
            self,
            prompt: str,
            sampling_params: Optional["VLLMSamplingParams"] = None,
            lora_request: Optional[str] = None,
        ) -> Generator[str, None, None]:
            if not self._loaded or self.engine is None:
                return

            params = sampling_params or VLLMSamplingParams(
                temperature=self.config.default_temperature,
                top_p=self.config.default_top_p,
                top_k=self.config.default_top_k,
                max_tokens=self.config.default_max_tokens,
                repetition_penalty=self.config.default_repetition_penalty,
            )

            try:
                if lora_request and self.config.lora.enabled:
                    from vllm.lora.request import LoRARequest
                    lora_path = self.config.lora.lora_modules.get(lora_request) or os.path.join(self.config.lora.lora_dir or '', lora_request)
                    lora = LoRARequest(lora_request, 1, lora_path)
                    for result in self.engine.generate([prompt], params, lora_request=lora):
                        for output in result.outputs:
                            for token_id in output.token_ids:
                                token_text = self.engine.get_tokenizer().decode([token_id])
                                if token_text:
                                    yield token_text
                else:
                    for result in self.engine.generate([prompt], params):
                        for output in result.outputs:
                            for token_id in output.token_ids:
                                token_text = self.engine.get_tokenizer().decode([token_id])
                                if token_text:
                                    yield token_text
            except Exception as e:
                print(f"[vLLM] 流式生成失败: {e}")

        def get_model_info(self) -> Dict[str, Any]:
            if not self._loaded or self.engine is None:
                return {"loaded": False}
            try:
                return {
                    "loaded": True,
                    "model": self._model_path,
                    "max_model_len": self.config.max_model_len,
                    "prefix_caching": self.config.enable_prefix_caching,
                    "speculative": self.config.speculative.enabled,
                    "tensor_parallel": self.config.tensor_parallel_size,
                    "lora": self.config.lora.enabled,
                    "dtype": self.config.dtype,
                }
            except Exception:
                return {"loaded": True}

        @property
        def is_loaded(self) -> bool:
            return self._loaded and self.engine is not None


# =============================================================================
# vLLM Server 模式 (OpenAI 兼容)
# =============================================================================

class VLLMServerManager:
    """管理外部 vLLM server 进程的生命周期"""

    def __init__(self, config: VLLMConfig):
        self.config = config
        self.serving = config.serving
        self.process: Optional[subprocess.Popen] = None
        self.monitor: Optional[VLLMHealthMonitor] = None
        self._server_ready = threading.Event()

    def _build_server_args(self) -> List[str]:
        spec = self.config.speculative
        lora = self.config.lora
        args = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_path or self.config.model_name,
            "--host", self.serving.host,
            "--port", str(self.serving.port),
            "--max-model-len", str(self.config.max_model_len),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--seed", str(self.config.seed),
        ]

        if self.config.trust_remote_code:
            args.append("--trust-remote-code")

        if self.serving.api_key:
            args.extend(["--api-key", self.serving.api_key])

        if self.config.dtype and self.config.dtype != "auto":
            args.extend(["--dtype", self.config.dtype])

        if self.serving.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")

        if self.config.enable_prefix_caching:
            args.append("--enable-prefix-caching")

        if self.serving.enforce_eager:
            args.append("--enforce-eager")

        if self.serving.max_num_seqs:
            args.extend(["--max-num-seqs", str(self.serving.max_num_seqs)])

        if self.serving.max_logprobs:
            args.extend(["--max-logprobs", str(self.serving.max_logprobs)])

        if self.serving.quantization and self.serving.quantization != "none":
            args.extend(["--quantization", self.serving.quantization])

        if self.serving.max_num_batched_tokens:
            args.extend(["--max-num-batched-tokens", str(self.serving.max_num_batched_tokens)])

        # Speculative decoding
        if spec.enabled:
            if spec.use_ngram:
                args.append("--speculative-model")
                args.append("[ngram]")
                args.extend(["--num-speculative-tokens", str(spec.num_speculative_tokens)])
                args.extend(["--ngram-prompt-lookup-max", str(spec.ngram_prompt_lookup_max)])
                args.extend(["--ngram-prompt-lookup-min", str(spec.ngram_prompt_lookup_min)])
            elif spec.use_eagle and spec.eagle_ckpt_dir:
                args.extend(["--speculative-model", spec.eagle_ckpt_dir])
                args.extend(["--num-speculative-tokens", str(spec.num_speculative_tokens)])
            elif spec.draft_model:
                args.extend(["--speculative-model", spec.draft_model])
                args.extend(["--num-speculative-tokens", str(spec.num_speculative_tokens)])
                if spec.draft_model_tp_size:
                    args.extend(["--speculative-draft-tensor-parallel-size", str(spec.draft_model_tp_size)])

        # LoRA
        if lora.enabled:
            args.append("--enable-lora")
            args.extend(["--max-loras", str(lora.max_loras)])
            args.extend(["--max-lora-rank", str(lora.max_lora_rank)])
            if lora.lora_dir:
                args.extend(["--lora-modules"] + [f"{k}={v}" for k, v in lora.lora_modules.items()])

        # Extra CLI args
        args.extend(self.serving.extra_cli_args)

        return args

    def start(self, wait_ready: bool = True, timeout: float = 120.0) -> bool:
        if self.process is not None:
            print("[vLLM Server] 已在运行")
            return True

        args = self._build_server_args()
        print(f"[vLLM Server] 启动命令:\n  {' '.join(args)}")

        try:
            self.process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as e:
            print(f"[vLLM Server] 无法启动: {e}")
            return False

        self.server_url = f"http://{self.serving.host}:{self.serving.port}"
        self.monitor = VLLMHealthMonitor(
            base_url=self.server_url,
            on_unhealthy=self._on_unhealthy,
        )
        self.monitor.start()

        if wait_ready:
            ready = self._wait_for_ready(timeout)
            if ready:
                print(f"[vLLM Server] 就绪: {self.server_url}")
            else:
                print(f"[vLLM Server] 启动超时 ({timeout}s)")
            return ready

        return True

    def _wait_for_ready(self, timeout: float) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.monitor and self.monitor._check():
                return True
            time.sleep(2)
        return False

    def _on_unhealthy(self):
        print("[vLLM Server] 健康检查失败, 尝试重启...")
        self.restart()

    def stop(self):
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        if self.process:
            print("[vLLM Server] 停止...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
        self._server_ready.clear()

    def restart(self):
        self.stop()
        return self.start()

    @property
    def is_running(self) -> bool:
        if self.process is None:
            return False
        ret = self.process.poll()
        return ret is None

    @property
    def base_url(self) -> str:
        return f"http://{self.serving.host}:{self.serving.port}"


# =============================================================================
# OpenAI 兼容路由 (将 vLLM Server API 封装为 Emind 统一接口)
# =============================================================================

class VLLMServerClient:
    """vLLM Server 的 HTTP 客户端 — 实现 Emind 统一后端接口"""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _post(self, endpoint: str, data: Dict) -> Dict:
        url = f"{self.base_url}{endpoint}"
        resp = requests.post(url, headers=self._headers, json=data, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def _stream_post(self, endpoint: str, data: Dict) -> Generator[Dict, None, None]:
        url = f"{self.base_url}{endpoint}"
        data["stream"] = True
        with requests.post(url, headers=self._headers, json=data, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue

    def chat(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: Optional[List[Dict]] = None,
    ) -> Union[Dict, Generator[Dict, None, None]]:
        data = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if tools:
            data["tools"] = tools
        return self._post("/v1/chat/completions", data)

    def chat_stream(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tools: Optional[List[Dict]] = None,
    ) -> Generator[Dict, None, None]:
        data = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if tools:
            data["tools"] = tools
        return self._stream_post("/v1/chat/completions", data)

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9) -> str:
        data = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        result = self._post("/v1/completions", data)
        return result.get("choices", [{}])[0].get("text", "")

    def generate_stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9) -> Generator[str, None, None]:
        data = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        for chunk in self._stream_post("/v1/completions", data):
            text = chunk.get("choices", [{}])[0].get("text", "")
            if text:
                yield text

    @property
    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def get_model_info(self) -> Dict:
        try:
            r = requests.get(f"{self.base_url}/v1/models", headers=self._headers, timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {"models": []}


# =============================================================================
# Dynamic LoRA Manager
# =============================================================================

class DynamicLoRAManager:
    """动态 LoRA 模块管理 — 支持运行时热加载/卸载"""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.environ.get("EMIND_LORA_DIR", "loras")
        self._modules: Dict[str, str] = {}
        self._active: Optional[str] = None
        self._scan()

    def _scan(self):
        if not os.path.isdir(self.base_dir):
            return
        for entry in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, entry)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                self._modules[entry] = path
                print(f"[LoRA] 发现: {entry}")

    def list(self) -> Dict[str, str]:
        return dict(self._modules)

    def activate(self, name: str) -> Optional[str]:
        if name in self._modules:
            self._active = name
            print(f"[LoRA] 激活: {name} → {self._modules[name]}")
            return self._modules[name]
        print(f"[LoRA] 未找到: {name}")
        return None

    def deactivate(self):
        self._active = None
        print("[LoRA] 已停用")

    def add(self, name: str, path: str):
        self._modules[name] = path

    def remove(self, name: str):
        self._modules.pop(name, None)
        if self._active == name:
            self._active = None

    @property
    def active(self) -> Optional[str]:
        return self._active

    @property
    def active_path(self) -> Optional[str]:
        if self._active:
            return self._modules.get(self._active)
        return None


# =============================================================================
# High-Level Unified Interface
# =============================================================================

class VLLMIntegratedEngine:
    """vLLM 统一引擎 — 自动选择嵌入式/Server 模式, 集成所有优化"""

    def __init__(self, config: Optional[VLLMConfig] = None):
        self.config = config or VLLMConfig()
        self._engine: Optional[EmbeddedVLLMEngine] = None
        self._server: Optional[VLLMServerManager] = None
        self._client: Optional[VLLMServerClient] = None
        self._lora_manager = DynamicLoRAManager()
        self._mode: str = "none"
        self._init()

    def _init(self):
        if not VLLM_AVAILABLE:
            print("[vLLM] vllm 未安装 (pip install vllm)")
            return

        if self.config.use_server_mode:
            self._mode = "server"
            self._server = VLLMServerManager(self.config)
            started = self._server.start(wait_ready=True)
            if started:
                self._client = VLLMServerClient(
                    self._server.base_url,
                    api_key=self.config.serving.api_key,
                )
                print(f"[vLLM] Server 模式已启动: {self._server.base_url}")
            else:
                print("[vLLM] Server 模式启动失败")
        else:
            self._mode = "embedded"
            self._engine = EmbeddedVLLMEngine(self.config)
            if self._engine.is_loaded:
                print(f"[vLLM] 嵌入式模式已加载: {self.config.model_path or self.config.model_name}")
            else:
                print("[vLLM] 嵌入式模式加载失败")

    # ================================================================
    # 核心接口
    # ================================================================

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        lora_name: Optional[str] = None,
    ) -> str:
        if self._mode == "server" and self._client:
            return self._client.generate(
                prompt,
                max_tokens=max_tokens or self.config.default_max_tokens,
                temperature=temperature or self.config.default_temperature,
                top_p=top_p or self.config.default_top_p,
            )
        elif self._mode == "embedded" and self._engine:
            params = VLLMSamplingParams(
                temperature=temperature or self.config.default_temperature,
                top_p=top_p or self.config.default_top_p,
                top_k=top_k or self.config.default_top_k,
                max_tokens=max_tokens or self.config.default_max_tokens,
                repetition_penalty=repetition_penalty or self.config.default_repetition_penalty,
                stop=stop or [],
            )
            results = self._engine.generate(prompt, params, lora_request=lora_name)
            return results[0] if results else ""
        return "vLLM 不可用"

    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        lora_name: Optional[str] = None,
    ) -> Generator[str, None, None]:
        if self._mode == "server" and self._client:
            for chunk in self._client.generate_stream(
                prompt,
                max_tokens=max_tokens or self.config.default_max_tokens,
                temperature=temperature or self.config.default_temperature,
                top_p=top_p or self.config.default_top_p,
            ):
                yield chunk
        elif self._mode == "embedded" and self._engine:
            params = VLLMSamplingParams(
                temperature=temperature or self.config.default_temperature,
                top_p=top_p or self.config.default_top_p,
                top_k=top_k or self.config.default_top_k,
                max_tokens=max_tokens or self.config.default_max_tokens,
                repetition_penalty=repetition_penalty or self.config.default_repetition_penalty,
                stop=stop or [],
            )
            yield from self._engine.generate_stream(prompt, params, lora_request=lora_name)

    def chat(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
    ) -> str:
        if self._mode == "server" and self._client:
            result = self._client.chat(
                messages,
                max_tokens=max_tokens or self.config.default_max_tokens,
                temperature=temperature or self.config.default_temperature,
                top_p=top_p or self.config.default_top_p,
                tools=tools,
            )
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            prompt = self._messages_to_prompt(messages)
            return self.generate(prompt, max_tokens, temperature, top_p)

    def chat_stream(
        self,
        messages: List[Dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        if self._mode == "server" and self._client:
            for chunk in self._client.chat_stream(
                messages,
                max_tokens=max_tokens or self.config.default_max_tokens,
                temperature=temperature or self.config.default_temperature,
                top_p=top_p or self.config.default_top_p,
                tools=tools,
            ):
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
        else:
            prompt = self._messages_to_prompt(messages)
            yield from self.generate_stream(prompt, max_tokens, temperature, top_p)

    @staticmethod
    def _messages_to_prompt(messages: List[Dict]) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)

    # ================================================================
    # 引擎管理
    # ================================================================

    @property
    def is_available(self) -> bool:
        if self._mode == "server" and self._client:
            return self._client.is_available
        elif self._mode == "embedded" and self._engine:
            return self._engine.is_loaded
        return False

    @property
    def mode(self) -> str:
        return self._mode

    def get_info(self) -> Dict[str, Any]:
        info = {
            "mode": self._mode,
            "available": self.is_available,
            "prefix_caching": self.config.enable_prefix_caching,
            "speculative_decoding": self.config.speculative.enabled,
            "lora_enabled": self.config.lora.enabled,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "tensor_parallel_size": self.config.tensor_parallel_size,
        }
        if self._mode == "server" and self._client:
            info["server_url"] = self._server.base_url if self._server else None
            info["models"] = self._client.get_model_info()
        elif self._mode == "embedded" and self._engine:
            info["model_info"] = self._engine.get_model_info()
        return info

    def stop(self):
        if self._server:
            self._server.stop()
        self._engine = None
        self._client = None
        self._mode = "none"

    def restart(self):
        self.stop()
        self._init()

    # ================================================================
    # LoRA 管理
    # ================================================================

    @property
    def lora_manager(self) -> DynamicLoRAManager:
        return self._lora_manager


# =============================================================================
# 工厂函数
# =============================================================================

def create_vllm_engine(
    model_path: Optional[str] = None,
    model_name: str = "emind-4b",
    tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.90,
    enable_prefix_caching: bool = True,
    dtype: str = "auto",
    use_server_mode: bool = False,
    enable_speculative: bool = False,
    speculative_draft_model: Optional[str] = None,
    num_speculative_tokens: int = 5,
    enable_lora: bool = False,
    lora_dir: Optional[str] = None,
    quantization: Optional[str] = None,
    port: int = 8000,
    api_key: Optional[str] = None,
    **kwargs,
) -> VLLMIntegratedEngine:
    """创建 vLLM 引擎的工厂函数"""
    spec_config = SpeculativeDecodingConfig(
        enabled=enable_speculative,
        draft_model=speculative_draft_model,
        num_speculative_tokens=num_speculative_tokens,
    )
    lora_config = LoRAConfig(
        enabled=enable_lora,
        lora_dir=lora_dir,
    )
    serving_config = VLLMServingConfig(
        port=port,
        api_key=api_key,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        quantization=quantization,
    )
    config = VLLMConfig(
        model_path=model_path,
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        dtype=dtype,
        use_server_mode=use_server_mode,
        speculative=spec_config,
        lora=lora_config,
        serving=serving_config,
    )
    return VLLMIntegratedEngine(config)


# =============================================================================
# 快速检测
# =============================================================================

def detect_vllm_capabilities() -> Dict[str, Any]:
    """检测当前环境中 vLLM 支持的能力"""
    caps = {"available": VLLM_AVAILABLE, "version": None, "features": []}
    if not VLLM_AVAILABLE:
        return caps

    try:
        import vllm
        caps["version"] = getattr(vllm, "__version__", "unknown")

        def _ver_at_least(ver_str: str, target: str) -> bool:
            v1 = tuple(int(x) for x in ver_str.split("."))
            v2 = tuple(int(x) for x in target.split("."))
            return v1 >= v2

        ver = caps["version"]
        if _ver_at_least(ver, "0.4.0"):
            caps["features"].append("prefix_caching")
        if _ver_at_least(ver, "0.5.0"):
            caps["features"].append("speculative_decoding")
        if _ver_at_least(ver, "0.5.2"):
            caps["features"].append("fp8")
        if _ver_at_least(ver, "0.6.0"):
            caps["features"].append("ngram_speculator")
            caps["features"].append("chunked_prefill")
            caps["features"].append("eagle")
        if _ver_at_least(ver, "0.6.3"):
            caps["features"].append("multi_lora")

        caps["tensor_parallel"] = torch.cuda.device_count() if TORCH_AVAILABLE and torch.cuda.is_available() else 0
        if TORCH_AVAILABLE:
            caps["has_bf16"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            caps["gpu_count"] = torch.cuda.device_count()
            caps["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception:
        pass

    return caps


# =============================================================================
# CLI Shortcut
# =============================================================================

def auto_configure_for_gpu() -> VLLMConfig:
    """自动检测 GPU 并推荐 vLLM 配置"""
    caps = detect_vllm_capabilities()
    gpu_count = caps.get("gpu_count", 0)
    gpu_names = caps.get("gpu_names", [])

    config = VLLMConfig()
    config.enable_prefix_caching = "prefix_caching" in caps.get("features", [])
    config.tensor_parallel_size = gpu_count

    print(f"[vLLM Auto] 检测到 {gpu_count} GPU:")
    for name in gpu_names:
        print(f"  - {name}")
    print(f"[vLLM Auto] 推荐 TP={gpu_count}, Prefix Caching={config.enable_prefix_caching}")

    # RTX PRO 6000 96GB 推荐配置
    if any("RTX PRO 6000" in n for n in gpu_names):
        config.max_model_len = 65536
        config.gpu_memory_utilization = 0.92
        config.enable_chunked_prefill = True
        config.max_num_batched_tokens = 16384
        config.max_num_seqs = 512
        print(f"[vLLM Auto] RTX PRO 6000 96GB 优化: max_model_len=65536, gpu_mem=0.92")
        print(f"[vLLM Auto] 推荐: Speculative Decoding + FP8 + Prefix Caching")

    return config


if __name__ == "__main__":
    caps = detect_vllm_capabilities()
    print(json.dumps(caps, indent=2, ensure_ascii=False))
    if caps["available"]:
        cfg = auto_configure_for_gpu()
        print(f"\n推荐配置:\n{json.dumps(cfg.__dict__, indent=2, ensure_ascii=False)}")
