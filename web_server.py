"""
亦梓·智脑 Emind AI Web 对话服务 v2.0
品牌: 亦梓科技
特色: 会话管理 · 思维过程可视化 · 上下文记忆 · 多模式 · 模型竞技场
"""

import uuid
import time
import asyncio
import json
import re
import os
import glob
import random
import threading
import hmac
import hashlib
import base64
from urllib.parse import urlencode
from datetime import datetime
from typing import Dict, List, Optional, Generator, Tuple, AsyncGenerator
from collections import OrderedDict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
import uvicorn
import httpx

# 保留 asyncio.to_thread 的向后兼容别名
async def aiterate_sync_gen(sync_gen):
    """在独立线程中迭代同步生成器，产出异步生成器，避免阻塞事件循环"""
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    SENTINEL = object()
    def _run():
        try:
            for item in sync_gen:
                loop.call_soon_threadsafe(queue.put_nowait, item)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, SENTINEL)
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    while True:
        item = await queue.get()
        if item is SENTINEL:
            break
        yield item

app = FastAPI(title="亦梓·智脑 Emind AI", version="2.0")

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "web", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "web", "templates"))

# 统一推理后端
try:
    from unified_inference import UnifiedInferenceEngine, BackendConfig, create_inference_engine, get_cloud_api_models
except ImportError:
    UnifiedInferenceEngine = None  # type: ignore
    BackendConfig = None  # type: ignore
    create_inference_engine = None  # type: ignore
    get_cloud_api_models = None  # type: ignore

# vLLM 深度集成
try:
    from vllm_integration import VLLM_INTEGRATION_AVAILABLE, detect_vllm_capabilities  # type: ignore
except ImportError:
    VLLM_INTEGRATION_AVAILABLE = False  # type: ignore
    detect_vllm_capabilities = None  # type: ignore

# ============================================================================
# Simple Session Middleware (no itsdangerous dependency)
# ============================================================================

_SESSION_SECRET = os.urandom(32).hex()

class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cookie = request.cookies.get("emind_session")
        request.state.session = {}

        if cookie:
            try:
                parts = cookie.split(".", 1)
                if len(parts) == 2:
                    sig, data = parts
                    expected = hmac.new(_SESSION_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
                    if hmac.compare_digest(sig, expected):
                        request.state.session = json.loads(base64.b64decode(data).decode())
            except:
                request.state.session = {}

        response = await call_next(request)

        session = request.state.session
        if session:
            data = base64.b64encode(json.dumps(session).encode()).decode()
            sig = hmac.new(_SESSION_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
            cookie_val = f"{sig}.{data}"
            response.set_cookie("emind_session", cookie_val, max_age=86400 * 7, httponly=True, samesite="lax")
        else:
            response.delete_cookie("emind_session")

        return response

app.add_middleware(SessionMiddleware)

# ============================================================================
# OAuth2 Configuration (亦梓科技聚合登录)
# ============================================================================

OAUTH_CONFIG = {
    "server": os.environ.get("OAUTH_SERVER", ""),
    "client_id": os.environ.get("OAUTH_CLIENT_ID", ""),
    "client_secret": os.environ.get("OAUTH_CLIENT_SECRET", ""),
    "redirect_uri": os.environ.get("OAUTH_REDIRECT_URI", "http://localhost:3333/auth/callback"),
    "debug": os.environ.get("OAUTH_DEBUG", "false").lower() == "true",
}

def oauth_debug(msg: str, data=None):
    if OAUTH_CONFIG["debug"]:
        print(f"[OAuth] {msg}")
        if data:
            print(f"  {data}")

# ============================================================================
# Mode Configuration
# ============================================================================

MODES = {
    "normal": {
        "label": "标准助手",
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "top_k": 40,
        "system_prompt": "你是一个有帮助的AI助手，名字叫Emind，由亦梓科技开发。请用清晰、准确的中文回答用户的问题。",
    },
    "deep": {
        "label": "深度思考",
        "temperature": 0.5,
        "max_tokens": 2048,
        "top_p": 0.95,
        "top_k": 60,
        "system_prompt": "你是一个深度思考的AI助手，名字叫Emind，由亦梓科技开发。请进行深入分析，展示推理过程，给出详尽的回答。在回答前先在<think>标签内写下你的思考过程。",
    },
    "code": {
        "label": "代码模式",
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.95,
        "top_k": 50,
        "system_prompt": "你是一个专业的编程助手，名字叫Emind，由亦梓科技开发。擅长代码生成、调试、优化。提供可直接运行的代码，包含必要的注释。",
    },
    "creative": {
        "label": "创意模式",
        "temperature": 0.9,
        "max_tokens": 1536,
        "top_p": 0.9,
        "top_k": 80,
        "system_prompt": "你是一个富有创意的AI助手，名字叫Emind，由亦梓科技开发。请发挥想象力，给出新颖独特的想法和方案。语言可以生动活泼。",
    },
    "analysis": {
        "label": "分析模式",
        "temperature": 0.4,
        "max_tokens": 1536,
        "top_p": 0.9,
        "top_k": 40,
        "system_prompt": "你是一个分析型AI助手，名字叫Emind，由亦梓科技开发。请从多角度分析问题，给出结构化、数据驱动的回答。使用分点、表格等方式组织内容。",
    },
    "translate": {
        "label": "翻译",
        "temperature": 0.2,
        "max_tokens": 1024,
        "top_p": 0.9,
        "top_k": 30,
        "system_prompt": "你是一个翻译AI助手，名字叫Emind，由亦梓科技开发。请准确翻译用户提供的内容，保持原文风格和语气。只返回翻译结果，不做额外说明。",
    },
}

MODE_PREFIXES = {
    "/深度思考": "deep",
    "/代码": "code",
    "/创意": "creative",
    "/分析": "analysis",
    "/翻译": "translate",
}

# ============================================================================
# Helpers (PRESET_RESPONSES, resolve_mode, build_prompt, estimate_tokens)
# ============================================================================

PRESET_RESPONSES = [
    "你好！我是Emind，由亦梓科技开发的AI助手。很高兴为你服务！",
    "这是一个很好的问题！让我从多个角度来分析一下。",
    "让我思考一下这个问题，从我的知识库中检索相关信息...",
    "感谢你的提问！以下是我的回答。",
    "这个问题很有意思，我来详细解释一下。",
]


def resolve_mode(message: str):
    """Detect mode prefix like '/深度思考' and return (clean_message, mode)"""
    for prefix, mode in MODE_PREFIXES.items():
        if message.startswith(prefix):
            return message[len(prefix):].strip(), mode
    return message, None


def build_prompt(messages, system_prompt: str) -> str:
    """Build prompt from messages list for local models"""
    parts = [f"系统: {system_prompt}"]
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"用户: {content}")
        elif role == "assistant":
            parts.append(f"助手: {content}")
    parts.append("助手:")
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token count estimate for display purposes"""
    return len(text) * 3 // 2


# ============================================================================
# Thinking Simulator
# ============================================================================

THINKING_STEPS = {
    "deep": ["深入理解问题背景", "分析关键要素与约束", "构建推理框架", "逐步推导结论", "验证答案完整性"],
    "code": ["解析功能需求", "设计代码结构", "选择合适算法", "编写与调试", "检查边界情况"],
    "creative": ["理解创意方向", "发散联想相关点", "筛选最佳创意", "细化实施方案"],
    "analysis": ["明确分析目标", "收集相关数据", "多维度分析", "综合得出结论"],
    "translate": ["理解原文含义", "考虑语境文化", "选择准确表达", "润色译文"],
    "normal": ["理解用户意图", "检索相关知识", "组织回答结构", "生成最终回复"],
}


class ThinkingSimulator:
    @staticmethod
    def simulate(question: str, mode: str = "normal") -> str:
        steps = THINKING_STEPS.get(mode, THINKING_STEPS["normal"])
        if "代码" in question or "编程" in question or "python" in question.lower():
            steps = THINKING_STEPS["code"]
        elif "翻译" in question:
            steps = THINKING_STEPS["translate"]
        elif any(kw in question for kw in ["分析", "对比", "趋势"]):
            steps = THINKING_STEPS["analysis"]
        elif any(kw in question for kw in ["创意", "设计", "想象"]):
            steps = THINKING_STEPS["creative"]
        return "\n".join(f"[步骤 {i+1}/{len(steps)}] {s}..." for i, s in enumerate(steps))

    @staticmethod
    def parse_think_tags(text: str) -> Tuple[Optional[str], str]:
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return match.group(1).strip(), text.replace(match.group(0), "").strip()
        return None, text


# ============================================================================
# Context Analyzer
# ============================================================================

ENTITY_PATTERNS = [
    (r"[\u4e00-\u9fa5]{2,4}(?:先生|女士|老师|博士|教授|同学|朋友)", "人物称呼"),
    (r"[\u4e00-\u9fa5]{2,6}(?:公司|集团|科技|技术|系统|平台|算法|模型|产品|方案|品牌)", "组织机构"),
    (r"(?:北京|上海|广州|深圳|杭州|成都|武汉|南京|西安|重庆|天津|苏州)(?:市|)", "地点"),
    (r"\d{4}年\d{1,2}月|\d{1,2}月\d{1,2}日|今天|明天|昨天|上周|下周|本月", "时间"),
    (r"[\u4e00-\u9fa5]{2,6}(?:技术|框架|语言|库|工具|协议|标准|算法)", "技术术语"),
]


class ContextAnalyzer:
    @staticmethod
    def extract_entities(text: str) -> List[Dict[str, str]]:
        entities, seen = [], set()
        for pattern, category in ENTITY_PATTERNS:
            for m in re.findall(pattern, text):
                if m not in seen:
                    seen.add(m)
                    entities.append({"name": m, "category": category})
        return entities

    @staticmethod
    def extract_keypoints(text: str, max_points: int = 5) -> List[str]:
        key_sentences = []
        for s in re.split(r"[。！？\n]+", text):
            s = s.strip()
            if 8 < len(s) < 80 and any(kw in s for kw in ["重要", "关键", "核心", "记住", "注意", "所以", "意味", "表明"]):
                key_sentences.append(s)
                if len(key_sentences) >= max_points:
                    break
        return key_sentences

    @staticmethod
    def extract_insights(messages: List[Dict], turn_count: int) -> List[str]:
        insights = []
        if turn_count >= 3:
            insights.append("多轮对话")
        if turn_count >= 6:
            insights.append("话题深入")
        if messages:
            last = messages[-1].get("content", "")
            if any(kw in last for kw in ["?" , "？" , "如何" , "为什么" , "怎样"]):
                insights.append("提问模式")
            if len(last) > 300:
                insights.append("长文本回复")
        return insights


# ============================================================================
# Session Management
# ============================================================================

class ChatSession:
    def __init__(self, session_id: str = None, mode: str = "normal"):
        self.id = session_id or str(uuid.uuid4())[:8]
        self.mode = mode
        self.messages: List[Dict] = []
        self.context = {
            "entities": [],
            "keypoints": [],
            "insights": [],
            "turn_count": 0,
            "total_tokens": 0,
        }
        self.created_at = datetime.now().isoformat()
        self.title = "新对话"

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "mode": self.mode,
            "created_at": self.created_at,
            "message_count": len(self.messages),
        }


class SessionManager:
    def __init__(self, max_sessions: int = 50):
        self.sessions: Dict[str, ChatSession] = OrderedDict()
        self.current_id: Optional[str] = None
        self.max_sessions = max_sessions

    def create(self, mode: str = "normal") -> ChatSession:
        session = ChatSession(mode=mode)
        self.sessions[session.id] = session
        self.current_id = session.id
        while len(self.sessions) > self.max_sessions:
            self.sessions.pop(next(iter(self.sessions)))
        return session

    def get(self, session_id: str = None) -> ChatSession:
        if session_id and session_id in self.sessions:
            self.current_id = session_id
            return self.sessions[session_id]
        if self.current_id and self.current_id in self.sessions:
            return self.sessions[self.current_id]
        return self.create()

    def switch(self, session_id: str) -> Optional[ChatSession]:
        if session_id in self.sessions:
            self.current_id = session_id
            return self.sessions[session_id]
        return None

    def delete(self, session_id: str) -> bool:
        if session_id in self.sessions:
            del self.sessions[session_id]
            if self.current_id == session_id:
                self.current_id = next(iter(self.sessions)) if self.sessions else None
            return True
        return False

    def clear(self, session_id: str = None):
        s = self.get(session_id)
        s.messages = []
        s.context = {"entities": [], "keypoints": [], "insights": [], "turn_count": 0, "total_tokens": 0}

    def list_sessions(self) -> List[Dict]:
        return [s.to_dict() for s in reversed(self.sessions.values())]

    def get_or_create(self, session_id: str = None, mode: str = "normal") -> ChatSession:
        if session_id and session_id in self.sessions:
            return self.get(session_id)
        return self.create(mode)


# ============================================================================
# Global State
# ============================================================================

session_mgr = SessionManager()
_global_engine = None

def get_engine() -> UnifiedInferenceEngine:
    global _global_engine
    if _global_engine is None:
        # 优先尝试加载本地 checkpoint
        model_path = None
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--model" and i + 1 < len(sys.argv[1:]):
                model_path = sys.argv[i + 2]
                break
        if model_path and os.path.exists(model_path):
            _global_engine = create_inference_engine("local", model_path=model_path)
        else:
            _global_engine = create_inference_engine("cloud_api")
    return _global_engine


def _update_context(chat_session: ChatSession):
    ctx = chat_session.context
    all_text = " ".join(m.get("content", "") for m in chat_session.messages[-6:])
    seen_names = set(e["name"] for e in ctx["entities"])
    for e in ContextAnalyzer.extract_entities(all_text):
        if e["name"] not in seen_names:
            ctx["entities"].append(e)
            seen_names.add(e["name"])
    for msg in reversed(chat_session.messages[-4:]):
        if msg["role"] == "assistant":
            for kp in ContextAnalyzer.extract_keypoints(msg.get("content", "")):
                if kp not in ctx["keypoints"]:
                    ctx["keypoints"].append(kp)
                    ctx["keypoints"] = ctx["keypoints"][-10:]
    ctx["insights"] = ContextAnalyzer.extract_insights(chat_session.messages, ctx["turn_count"])
    if chat_session.title == "新对话":
        for msg in chat_session.messages:
            if msg["role"] == "user":
                title = msg["content"][:30]
                if len(msg["content"]) > 30:
                    title += "..."
                chat_session.title = title
                break


# ============================================================================
# Pydantic Models
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    mode: Optional[str] = None
    show_thinking: bool = True
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


class SessionCreateRequest(BaseModel):
    mode: str = "normal"


class SessionSwitchRequest(BaseModel):
    session_id: str


class SessionDeleteRequest(BaseModel):
    session_id: str


class ClearRequest(BaseModel):
    session_id: Optional[str] = None


class ArenaRequest(BaseModel):
    prompt: str
    models: List[str]
    max_tokens: int = 500
    temperature: float = 0.7


class BackendSwitchRequest(BaseModel):
    backend_type: str = "auto"
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    base_url: str = "http://localhost:11434"
    # vLLM 参数
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


class OpenAIRequest(BaseModel):
    messages: List[Dict]
    stream: bool = False
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None  # "auto", "none", or "required"


class CompletionsRequest(BaseModel):
    prompt: str = ""
    stream: bool = False
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9


# ============================================================================
# Routes
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
async def chat(req: ChatRequest):
    clean_message, detected_mode = resolve_mode(req.message)
    mode = req.mode or detected_mode or "normal"

    max_tokens = req.max_tokens or MODES.get(mode, MODES["normal"])["max_tokens"]
    temperature = req.temperature or MODES.get(mode, MODES["normal"])["temperature"]
    top_p = req.top_p or MODES.get(mode, MODES["normal"])["top_p"]
    top_k = req.top_k or MODES.get(mode, MODES["normal"])["top_k"]
    show_thinking = req.show_thinking

    chat_session = session_mgr.get_or_create(req.session_id, mode)
    chat_session.messages.append({"role": "user", "content": clean_message})

    mode_config = MODES.get(mode, MODES["normal"])
    system_prompt = mode_config["system_prompt"]
    engine = get_engine()

    async def event_stream() -> AsyncGenerator[str, None]:
        assistant_response = ""
        thinking_text = ""
        try:
            if engine and engine.is_available():
                prompt = build_prompt(chat_session.messages, system_prompt)
                config = BackendConfig(
                    backend_type=engine.config.backend_type,
                    model_name=engine.config.model_name,
                    api_key=engine.config.api_key,
                    base_url=engine.config.base_url,
                    model_path=engine.config.model_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                if show_thinking:
                    simulated = ThinkingSimulator.simulate(clean_message, mode)
                    for line in simulated.split("\n"):
                        yield f"data: {json.dumps({'thinking': line})}\n\n"
                        await asyncio.sleep(0.04)
                    thinking_text = simulated

                async for token in aiterate_sync_gen(engine.stream_generate(prompt, config=config)):
                    if token:
                        assistant_response += token
                        yield f"data: {json.dumps({'token': token})}\n\n"

                parsed_thinking, clean_response = ThinkingSimulator.parse_think_tags(assistant_response)
                if parsed_thinking:
                    thinking_text = parsed_thinking
                    assistant_response = clean_response
            else:
                full = random.choice(PRESET_RESPONSES)
                for char in full:
                    assistant_response += char
                    yield f"data: {json.dumps({'token': char})}\n\n"
                    await asyncio.sleep(0.015)
        except Exception as e:
            print(f"生成错误: {e}")
            if not assistant_response:
                fallback = random.choice(PRESET_RESPONSES)
                for char in fallback:
                    assistant_response += char
                    yield f"data: {json.dumps({'token': char})}\n\n"
                    await asyncio.sleep(0.015)

        chat_session.messages.append({
            "role": "assistant", "content": assistant_response, "thinking": thinking_text,
        })
        chat_session.context["turn_count"] += 1
        chat_session.context["total_tokens"] += estimate_tokens(clean_message) + estimate_tokens(assistant_response)
        _update_context(chat_session)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat/simple")
async def chat_simple(req: ChatRequest):
    clean_message, detected_mode = resolve_mode(req.message)
    mode = req.mode or detected_mode or "normal"
    max_tokens = req.max_tokens or MODES.get(mode, MODES["normal"])["max_tokens"]
    temperature = req.temperature or MODES.get(mode, MODES["normal"])["temperature"]

    chat_session = session_mgr.get_or_create(req.session_id, mode)
    chat_session.messages.append({"role": "user", "content": clean_message})

    mode_config = MODES.get(mode, MODES["normal"])
    system_prompt = mode_config["system_prompt"]
    engine = get_engine()
    response_text = ""

    try:
        if engine and engine.is_available():
            prompt = build_prompt(chat_session.messages[:-1], system_prompt)
            config = BackendConfig(
                backend_type=engine.config.backend_type,
                model_name=engine.config.model_name,
                api_key=engine.config.api_key,
                base_url=engine.config.base_url,
                model_path=engine.config.model_path,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response_text = await asyncio.to_thread(engine.generate, prompt, config)
    except Exception as e:
        print(f"生成错误: {e}")

    if not response_text.strip():
        response_text = random.choice(PRESET_RESPONSES)

    chat_session.messages.append({"role": "assistant", "content": response_text})
    chat_session.context["turn_count"] += 1
    _update_context(chat_session)

    return {"response": response_text, "session_id": chat_session.id}


@app.post("/api/session/create")
async def create_session(req: SessionCreateRequest):
    s = session_mgr.create(req.mode)
    return {"status": "ok", "session": s.to_dict()}


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": session_mgr.list_sessions(), "current_id": session_mgr.current_id}


@app.post("/api/session/switch")
async def switch_session(req: SessionSwitchRequest):
    s = session_mgr.switch(req.session_id)
    if not s:
        raise HTTPException(404, "会话不存在")
    return {"status": "ok", "session": s.to_dict(), "messages": s.messages, "context": s.context}


@app.post("/api/session/delete")
async def delete_session(req: SessionDeleteRequest):
    if session_mgr.delete(req.session_id):
        return {"status": "ok"}
    raise HTTPException(404, "会话不存在")


@app.post("/api/clear")
async def clear_history(req: ClearRequest):
    session_mgr.clear(req.session_id)
    return {"status": "ok", "message": "对话历史已清除"}


@app.get("/api/context")
async def get_context(session_id: Optional[str] = None):
    s = session_mgr.get(session_id)
    return {"session_id": s.id, "context": s.context, "message_count": len(s.messages), "mode": s.mode}


@app.post("/api/arena")
async def arena_compare(req: ArenaRequest):
    if not req.prompt:
        raise HTTPException(400, "prompt 必填")
    if len(req.models) < 2:
        raise HTTPException(400, "请选择至少 2 个模型")

    results = []
    from unified_inference import OllamaBackend, LocalModelBackend

    for model_ref in req.models:
        try:
            if model_ref == "emind-7b":
                engine = create_inference_engine("auto")
            elif model_ref == "emind-fast":
                if get_engine().is_available():
                    engine = get_engine()
                else:
                    engine = create_inference_engine("local")
            elif model_ref == "emind-deep":
                engine = create_inference_engine("auto")
            else:
                engine = get_engine()

            if engine and engine.is_available():
                config = BackendConfig(
                    backend_type=engine.config.backend_type,
                    model_name=engine.config.model_name,
                    api_key=engine.config.api_key,
                    base_url=engine.config.base_url,
                    model_path=engine.config.model_path,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
                response = engine.generate(req.prompt, config=config)
                results.append({"model": model_ref, "response": response, "error": None})
            else:
                results.append({"model": model_ref, "response": None, "error": "后端不可用"})
        except Exception as e:
            results.append({"model": model_ref, "response": None, "error": str(e)})

    return {"results": results}


@app.get("/api/backends")
async def list_backends():
    backends = []

    # vLLM
    vllm_available = False
    vllm_caps = {}
    try:
        if VLLM_INTEGRATION_AVAILABLE:
            vllm_caps = detect_vllm_capabilities()
            vllm_available = vllm_caps.get("available", False)
    except:
        pass
    backends.append({
        "type": "vllm",
        "name": "vLLM (深度推理引擎)",
        "available": vllm_available,
        "models": [],
        "features": vllm_caps.get("features", []),
        "description": "vLLM 高性能推理 — PagedAttention + Prefix Caching + Speculative Decoding + 量化",
    })

    # 亦API 云端模型
    cloud_models = []
    cloud_available = False
    try:
        from unified_inference import get_cloud_api_models
        cloud_models = get_cloud_api_models()
        cloud_available = len(cloud_models) > 0
    except:
        pass
    backends.append({"type": "cloud_api", "name": "亦API 云端模型", "available": cloud_available, "models": cloud_models, "description": "亦梓科技云端 API"})

    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            backends.append({"type": "ollama", "name": "Ollama", "available": True, "models": models, "description": "本地运行的 LLM 服务"})
    except:
        backends.append({"type": "ollama", "name": "Ollama", "available": False, "models": [], "description": "本地运行的 LLM 服务"})
    try:
        from llama_cpp import Llama
        backends.append({"type": "llama_cpp", "name": "llama.cpp (LLM)", "available": True, "models": [], "description": "基于 GGUF 模型的本地推理"})
    except:
        backends.append({"type": "llama_cpp", "name": "llama.cpp (LLM)", "available": False, "models": [], "description": "基于 GGUF 模型的本地推理"})
    try:
        from transformers import AutoModelForCausalLM
        backends.append({"type": "huggingface", "name": "HuggingFace", "available": True, "models": [], "description": "HuggingFace Transformers 模型"})
    except:
        backends.append({"type": "huggingface", "name": "HuggingFace", "available": False, "models": [], "description": "HuggingFace Transformers 模型"})
    local_available = bool(glob.glob("checkpoints/**/*.pt", recursive=True))
    backends.append({"type": "local", "name": "本地模型", "available": local_available, "models": ["Emind Model"] if local_available else [], "description": "Emind 本地训练模型"})

    engine = get_engine()
    return {"backends": backends, "current_backend": engine.config.backend_type if engine else None}


@app.post("/api/backend/switch")
async def switch_backend(req: BackendSwitchRequest):
    global _global_engine
    try:
        # vLLM 模式: 走深度集成
        if req.backend_type in ("vllm", "vllm_server"):
            if VLLM_INTEGRATION_AVAILABLE:
                from vllm_integration import VLLMIntegratedEngine, VLLMConfig, VLLMServingConfig, SpeculativeDecodingConfig, LoRAConfig
                spec = SpeculativeDecodingConfig(
                    enabled=req.vllm_enable_speculative,
                    num_speculative_tokens=req.vllm_num_speculative_tokens,
                )
                lora = LoRAConfig(
                    enabled=req.vllm_enable_lora,
                    lora_dir=req.vllm_lora_dir,
                )
                vllm_cfg = VLLMConfig(
                    model_path=req.model_path or req.model_name,
                    model_name=req.model_name or "emind",
                    use_server_mode=(req.backend_type == "vllm_server"),
                    enable_prefix_caching=req.vllm_enable_prefix_caching,
                    gpu_memory_utilization=req.vllm_gpu_memory_utilization,
                    tensor_parallel_size=req.vllm_tensor_parallel_size,
                    dtype=req.vllm_dtype,
                    speculative=spec,
                    lora=lora,
                )
                engine = VLLMIntegratedEngine(vllm_cfg)
                if engine.is_available:
                    print(f"vLLM 后端切换成功: {engine.mode}")
                    return {"status": "ok", "backend_type": f"vllm_{engine.mode}", "is_available": True, "info": engine.get_info()}
            return {"status": "error", "message": "vLLM 不可用"}

        _global_engine = create_inference_engine(
            backend_type=req.backend_type,
            model_name=req.model_name,
            model_path=req.model_path,
            base_url=req.base_url,
        )
        return {"status": "ok", "backend_type": req.backend_type, "is_available": _global_engine.is_available(), "info": _global_engine.get_backend_info()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/models")
async def list_models():
    """获取所有可用模型（含亦API 云端模型 + vLLM 能力）"""
    models = {"vllm": {"available": False, "features": []}, "cloud_api": [], "ollama": [], "local": []}

    # vLLM
    try:
        if VLLM_INTEGRATION_AVAILABLE:
            caps = detect_vllm_capabilities()
            models["vllm"] = {
                "available": caps.get("available", False),
                "version": caps.get("version"),
                "features": caps.get("features", []),
                "gpu_count": caps.get("gpu_count", 0),
                "has_bf16": caps.get("has_bf16", False),
            }
    except:
        pass

    try:
        from unified_inference import get_cloud_api_models
        models["cloud_api"] = get_cloud_api_models()
    except:
        pass
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            models["ollama"] = [m["name"] for m in r.json().get("models", [])]
    except:
        pass
    checkpoints_dir = "checkpoints"
    if os.path.exists(checkpoints_dir):
        models["local"] = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
    return models


@app.get("/v1/models")
async def openai_list_models():
    """OpenAI 兼容的模型列表 API"""
    from unified_inference import get_cloud_api_models
    data = [{"id": "emind-7b", "object": "model", "created": int(time.time()), "owned_by": "emind"}]
    try:
        cloud_models = get_cloud_api_models()
        for m in cloud_models:
            data.append({"id": m, "object": "model", "created": int(time.time()), "owned_by": "yiziyun"})
    except:
        pass
    return {"object": "list", "data": data}


@app.post("/v1/completions")
async def openai_completions(req: CompletionsRequest):
    """OpenAI 兼容的 completions API (非聊天)"""
    engine = get_engine()
    prompt_text = req.prompt

    async def stream():
        response_text = ""
        try:
            if engine and engine.is_available():
                prompt_text = req.prompt
                config = BackendConfig(
                    backend_type=engine.config.backend_type,
                    model_name=engine.config.model_name,
                    api_key=engine.config.api_key,
                    base_url=engine.config.base_url,
                    model_path=engine.config.model_path,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                )
                async for token in aiterate_sync_gen(engine.stream_generate(prompt_text, config=config)):
                    if token:
                        response_text += token
                        if req.stream:
                            chunk = {
                                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                                "object": "text_completion",
                                "created": int(time.time()),
                                "model": engine.config.model_name,
                                "choices": [{"text": token, "index": 0}],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"OpenAI completions error: {e}")
        if req.stream:
            yield "data: [DONE]\n\n"
        else:
            result = {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": engine.config.model_name if engine else "emind",
                "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": estimate_tokens(prompt_text), "completion_tokens": estimate_tokens(response_text),
                          "total_tokens": estimate_tokens(prompt_text) + estimate_tokens(response_text)},
            }
            yield json.dumps(result, ensure_ascii=False)

    media_type = "text/event-stream" if req.stream else "application/json"
    return StreamingResponse(stream(), media_type=media_type)


@app.get("/api/health")
async def health():
    engine = get_engine()
    status = {"status": "ok", "version": "2.0", "sessions": len(session_mgr.sessions), "current_session": session_mgr.current_id}
    if engine:
        status["current_backend"] = engine.config.backend_type
        status["is_available"] = engine.is_available()
        status["backend_info"] = engine.get_backend_info()
    return status


def _build_tool_prompt(tools: List[Dict]) -> str:
    """Inject tool definitions into system prompt for function calling."""
    lines = ["\n\n你可以使用以下工具："]
    for i, tool in enumerate(tools):
        name = tool.get("function", tool).get("name", f"tool_{i}")
        desc = tool.get("function", tool).get("description", "")
        params = tool.get("function", tool).get("parameters", {})
        lines.append(f"\n工具 {i+1}: {name}")
        lines.append(f"  描述: {desc}")
        lines.append(f"  参数: {json.dumps(params, ensure_ascii=False)}")
    lines.append("\n当你需要调用工具时，请按以下格式输出（只输出一次，不要额外说明）：")
    lines.append('  <tool_call>{"name": "工具名", "arguments": {"参数名": "值"}}</tool_call>')
    return "\n".join(lines)


def _parse_tool_call(text: str) -> Optional[Dict]:
    """Parse <tool_call>...</tool_call> from model output."""
    m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            return {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)},
            }
        except (json.JSONDecodeError, KeyError):
            pass
    return None


@app.post("/v1/chat/completions")
async def openai_compatible_chat(req: OpenAIRequest):
    system_prompt = "你是一个有帮助的AI助手，名字叫Emind，由亦梓科技开发。"

    # Inject tool definitions if provided
    if req.tools:
        system_prompt += _build_tool_prompt(req.tools)

    # Build conversation with full history
    conversation_parts = []
    for msg in req.messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
            if req.tools:
                system_prompt += _build_tool_prompt(req.tools)
        elif role == "user":
            conversation_parts.append(f"用户: {content}")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    conversation_parts.append(f"助手: [调用工具 {fn.get('name', '')} 参数 {fn.get('arguments', '')}]")
            else:
                conversation_parts.append(f"助手: {content}")
        elif role == "tool":
            conversation_parts.append(f"工具结果: {content}")

    conversation_parts.append("助手:")
    user_message = "\n".join(conversation_parts)

    engine = get_engine()

    async def generate_openai():
        response_text = ""
        try:
            if engine and engine.is_available():
                prompt = f"系统: {system_prompt}\n{user_message}"
                config = BackendConfig(
                    backend_type=engine.config.backend_type,
                    model_name=engine.config.model_name,
                    api_key=engine.config.api_key,
                    base_url=engine.config.base_url,
                    model_path=engine.config.model_path,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                )
                async for token in aiterate_sync_gen(engine.stream_generate(prompt, config=config)):
                    if token:
                        response_text += token
                        if req.stream:
                            chunk = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": engine.config.model_name,
                                "choices": [{"delta": {"content": token}, "index": 0}],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            print(f"OpenAI API 生成错误: {e}")
            if not response_text:
                response_text = random.choice(PRESET_RESPONSES)

        if req.stream:
            yield "data: [DONE]\n\n"
        else:
            # Check for tool call in response
            tool_call = _parse_tool_call(response_text)
            if tool_call:
                clean_text = re.sub(r'<tool_call>.*?</tool_call>', '', response_text, flags=re.DOTALL).strip()
                message = {"role": "assistant", "content": clean_text or None, "tool_calls": [tool_call]}
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": response_text}
                finish = "stop"

            result = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": engine.config.model_name if engine else "emind",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": estimate_tokens(user_message),
                    "completion_tokens": estimate_tokens(response_text),
                    "total_tokens": estimate_tokens(user_message) + estimate_tokens(response_text),
                },
            }
            yield json.dumps(result, ensure_ascii=False)

    media_type = "text/event-stream" if req.stream else "application/json"
    return StreamingResponse(generate_openai(), media_type=media_type)


# ============================================================================
# OAuth2 Authentication (亦梓科技聚合登录)
# ============================================================================

def _oauth_guard(request: Request):
    """Check OAuth is configured before proceeding."""
    if not OAUTH_CONFIG.get("server") or not OAUTH_CONFIG.get("client_id"):
        raise HTTPException(503, "OAuth 未配置 (需要设置 OAUTH_SERVER / OAUTH_CLIENT_ID 环境变量)")


@app.get("/auth/login")
async def auth_login(request: Request):
    """Redirect to OAuth server for login"""
    _oauth_guard(request)
    import secrets
    state = secrets.token_urlsafe(32)
    request.state.session["oauth_state"] = state

    params = {
        "client_id": OAUTH_CONFIG["client_id"],
        "redirect_uri": OAUTH_CONFIG["redirect_uri"],
        "state": state,
        "response_type": "code",
    }

    auth_url = f"{OAUTH_CONFIG['server']}/oauth/authorize.php?{urlencode(params)}"
    oauth_debug("Login redirect", auth_url)
    return RedirectResponse(url=auth_url)


@app.get("/auth/callback")
async def auth_callback(request: Request):
    """Handle OAuth callback"""
    _oauth_guard(request)
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if not code:
        raise HTTPException(400, "缺少授权码(code)参数")
    if not state:
        raise HTTPException(400, "缺少state参数")

    # Verify state
    session_state = request.state.session.get("oauth_state")
    if not session_state or session_state != state:
        raise HTTPException(400, "State验证失败")

    oauth_debug("Received code", {"code": code[:20] + "..."})

    try:
        # 1. Exchange code for access_token
        token_url = f"{OAUTH_CONFIG['server']}/oauth/token.php"
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "client_id": OAUTH_CONFIG["client_id"],
            "client_secret": OAUTH_CONFIG["client_secret"],
            "redirect_uri": OAUTH_CONFIG["redirect_uri"],
        }

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(token_url, data=token_data)
            token_info = resp.json()

        oauth_debug("Token response", token_info)

        if "error" in token_info:
            raise HTTPException(400, f"获取token失败: {token_info.get('error_description', token_info.get('error'))}")

        access_token = token_info.get("access_token")
        if not access_token:
            raise HTTPException(400, "响应中缺少access_token")

        # 2. Get user info
        user_info_url = f"{OAUTH_CONFIG['server']}/oauth/userinfo.php"
        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(user_info_url, headers=headers)
            user_info = resp.json()

        oauth_debug("User info", user_info)

        if "error" in user_info:
            raise HTTPException(400, f"获取用户信息失败: {user_info.get('error_description', '未知错误')}")

        if "uid" not in user_info:
            raise HTTPException(400, "响应中缺少用户ID")

        # 3. Save to session
        request.state.session["user"] = user_info
        request.state.session["access_token"] = access_token
        request.state.session["login_time"] = int(time.time())

        # Clean up
        request.state.session.pop("oauth_state", None)

        oauth_debug("Login complete", {"uid": user_info["uid"], "username": user_info.get("username")})

        return RedirectResponse(url="/", status_code=302)

    except HTTPException:
        raise
    except Exception as e:
        oauth_debug("Auth error", str(e))
        raise HTTPException(500, f"登录失败: {str(e)}")


@app.post("/auth/logout")
async def auth_logout(request: Request):
    """Logout and clear session"""
    request.state.session.clear()
    return {"status": "ok", "message": "已退出登录"}


@app.get("/auth/me")
async def auth_me(request: Request):
    """Get current user info"""
    user = request.state.session.get("user")
    if user:
        return {
            "logged_in": True,
            "user": user,
            "login_time": request.state.session.get("login_time"),
        }
    return {"logged_in": False}


@app.get("/auth/status")
async def auth_status(request: Request):
    """Check auth status"""
    return {
        "logged_in": "user" in request.state.session,
        "user": request.state.session.get("user"),
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    port = 3333
    model_path = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--port" and i + 1 < len(sys.argv[1:]):
            port = int(sys.argv[i + 2])
            break
        if arg == "--model" and i + 1 < len(sys.argv[1:]):
            model_path = sys.argv[i + 2]

    print("\n" + "=" * 60)
    print("  亦梓·智脑 Emind AI Web 对话服务 v2.0")
    print("  亦梓科技 © 2026")
    print("=" * 60)

    if model_path and os.path.exists(model_path):
        engine = create_inference_engine("local", model_path=model_path)
        _global_engine = engine
    else:
        engine = get_engine()

    if engine:
        info = engine.get_backend_info()
        print(f"  后端: {info['backend_type']} | 可用: {info['is_available']}")
    print(f"  地址: http://localhost:{port}")
    print(f"  API 文档: http://localhost:{port}/docs")

    # vLLM 能力检测
    if VLLM_INTEGRATION_AVAILABLE:
        try:
            caps = detect_vllm_capabilities()
            if caps.get("available"):
                print(f"\n  [vLLM {caps.get('version', '')} 已就绪]")
                features = caps.get("features", [])
                if features:
                    print(f"  特性: {', '.join(features)}")
                if caps.get("gpu_count", 0) > 0:
                    print(f"  GPU: {caps['gpu_count']}× {caps['gpu_names'][0] if caps.get('gpu_names') else ''}")
        except:
            pass

    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
