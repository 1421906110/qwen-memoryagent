"""
MemoryAgent — LLM integration layer with Qwen/DeepSeek optimized interface.

Key design for domestic models:
  - Qwen native features: enable_search, enable_thinking, JSON mode
  - Per-scenario temperature: tool calling 0.1, chat 0.5, creative 0.7
  - DeepSeek v4 flash optimized: higher token limits, 429 retry, stream usage
  - Built-in web search (Qwen native) instead of hand-crawled Bing
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── Default config ──
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-6-plus"
DEFAULT_FAST_MODEL = "qwen3-6-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v3"
DEFAULT_ENABLE_SEARCH = True
DEFAULT_THINKING_BUDGET = 1024

# ════════════════════════════════════════════
#  DeepSeek 深度优化配置
# ════════════════════════════════════════════
DEEPSEEK_DEFAULTS = {
    # ⭐ 参考官方文档: https://api-docs.deepseek.com/api/create-chat-completion
    "max_tokens": 16384,             # 聊天输出上限
    "stream_max_tokens": 8192,       # 流式输出上限
    "tool_max_tokens": 8192,         # 工具调用输出上限
    "json_max_tokens": 8192,         # JSON mode 输出上限
    "context_window": 1048576,       # 1M 上下文
    "temperature_chat": 0.7,         # 聊天温度（thinking 模式下失效）
    "temperature_tool": 0.1,         # 工具调用
    "temperature_json": 0.05,        # JSON 输出
    "temperature_creative": 0.9,     # 创意场景
    # ⭐ thinking 推理强度
    "reasoning_effort": "high",      # high=复杂任务, max=全力以赴
    # 429 限流
    "retry_max_429": 5,
    "retry_base_429": 2.0,
    "retry_max_other": 3,
    "retry_base_other": 1.0,
}
QWEN_DEFAULTS = {
    "max_tokens": 4096,
    "stream_max_tokens": 2048,
    "tool_max_tokens": 2048,
    "json_max_tokens": 2048,
    "context_window": 131072,
    "temperature_chat": 0.5,
    "temperature_tool": 0.1,
    "temperature_json": 0.05,
    "temperature_creative": 0.7,
    "retry_max_429": 3,
    "retry_base_429": 1.0,
    "retry_max_other": 3,
    "retry_base_other": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# Approximate token counting
CHARS_PER_TOKEN = 2.0
MAX_INPUT_TOKENS = {
    "qwen-plus": 131072, "qwen-max": 32768,
    "qwen-max-longcontext": 1000000,
    "qwen3-6-plus": 131072, "qwen3-6-flash": 131072, "qwen3-7-max": 131072,
    "deepseek-v4-flash": 65536,
}


def estimate_tokens(text: str) -> int:
    """Approximate token count."""
    return int(len(text) / CHARS_PER_TOKEN)


# ⭐ API 调用自动重试（指数退避），应对网络抖动/限流
def _api_call_with_retry(call_fn, max_retries: int = 3, base_delay: float = 1.0,
                         is_429: bool = False):
    """执行 API 调用，失败时自动重试。

    DeepSeek 429 限流更严：用更多重试 + 更长退避。

    Args:
        call_fn: 无参可调用对象
        max_retries: 最大重试次数
        base_delay: 初始退避秒数（每次翻倍）
        is_429: 如果是 429 错误，使用更激进的退避策略
    """
    for attempt in range(max_retries + 1):
        try:
            return call_fn()
        except Exception as e:
            if attempt < max_retries and _is_retryable(e):
                # 429 限流：退避更久，加随机抖动
                if is_429 or "429" in str(e) or "rate_limit" in str(e).lower():
                    delay = base_delay * (2 ** attempt) + (time.time() % 1) * 0.5
                else:
                    delay = base_delay * (2 ** attempt)
                logger.warning("🔄 API 调用失败（第 %d 次），%.1fs 后重试: %s",
                               attempt + 1, delay, str(e)[:80])
                time.sleep(delay)
            else:
                raise


def _is_retryable(e: Exception) -> bool:
    """判断错误是否值得重试（网络/限流/服务端错误）"""
    if isinstance(e, TimeoutError):
        return True
    msg = str(e).lower()
    if "timeout" in msg or "connection" in msg or "reset" in msg:
        return True
    if "429" in msg or "503" in msg or "502" in msg:
        return True
    if "rate_limit" in msg or "too many" in msg or "insufficient_quota" in msg:
        return True
    if "server error" in msg or "internal" in msg:
        return True
    if "bad gateway" in msg or "service temporarily" in msg:
        return True
    return False


class LLMClient:
    """LLM client optimized for Chinese domestic models (Qwen / DeepSeek).

    🔥 v0.17: 支持 provider:model 语法 + 运行时切换模型

    Key differences from standard OpenAI SDK wrapper:
    - Qwen-native features: enable_search, enable_thinking, JSON mode
    - Per-scenario temperature defaults (tool calling ≠ chat ≠ creative)
    - Built-in web search via Qwen (no need for external search tool)
    - Proper Chinese text generation parameters
    """

    # 🔥 v0.17: 支持的 provider → (类名, 默认 base_url)
    PROVIDERS = {
        "deepseek": ("openai", "https://api.deepseek.com/v1"),
        "qwen": ("openai", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        "openai": ("openai", "https://api.openai.com/v1"),
        "anthropic": ("anthropic", "https://api.anthropic.com/v1"),
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("QWEN_API_KEY", "")
        self.base_url = (base_url or os.getenv("QWEN_BASE_URL", "")).rstrip("/")
        self._model_str = model or os.getenv("QWEN_MODEL", DEFAULT_MODEL)
        self.fast_model = os.getenv("QWEN_FAST_MODEL", DEFAULT_FAST_MODEL)
        self.embedding_model = embedding_model or os.getenv(
            "QWEN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
        )
        self.long_context_model = os.getenv("QWEN_LONG_CONTEXT_MODEL", "qwen-max-longcontext")
        self.enable_search = os.getenv("QWEN_ENABLE_SEARCH", "1") in ("1", "true", "yes")

        # 🔥 v0.17: 解析 provider:model 语法
        self.provider = "deepseek"
        self.model = self._model_str
        self._parse_model_string(self._model_str)
        self._is_deepseek = "deepseek" in (self.provider or "")
        self._is_qwen = self.provider == "qwen"

        # 🔥 v0.17: 按需构建后端 client
        self.client = None
        self._build_client()

        # 兼容旧属性（health check 等用到）
        self._extra_body: dict = {}
        self._cfg = DEEPSEEK_DEFAULTS if self._is_deepseek else QWEN_DEFAULTS

    def _parse_model_string(self, model_str: str):
        """解析 'deepseek:deepseek-chat' → provider='deepseek', model='deepseek-chat'
        裸 'gpt-5.6-sol' → provider='openai', model='gpt-5.6-sol'
        """
        if not model_str:
            return
        if ":" in model_str:
            parts = model_str.split(":", 1)
            candidate = parts[0].lower()
            if candidate in self.PROVIDERS:
                self.provider = candidate
                self.model = parts[1]
                return
        # 无前缀或未知前缀 → 按 base_url 推断
        if "dashscope" in self.base_url or "qwen" in self.base_url.lower():
            self.provider = "qwen"
        elif "deepseek" in self.base_url.lower():
            self.provider = "deepseek"
        elif "openai" in self.base_url.lower() or not self.base_url:
            self.provider = "openai"
        else:
            self.provider = "deepseek"  # 默认
        self.model = model_str

    def _build_client(self):
        """🔥 v0.17: 按需构建 provider 对应的 SDK client（不预加载全部）"""
        provider_info = self.PROVIDERS.get(self.provider, ("openai", ""))
        backend_type, default_base_url = provider_info

        if backend_type == "anthropic":
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic SDK not installed, falling back to OpenAI")
                self.client = self._build_openai_client()
        else:
            self.client = self._build_openai_client()

    def _build_openai_client(self):
        """构建 OpenAI 兼容客户端"""
        from openai import OpenAI
        provider_info = self.PROVIDERS.get(self.provider, ("openai", ""))
        _, default_base_url = provider_info
        base = self.base_url or default_base_url
        # DeepSeek 特殊处理
        if "deepseek" in base.lower() and base.endswith("/v1"):
            base = base[:-3]
        return OpenAI(api_key=self.api_key, base_url=base)

    def switch_model(self, model_str: str):
        """🔥 v0.17: 运行时切换模型（不重启进程）

        Args:
            model_str: 如 'deepseek:deepseek-chat' 或 'qwen:qwen3-6-plus'
        """
        old_provider = self.provider
        self._parse_model_string(model_str)
        if self.provider != old_provider or self.client is None:
            self._build_client()  # 🔥 只切换目标 provider
        self._is_deepseek = "deepseek" in (self.provider or "")
        logger.info("🔄 Switched model: %s/%s", self.provider, self.model)
        self._is_qwen = "dashscope" in self.base_url.lower()
        self._extra_body: dict = {}
        self._cfg = DEEPSEEK_DEFAULTS if self._is_deepseek else QWEN_DEFAULTS

        # ════════════════════════════════════════════
        #  DeepSeek 深度优化
        #  ⭐ 参考: https://api-docs.deepseek.com/api/create-chat-completion
        # ════════════════════════════════════════════
        if self._is_deepseek:
            # 1. thinking_mode: flash 默认 = non-thinking（最快）
            #    - 需要推理时在调用时传 thinking_mode: "thinking"
            #    - thinking_max 消耗最多 token
            #    官方文档说 flash 默认 non-thinking，不需要显式设置

            # 2. stream_options: 流式调用时获取 token 用量
            #    在 chat_stream 中动态添加

            logger.info("🚀 DeepSeek 深度优化已加载: 模型=%s, max_tokens=%d, ctx=1M, 429重试=%d",
                        self.model, self._cfg["max_tokens"],
                        self._cfg["retry_max_429"])

        # ════════════════════════════════════════════
        #  Qwen 优化（已有）
        # ════════════════════════════════════════════
        elif self._is_qwen:
            if self.enable_search:
                self._extra_body["enable_search"] = True
                logger.info("🔍 Qwen 内置搜索已开启")

        # ⭐ DeepSeek thinking 模式每次调用可长达 60s+
        # timeout 太短 → 频繁超时重试 → event loop 阻塞 → 浏览器超时
        _timeout = 120.0 if self._is_deepseek else 60.0
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=_timeout)

        # ── Independent embedding client ──
        _ek = os.getenv("EMBEDDING_API_KEY", "")
        _eb = os.getenv("EMBEDDING_BASE_URL", "")
        if _ek:
            self._embed_client = OpenAI(api_key=_ek, base_url=_eb or self.base_url, timeout=_timeout)
            logger.info("📐 独立 embedding: %s", _eb or "主客户端")
        else:
            self._embed_client = self.client

        # ── Warm-up ──
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1, temperature=0,
            )
            self._warmed_up = True
            logger.info("🔥 LLM 连接预热完成 (%s)", self.model)
        except Exception:
            logger.info("LLM 连接预热失败（不影响使用）")

    # ════════════════════════════════════════════
    #  Embeddings
    # ════════════════════════════════════════════

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector."""
        if self._is_deepseek:
            # DeepSeek 不支持 embedding — 留空即可
            logger.warning("DeepSeek 不支持 embedding — 返回空向量")
            return [0.0] * 384
        resp = _api_call_with_retry(
            lambda: self._embed_client.embeddings.create(
                model=self.embedding_model, input=text
            )
        )
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding generation."""
        if self._is_deepseek:
            return [[0.0] * 384 for _ in texts]
        resp = _api_call_with_retry(
            lambda: self._embed_client.embeddings.create(
                model=self.embedding_model, input=texts
            )
        )
        return [d.embedding for d in resp.data]

    # ════════════════════════════════════════════
    #  Chat — 根据不同场景调不同参数
    # ════════════════════════════════════════════

    def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        enable_search: bool | None = None,
        enable_thinking: bool | None = None,
    ) -> str:
        """通用聊天。根据提供的内容自动选择合适参数。

        Args:
            enable_search: None=跟随全局默认, True=开启联网搜索, False=关闭
            enable_thinking: None=不显式设置, True=开启思考链, False=关闭
        """
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        extra = dict(self._extra_body)

        # 场景感知搜索（仅 Qwen）
        combined_text = " ".join(m.get("content", "") for m in full_messages if m.get("content"))
        _needs_search = any(kw in combined_text for kw in ["搜", "查", "搜索", "最新", "今天", "明天", "新闻", "天气", "价格"])
        if enable_search is True or (enable_search is None and _needs_search and self._is_qwen):
            extra["enable_search"] = True

        # Qwen: thinking 链
        if enable_thinking is not None and self._is_qwen:
            extra["enable_thinking"] = enable_thinking
            if enable_thinking:
                extra.setdefault("thinking_budget", 1024)

        # ⭐ DeepSeek: thinking 模式（默认开启）
        # 参考: https://api-docs.deepseek.com/api/create-chat-completion
        # 参数格式: thinking: {"type": "enabled"} + reasoning_effort
        # 注意: 开启 thinking 后 temperature/top_p 失效
        if self._is_deepseek:
            _thinking = enable_thinking if enable_thinking is not None else True
            if _thinking:
                extra["thinking"] = {"type": "enabled"}
                # reasoning_effort: high=复杂任务, max=全力以赴
                extra["reasoning_effort"] = self._cfg.get("reasoning_effort", "high")
            else:
                extra["thinking"] = {"type": "disabled"}

        if temperature is None:
            temperature = self._cfg["temperature_chat"]
        if max_tokens is None:
            max_tokens = self._cfg["max_tokens"]

        resp = _api_call_with_retry(
            lambda: self.client.chat.completions.create(
                model=model or self.model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra or None,
            )
        )
        return resp.choices[0].message.content or ""

    def chat_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        tool_choice: str | dict = "auto",
    ):
        """Chat completion with tool/function calling.

        ⭐ Tool calling 用低温度（0.1），减少幻觉工具名/参数。
        ⭐ DeepSeek: tool_max_tokens 上调至 8k，支持超长工具结果。
        """
        extra = dict(self._extra_body)

        if max_tokens is None:
            max_tokens = self._cfg["tool_max_tokens"]

        kwargs = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._cfg["temperature_tool"],
            "max_tokens": max_tokens,
            "extra_body": extra or None,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        # ⭐ 工具调用：超时不必重试（DeepSeek thinking 模式本来就慢）
        # 只重试 429/5xx 这种服务器端错误
        _max_retries = 1 if self._is_deepseek else 2
        return _api_call_with_retry(
            lambda: self.client.chat.completions.create(**kwargs),
            max_retries=_max_retries,
            base_delay=self._cfg.get("retry_base_other", 1.0),
        )

    def chat_stream(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        enable_search: bool | None = None,
    ):
        """流式聊天。简单问答场景用 fast_model。

        DeepSeek: 自动加 stream_options 获取 token 用量。
        Qwen: 自动检测是否需要开启内置搜索。
        """
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        extra = dict(self._extra_body)

        # 场景感知搜索：Qwen + DeepSeek 都支持 enable_search
        if enable_search is None or enable_search is True:
            combined_text = " ".join(m.get("content", "") for m in full_messages if m.get("content"))
            if any(kw in combined_text for kw in ["搜", "查", "最新", "今天", "天气", "新闻",
                                                      "咨询", "了解", "介绍", "行情", "股价",
                                                      "动态", "热点"]):
                extra["enable_search"] = True

        # ⭐ DeepSeek: 流式默认开启思考模式
        if self._is_deepseek:
            extra["thinking"] = {"type": "enabled"}
            extra["reasoning_effort"] = self._cfg.get("reasoning_effort", "high")

        if temperature is None:
            temperature = self._cfg["temperature_chat"]

        # max_tokens: 未显式传入时使用 provider 特定默认值
        if max_tokens is None:
            max_tokens = self._cfg["stream_max_tokens"]

        # ⭐ 简单问答用 flash 模型（更快更便宜）
        use_model = model or self.model
        if model is None and system_prompt is None and len(full_messages) <= 2:
            use_model = self.fast_model

        # ⭐ DeepSeek: 流式调用时获取 token 用量统计
        kwargs = dict(
            model=use_model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra or None,
        )
        if self._is_deepseek:
            kwargs["stream_options"] = {"include_usage": True}

        stream = _api_call_with_retry(
            lambda: self.client.chat.completions.create(**kwargs)
        )
        total_tokens = 0
        for chunk in stream:
            if self._is_deepseek and hasattr(chunk, 'usage') and chunk.usage:
                total_tokens = chunk.usage.total_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue
            # 🔥 v0.17: 源头分离 — 只 yield content，不 yield reasoning_content
            # 之前做法是事后用 _strip_thinking_text() 正则过滤，总有漏网之鱼
            # 现在在源头就丢弃思考内容，前端永远收不到 reasoning
            if delta.content:
                yield delta.content
        if self._is_deepseek and total_tokens:
            logger.debug("📊 DeepSeek stream 用量: %d tokens", total_tokens)

    # ════════════════════════════════════════════
    #  JSON mode — 结构化输出专用
    # ════════════════════════════════════════════

    def chat_json(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        enable_search: bool | None = None,
    ) -> dict:
        """用 JSON mode 输出。

        Qwen: 原生 response_format 支持。
        DeepSeek: 通过 OpenAI 兼容 API 也支持 response_format。<｜end▁of▁thinking｜>_format。
        适合：记忆提取、规划、分类等需要结构化输出的场景。
        """
        extra = dict(self._extra_body)
        if temperature is None:
            temperature = self._cfg["temperature_json"]
        if max_tokens is None:
            max_tokens = self._cfg["json_max_tokens"]

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Qwen: 搜索 + JSON mode 提示
        if self._is_qwen:
            if enable_search is not None:
                extra["enable_search"] = enable_search
            if not system_prompt or "json" not in system_prompt.lower():
                if full_messages and full_messages[-1]["role"] == "user":
                    full_messages[-1] = {
                        "role": "user",
                        "content": full_messages[-1]["content"] + "\n\n请用 JSON 格式输出。"
                    }

        # DeepSeek: 也支持 OpenAI 兼容的 response_format
        try:
            resp = _api_call_with_retry(
                lambda: self.client.chat.completions.create(
                    model=model or self.model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    extra_body=extra or None,
                )
            )
            content = resp.choices[0].message.content or "{}"
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning("JSON mode returned non-JSON: %.200s", content)
                # fallthrough to manual parse
        except Exception as e:
            logger.warning("JSON mode failed, fallback to text parse: %s", str(e)[:60])

        # Fallback: 走普通 chat + 手动 parse（兼容所有模型）
        result = self.chat(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for attempt in [
            lambda: json.loads(result.strip()),
            lambda: json.loads(result.strip().removeprefix("```json").removesuffix("```").strip()),
        ]:
            try:
                return attempt()
            except (json.JSONDecodeError, AttributeError):
                continue
        return {"_raw": result, "_parse_error": True}

    # ════════════════════════════════════════════
    #  Long-context processing
    # ════════════════════════════════════════════

    def process_long_transcript(
        self,
        transcript: str,
        instruction: str = "Summarize the key points from this transcript.",
        system_prompt: str | None = None,
        temperature: float = 0.3,
    ) -> str:
        """Process very long text using long-context model (1M tokens)."""
        token_est = estimate_tokens(transcript)
        safe_max = 950000

        if token_est > safe_max:
            logger.warning("Transcript ~%d tokens exceeds safe limit", token_est)
            max_chars = int(safe_max * CHARS_PER_TOKEN)
            transcript = transcript[:max_chars]

        logger.info("Processing long text (~%d tokens) with %s", token_est, self.long_context_model)

        sys = system_prompt or (
            "你是一个精确的文档分析助手。准确提取关键信息，不要添加或推断未明确提到的细节。"
        )

        return self.chat(
            messages=[{"role": "user", "content": f"{instruction}\n\n---\n{transcript}"}],
            system_prompt=sys,
            temperature=temperature,
            max_tokens=4096,
            model=self.long_context_model,
        )

    def extract_memories_from_long_transcript(
        self,
        transcript: str,
    ) -> list[dict[str, Any]]:
        """Extract memories from a long transcript using JSON mode."""
        system = (
            "你是记忆提取系统。从文本中提取持久记忆："
            "事实、偏好、决策、目标和重要观察。"
            "返回 JSON 数组：[{type, content, confidence, tags}]。"
            "类型: fact, preference, decision, goal, observation。"
            "只提取明确陈述的信息。"
        )
        instruction = "从这段文字中提取所有重要的记忆信息。"

        try:
            resp = self.chat_json(
                messages=[{"role": "user", "content": f"{instruction}\n\n---\n{transcript}"}],
                system_prompt=system,
                temperature=0.1,
                max_tokens=4096,
            )
            if isinstance(resp, list):
                return resp
            if isinstance(resp, dict) and "_parse_error" not in resp:
                # Assume it's wrapped: {"memories": [...]}
                for key in ("memories", "data", "results"):
                    val = resp.get(key)
                    if isinstance(val, list):
                        return val
                return [resp]
        except Exception as e:
            logger.warning("Long transcript extraction failed: %s", e)

        return []

    # ════════════════════════════════════════════
    #  Memory extraction (JSON mode)
    # ════════════════════════════════════════════

    def extract_memories(
        self, conversation: list[dict]
    ) -> list[dict[str, Any]]:
        """Extract structured memory candidates using JSON mode.

        Uses chat_json for reliable structured output.
        """
        if not conversation or len(conversation) < 2:
            return []

        system = (
            "你是记忆提取系统。从对话中提取持久记忆。"
            "只提取明确陈述的信息。忽略问候、闲聊和琐碎交流。"
            "返回 JSON 数组：[{type, content, confidence, tags}]。\n"
            "类型: fact(事实), preference(偏好), decision(决策), goal(目标), observation(观察)。\n"
            "content 保持简洁（10-30字）。\n"
            "只返回 JSON 数组，不要解释。"
        )

        text = json.dumps(
            [{"role": m.get("role", ""), "content": m.get("content", "")} for m in conversation],
            ensure_ascii=False,
        )

        resp = self.chat_json(
            messages=[{"role": "user", "content": f"对话:\n{text}\n\n提取记忆:"}],
            system_prompt=system,
            temperature=0.05,
            max_tokens=2048,
        )

        # Handle various response shapes
        if isinstance(resp, list):
            valid = [c for c in resp if isinstance(c, dict) and c.get("content")]
            if valid:
                return valid
        # Maybe wrapped: {"memories": [...]}
        if isinstance(resp, dict):
            for key in ("memories", "data"):
                val = resp.get(key)
                if isinstance(val, list):
                    return [c for c in val if isinstance(c, dict) and c.get("content")]

        # Fallback: try regex
        import re
        m = re.search(r'\[.*?\]', json.dumps(resp, ensure_ascii=False), re.DOTALL)
        if m:
            try:
                candidates = json.loads(m.group())
                if isinstance(candidates, list):
                    return [c for c in candidates if isinstance(c, dict) and c.get("content")]
            except json.JSONDecodeError:
                pass

        logger.warning("Memory extraction failed: %.200s", resp)
        return []

    # ════════════════════════════════════════════
    #  Memory-augmented answer
    # ════════════════════════════════════════════

    def answer_with_memories(
        self,
        query: str,
        memories: list[dict],
        preferences: list[dict] | None = None,
        conversation_history: str = "",
        max_context_tokens: int = 8000,
    ) -> str:
        """Answer using retrieved memories as context.

        Qwen 内置搜索会自动补充实时信息。
        """
        # select_memories_for_context 定义在同一个文件顶层，直接可用
        selected = select_memories_for_context(memories, max_tokens=max_context_tokens)

        context_parts = []
        for m in selected:
            context_parts.append(
                f"[{m.get('memory_type', 'unknown')} | 可信度:{m.get('confidence', 0):.2f}] "
                f"{m.get('content', '')}"
            )

        system = (
            "你是小明，带长期记忆的 AI 助手。\n"
            "根据记忆上下文和你的知识回答问题。"
        )
        if context_parts:
            system += "\n\n## 相关记忆\n" + "\n".join(context_parts)
        if conversation_history:
            system += f"\n\n## 最近对话\n{conversation_history}"
        if preferences:
            pref_selected = select_memories_for_context(
                [{"content": p.get("content", ""), "confidence": p.get("confidence", 0.5), **p}
                 for p in preferences],
                max_tokens=2000,
            )
            if pref_selected:
                pref_text = "\n".join(f"- {p.get('content', '')}" for p in pref_selected)
                system += f"\n\n## 已知偏好\n{pref_text}"

        return self.chat(
            messages=[{"role": "user", "content": query}],
            system_prompt=system,
            temperature=0.3,
        )
