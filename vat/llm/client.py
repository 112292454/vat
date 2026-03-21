"""Unified LLM client for the application."""

import os
import time
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import google.auth
import httpx
import openai
import requests
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
from openai import OpenAI
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from vat.utils.cache import get_llm_cache, memoize
from vat.utils.logger import setup_logger

_global_client: Optional[OpenAI] = None
_client_lock = threading.Lock()

# Per-config client registry: (normalized_base_url, api_key) -> OpenAI
_client_registry: Dict[Tuple[str, str], OpenAI] = {}
_registry_lock = threading.Lock()

# Vertex ADC access token cache: cache_key -> (token, expiry_ts)
_vertex_token_cache: Dict[str, Tuple[str, Optional[float]]] = {}
_vertex_token_lock = threading.Lock()
_VERTEX_TOKEN_REFRESH_SKEW_SEC = 120
_VERTEX_TOKEN_REFRESH_RETRY_DELAYS_SEC = [1, 2]

logger = setup_logger("llm_client")


def _get_env_provider() -> str:
    return os.getenv("VAT_LLM_PROVIDER", "openai_compatible").strip() or "openai_compatible"


def _get_env_vertex_location() -> str:
    return os.getenv("VAT_VERTEX_LOCATION", "global").strip() or "global"


def _get_env_vertex_auth_mode() -> str:
    return os.getenv("VAT_VERTEX_AUTH_MODE", "api_key").strip() or "api_key"


def _get_env_vertex_project_id() -> str:
    return os.getenv("VAT_VERTEX_PROJECT_ID", "").strip()


def _get_env_vertex_credentials() -> str:
    return os.getenv("VAT_VERTEX_CREDENTIALS", "").strip()


def _resolve_provider(base_url: str = "") -> str:
    if base_url.strip():
        return "openai_compatible"
    return _get_env_provider()


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    raise ValueError(f"当前仅支持字符串消息内容，实际类型: {type(content).__name__}")


def _build_vertex_request(messages: List[dict], temperature: float, **kwargs: Any) -> Dict[str, Any]:
    system_parts: List[Dict[str, str]] = []
    contents: List[Dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "user")
        text = _extract_message_text(message.get("content", ""))
        if not text:
            continue
        if role == "system":
            system_parts.append({"text": text})
            continue
        vertex_role = "model" if role == "assistant" else "user"
        contents.append({"role": vertex_role, "parts": [{"text": text}]})

    request_body: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature},
    }
    if system_parts:
        request_body["systemInstruction"] = {"parts": system_parts}

    max_tokens = kwargs.get("max_tokens")
    if max_tokens is not None:
        request_body["generationConfig"]["maxOutputTokens"] = max_tokens

    return request_body


def _adapt_vertex_response(response_json: Dict[str, Any]) -> Any:
    candidates = response_json.get("candidates") or []
    content = ""
    if candidates:
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
        texts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        content = "".join(texts)

    prompt_feedback = response_json.get("promptFeedback") or {}
    diagnostics = {
        "candidate_count": len(candidates),
        "finish_reasons": [
            candidate.get("finishReason", "")
            for candidate in candidates
            if isinstance(candidate, dict)
        ],
        "text_part_counts": [
            len(
                [
                    part for part in (((candidate or {}).get("content") or {}).get("parts") or [])
                    if isinstance(part, dict) and part.get("text")
                ]
            )
            for candidate in candidates
            if isinstance(candidate, dict)
        ],
        "block_reason": prompt_feedback.get("blockReason", ""),
        "block_reason_message": prompt_feedback.get("blockReasonMessage", ""),
    }

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ],
        _vertex_diagnostics=diagnostics,
    )


def _raise_vertex_http_error(exc: httpx.HTTPStatusError) -> None:
    response = exc.response
    status_code = response.status_code
    try:
        body = response.json()
    except Exception:
        body = {"error": {"message": response.text}}

    message = ((body.get("error") or {}).get("message")) or response.text or "Vertex API request failed"

    if status_code == 400:
        raise openai.BadRequestError(message=message, response=response, body=body) from exc
    if status_code == 401:
        raise openai.AuthenticationError(message=message, response=response, body=body) from exc
    if status_code == 404:
        raise openai.NotFoundError(message=message, response=response, body=body) from exc
    if status_code == 429:
        raise openai.RateLimitError(message=message, response=response, body=body) from exc
    raise RuntimeError(f"Vertex API 请求失败({status_code}): {message}") from exc


def _resolve_vertex_credentials_path(credentials_path: str = "") -> str:
    raw_path = credentials_path or _get_env_vertex_credentials()
    if not raw_path:
        return ""

    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    if not candidate.is_absolute():
        home_candidate = Path.home() / raw_path
        if home_candidate.exists():
            raise ValueError(
                f"Vertex ADC 凭据路径不存在: {raw_path}。"
                f"检测到用户目录下存在可用文件: {home_candidate}"
            )

    raise ValueError(f"Vertex ADC 凭据路径不存在: {raw_path}")


def _get_vertex_token_cache_key(resolved_path: str) -> str:
    return resolved_path or "__adc_default__"


def _vertex_token_is_fresh(expiry_ts: Optional[float]) -> bool:
    if expiry_ts is None:
        return False
    now_ts = datetime.now(timezone.utc).timestamp()
    return expiry_ts - now_ts > _VERTEX_TOKEN_REFRESH_SKEW_SEC


def _get_vertex_access_token(credentials_path: str = "", proxy: str = "") -> str:
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    resolved_path = _resolve_vertex_credentials_path(credentials_path)
    cache_key = _get_vertex_token_cache_key(resolved_path)

    with _vertex_token_lock:
        cached = _vertex_token_cache.get(cache_key)
        if cached:
            cached_token, cached_expiry_ts = cached
            if cached_token and _vertex_token_is_fresh(cached_expiry_ts):
                return cached_token

        def _load_credentials():
            if resolved_path:
                return service_account.Credentials.from_service_account_file(
                    resolved_path,
                    scopes=scopes,
                )
            credentials, _ = google.auth.default(scopes=scopes)
            return credentials

        def _refresh_with_proxy(refresh_proxy: Optional[str]):
            credentials = _load_credentials()
            request_kwargs = {}
            if refresh_proxy is None:
                session = requests.Session()
                session.trust_env = False
                request_kwargs["session"] = session
            elif refresh_proxy:
                session = requests.Session()
                session.proxies.update({"http": refresh_proxy, "https": refresh_proxy})
                request_kwargs["session"] = session
            credentials.refresh(GoogleAuthRequest(**request_kwargs))
            return credentials

        last_exc: Optional[Exception] = None
        refresh_proxies: List[Optional[str]] = [proxy] if proxy else [""]
        if proxy:
            refresh_proxies.append(None)

        for refresh_proxy in refresh_proxies:
            for attempt in range(len(_VERTEX_TOKEN_REFRESH_RETRY_DELAYS_SEC) + 1):
                try:
                    if proxy and not refresh_proxy:
                        logger.warning(
                            f"Vertex ADC token refresh via proxy failed, retry without proxy: {last_exc}"
                        )
                    credentials = _refresh_with_proxy(refresh_proxy)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt >= len(_VERTEX_TOKEN_REFRESH_RETRY_DELAYS_SEC):
                        continue
                    time.sleep(_VERTEX_TOKEN_REFRESH_RETRY_DELAYS_SEC[attempt])
            if last_exc is None:
                break

        if last_exc is not None:
            raise RuntimeError(f"Vertex ADC 认证失败: {last_exc}") from last_exc

        token = getattr(credentials, "token", "")
        if not token:
            raise RuntimeError("Vertex ADC 认证失败: 未获取到 access token")

        expiry = getattr(credentials, "expiry", None)
        expiry_ts: Optional[float] = None
        if expiry is not None:
            try:
                expiry_ts = expiry.timestamp()
            except Exception:
                expiry_ts = None

        _vertex_token_cache[cache_key] = (token, expiry_ts)
        return token


def _call_vertex_native(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    api_key: str = "",
    proxy: str = "",
    **kwargs: Any,
) -> Any:
    effective_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    location = _get_env_vertex_location()
    auth_mode = _get_env_vertex_auth_mode()
    project_id = _get_env_vertex_project_id()

    request_body = _build_vertex_request(messages, temperature, **kwargs)
    headers = {"Content-Type": "application/json"}

    if auth_mode == "adc":
        if not project_id:
            raise ValueError("Vertex LLM 配置不完整: project_id=缺失")
        if not location:
            raise ValueError("Vertex LLM 配置不完整: location=缺失")
        headers["Authorization"] = f"Bearer {_get_vertex_access_token(proxy=proxy)}"
        endpoint = (
            "https://aiplatform.googleapis.com/v1/projects/"
            f"{project_id}/locations/{location}/publishers/google/models/{model}:generateContent"
        )
    else:
        if not effective_key:
            raise ValueError("Vertex LLM 配置不完整: api_key=缺失")
        endpoint = (
            "https://aiplatform.googleapis.com/v1/publishers/google/models/"
            f"{model}:generateContent?key={effective_key}"
        )

    try:
        response = httpx.post(
            endpoint,
            json=request_body,
            headers=headers,
            timeout=openai.DEFAULT_TIMEOUT,
            proxy=proxy or None,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        _raise_vertex_http_error(exc)
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Vertex API 网络请求失败: {exc}") from exc

    return _adapt_vertex_response(response.json())


def normalize_base_url(base_url: str) -> str:
    """Normalize API base URL by ensuring /v1 suffix when needed.

    Handles various edge cases:
    - Removes leading/trailing whitespace
    - Only adds /v1 if domain has no path, or path is empty/root
    - Removes trailing slashes from /v1 (e.g., /v1/ -> /v1)
    - Preserves custom paths (e.g., /custom stays as /custom)

    Args:
        base_url: Raw base URL string

    Returns:
        Normalized base URL

    Examples:
        >>> normalize_base_url("https://api.openai.com")
        'https://api.openai.com/v1'
        >>> normalize_base_url("https://api.openai.com/v1/")
        'https://api.openai.com/v1'
        >>> normalize_base_url("https://api.openai.com/custom")
        'https://api.openai.com/custom'
        >>> normalize_base_url("  https://api.openai.com  ")
        'https://api.openai.com/v1'
    """
    url = base_url.strip()
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    if not path:
        path = "/v1"

    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )

    return normalized


def get_llm_client() -> Any:
    """Get global LLM client instance (thread-safe singleton).

    Returns:
        Global OpenAI client instance

    Raises:
        ValueError: If OPENAI_BASE_URL or OPENAI_API_KEY env vars not set
    """
    global _global_client

    provider = _get_env_provider()
    if provider != "openai_compatible":
        return None

    if _global_client is None:
        with _client_lock:
            # Double-check locking pattern
            if _global_client is None:
                base_url = os.getenv("OPENAI_BASE_URL", "").strip()
                base_url = normalize_base_url(base_url)
                api_key = os.getenv("OPENAI_API_KEY", "").strip()
                logger.debug(f"OPENAI_BASE_URL: {base_url}, OPENAI_API_KEY: {api_key}")

                if not base_url or not api_key:
                    raise ValueError(
                        "OPENAI_BASE_URL and OPENAI_API_KEY environment variables must be set"
                    )

                _global_client = OpenAI(base_url=base_url, api_key=api_key)

    return _global_client


def get_or_create_client(api_key: str = "", base_url: str = "", proxy: str = "") -> Any:
    """Get or create an OpenAI client for specific credentials.
    
    If both api_key and base_url are empty and no proxy override, returns the global client.
    For missing params, falls back to the corresponding environment variable.
    
    Args:
        api_key: API key (empty string = use global)
        base_url: Base URL (empty string = use global)
        proxy: Proxy URL override for this client (empty string = use env vars)
    
    Returns:
        OpenAI client instance (cached per unique credentials + proxy)
    """
    provider = _resolve_provider(base_url)

    if provider != "openai_compatible":
        return None

    if not api_key and not base_url and not proxy:
        return get_llm_client()
    
    # Fill missing with global env vars
    effective_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    effective_url = base_url or os.getenv("OPENAI_BASE_URL", "").strip()
    effective_url = normalize_base_url(effective_url)
    
    if not effective_key or not effective_url:
        raise ValueError(
            f"LLM 配置不完整: api_key={'设置' if effective_key else '缺失'}, "
            f"base_url={'设置' if effective_url else '缺失'}"
        )
    
    # proxy 为空时不纳入 registry_key（使用环境变量，与无 proxy 等效）
    registry_key = (effective_url, effective_key, proxy)
    
    if registry_key not in _client_registry:
        with _registry_lock:
            if registry_key not in _client_registry:
                client_kwargs = dict(base_url=effective_url, api_key=effective_key)
                if proxy:
                    # 显式指定代理，覆盖环境变量
                    # 必须传入与 OpenAI SDK 一致的 timeout 和 limits，
                    # 否则 httpx 裸默认值（pool_timeout=5s）会导致高并发下 PoolTimeout
                    client_kwargs["http_client"] = httpx.Client(
                        proxy=proxy,
                        timeout=openai.DEFAULT_TIMEOUT,
                        limits=openai.DEFAULT_CONNECTION_LIMITS,
                    )
                    logger.debug(f"Created LLM client for: {effective_url} (proxy: {proxy})")
                else:
                    logger.debug(f"Created LLM client for: {effective_url}")
                _client_registry[registry_key] = OpenAI(**client_kwargs)
    
    return _client_registry[registry_key]


def before_sleep_log(retry_state: RetryCallState) -> None:
    logger.warning(
        "Rate Limit Error, sleeping and retrying... Please lower your thread concurrency or use better OpenAI API."
    )


@memoize(get_llm_cache, expire=3600, typed=True)
@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type(openai.RateLimitError),
    before_sleep=before_sleep_log,
)
def call_llm(
    messages: List[dict],
    model: str,
    temperature: float = 1,
    api_key: str = "",
    base_url: str = "",
    proxy: str = "",
    **kwargs: Any,
) -> Any:
    """Call LLM API with automatic caching.

    Args:
        messages: Chat messages list
        model: Model name
        temperature: Sampling temperature
        api_key: Per-call API key override (empty = use global)
        base_url: Per-call base URL override (empty = use global)
        proxy: Per-call proxy override (empty = use env vars)
        **kwargs: Additional parameters for API call

    Returns:
        API response object

    Raises:
        ValueError: If response is invalid (empty choices or content)
    """
    provider = _resolve_provider(base_url)
    if provider == "vertex_native":
        response = _call_vertex_native(
            messages=messages,
            model=model,
            temperature=temperature,
            api_key=api_key,
            proxy=proxy,
            **kwargs,
        )
    else:
        client = get_or_create_client(api_key, base_url, proxy)
    # logger.trace(f"Calling LLM API: {model}, {messages}, {temperature}, {kwargs}")

        # 火山引擎方舟：自动关闭 thinking 模式
        # kimi-k2.5 默认走 thinking 路径（34s/call），关闭后降至 0.7s/call
        # 实验验证 thinking 对翻译/ASR纠错无帮助，仅增加延迟
        effective_base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        if "ark.cn-beijing.volces.com" in effective_base_url:
            kwargs.setdefault("extra_body", {})
            kwargs["extra_body"].setdefault("thinking", {"type": "disabled"})

        response = client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            temperature=temperature,
            **kwargs,
        )

    # Validate response (exceptions are not cached by diskcache)
    if not (
        response
        and hasattr(response, "choices")
        and response.choices
        and len(response.choices) > 0
        and hasattr(response.choices[0], "message")
        and response.choices[0].message.content
    ):
        if provider == "vertex_native":
            diagnostics = getattr(response, "_vertex_diagnostics", {}) or {}
            diagnostic_parts = []
            block_reason = diagnostics.get("block_reason", "")
            if block_reason:
                diagnostic_parts.append(f"promptFeedback.blockReason={block_reason}")
            block_reason_message = diagnostics.get("block_reason_message", "")
            if block_reason_message:
                diagnostic_parts.append(f"promptFeedback.blockReasonMessage={block_reason_message}")
            finish_reasons = [reason for reason in diagnostics.get("finish_reasons", []) if reason]
            if finish_reasons:
                diagnostic_parts.append(f"finishReason={finish_reasons}")
            candidate_count = diagnostics.get("candidate_count")
            if candidate_count is not None:
                diagnostic_parts.append(f"candidate_count={candidate_count}")
            text_part_counts = diagnostics.get("text_part_counts", [])
            if text_part_counts:
                diagnostic_parts.append(f"text_part_counts={text_part_counts}")

            diagnostic_msg = ", ".join(diagnostic_parts) if diagnostic_parts else "no diagnostics"
            logger.warning(f"Vertex 空响应诊断: {diagnostic_msg}")
            raise ValueError(f"Invalid Vertex API response: empty choices or content ({diagnostic_msg})")

        raise ValueError("Invalid OpenAI API response: empty choices or content")

    return response
