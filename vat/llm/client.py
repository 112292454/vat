"""Unified LLM client for the application."""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import openai
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

logger = setup_logger("llm_client")


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


def get_llm_client() -> OpenAI:
    """Get global LLM client instance (thread-safe singleton).

    Returns:
        Global OpenAI client instance

    Raises:
        ValueError: If OPENAI_BASE_URL or OPENAI_API_KEY env vars not set
    """
    global _global_client

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


def get_or_create_client(api_key: str = "", base_url: str = "") -> OpenAI:
    """Get or create an OpenAI client for specific credentials.
    
    If both api_key and base_url are empty, returns the global client.
    For missing params, falls back to the corresponding environment variable.
    
    Args:
        api_key: API key (empty string = use global)
        base_url: Base URL (empty string = use global)
    
    Returns:
        OpenAI client instance (cached per unique credentials)
    """
    if not api_key and not base_url:
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
    
    registry_key = (effective_url, effective_key)
    
    if registry_key not in _client_registry:
        with _registry_lock:
            if registry_key not in _client_registry:
                _client_registry[registry_key] = OpenAI(
                    base_url=effective_url, api_key=effective_key
                )
                logger.debug(f"Created LLM client for: {effective_url}")
    
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
    **kwargs: Any,
) -> Any:
    """Call LLM API with automatic caching.

    Args:
        messages: Chat messages list
        model: Model name
        temperature: Sampling temperature
        api_key: Per-call API key override (empty = use global)
        base_url: Per-call base URL override (empty = use global)
        **kwargs: Additional parameters for API call

    Returns:
        API response object

    Raises:
        ValueError: If response is invalid (empty choices or content)
    """
    client = get_or_create_client(api_key, base_url)
    # logger.trace(f"Calling LLM API: {model}, {messages}, {temperature}, {kwargs}")

    response = client.chat.completions.create(
        model=model,
        messages=messages,  # pyright: ignore[reportArgumentType]
        temperature=temperature,
        # reasoning_effort="none",
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
        raise ValueError("Invalid OpenAI API response: empty choices or content")

    return response
