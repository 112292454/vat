"""Disk cache utility for API responses and computation results.

This module provides a simple interface for caching using diskcache.
Can be used by translation, ASR, and other modules that need caching.
"""

import functools
import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Optional
from pathlib import Path

from diskcache import Cache

# Global cache switch
_cache_enabled = True
_cache_dir: Optional[str] = None

# Cache instances (lazy initialized)
_llm_cache: Optional[Cache] = None
_asr_cache: Optional[Cache] = None
_tts_cache: Optional[Cache] = None
_translate_cache: Optional[Cache] = None
_version_state_cache: Optional[Cache] = None


def init_caches(cache_dir: str) -> None:
    """Initialize cache instances with the specified directory."""
    global _cache_dir, _llm_cache, _asr_cache, _tts_cache, _translate_cache, _version_state_cache
    
    _cache_dir = cache_dir
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Close existing caches if any
    for c in [_llm_cache, _asr_cache, _tts_cache, _translate_cache, _version_state_cache]:
        if c:
            c.close()
            
    _llm_cache = Cache(str(cache_path / "llm_translation"))
    _asr_cache = Cache(str(cache_path / "asr_results"), tag_index=True)
    _tts_cache = Cache(str(cache_path / "tts_audio"))
    _translate_cache = Cache(str(cache_path / "translate_results"))
    _version_state_cache = Cache(str(cache_path / "version_state"))


def _ensure_cache_init():
    """Ensure caches are initialized. If not, use a default path."""
    if _llm_cache is None:
        # Fallback to a default path if not explicitly initialized
        default_cache_dir = str(Path.home() / ".vat" / "cache")
        init_caches(default_cache_dir)


def enable_cache() -> None:
    """Enable caching globally."""
    global _cache_enabled
    _cache_enabled = True


def disable_cache() -> None:
    """Disable caching globally."""
    global _cache_enabled
    _cache_enabled = False


def is_cache_enabled() -> bool:
    """Check if caching is enabled."""
    return _cache_enabled


def get_llm_cache() -> Cache:
    """Get LLM translation cache instance."""
    _ensure_cache_init()
    return _llm_cache  # type: ignore


def get_asr_cache() -> Cache:
    """Get ASR results cache instance."""
    _ensure_cache_init()
    return _asr_cache  # type: ignore


def get_translate_cache() -> Cache:
    """Get translate cache instance."""
    _ensure_cache_init()
    return _translate_cache  # type: ignore


def get_tts_cache() -> Cache:
    """Get TTS audio cache instance."""
    _ensure_cache_init()
    return _tts_cache  # type: ignore


def get_version_state_cache() -> Cache:
    """Get version check state cache instance."""
    _ensure_cache_init()
    return _version_state_cache  # type: ignore


def memoize(cache_instance_getter, **kwargs):
    """Decorator to cache function results with global switch support.

    Args:
        cache_instance_getter: Function that returns a Cache instance (e.g., get_llm_cache)
        **kwargs: Arguments passed to cache.memoize() (expire, typed, etc.)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if not _cache_enabled:
                return func(*args, **kw)
            
            # Get cache instance at runtime
            cache_instance = cache_instance_getter()
            # Use the underlying memoize implementation
            return cache_instance.memoize(**kwargs)(func)(*args, **kw)

        return wrapper

    return decorator


def generate_cache_key(data: Any) -> str:
    """Generate cache key from data (supports dataclasses, dicts, lists)."""

    def _serialize(obj: Any) -> Any:
        """Recursively serialize object to JSON-serializable format"""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)  # type: ignore
        elif isinstance(obj, list):
            return [_serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        else:
            return obj

    serialized_data = _serialize(data)
    data_str = json.dumps(serialized_data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()
