"""Disk cache utility for API responses and computation results.

通过 config 中 storage.cache_enabled 控制是否启用 diskcache。
禁用时（默认）：所有 cache 操作直接跳过，零 SQLite 依赖，零性能开销。
启用时：使用 diskcache（基于 SQLite）持久化缓存 LLM/翻译结果，
        重跑同一视频可复用已缓存结果。高并发时可能出现 SQLite 锁冲突。
"""

import functools
import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, is_dataclass
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# diskcache 内部使用 SQLite，高并发场景需要足够长的超时
_CACHE_SQLITE_TIMEOUT = 300  # 秒（diskcache 传给 sqlite3.connect(timeout=...) 的值）

# 全局状态
_cache_enabled = False  # 默认关闭，由 init_caches(enabled=True) 显式启用
_cache_initialized = False
_cache_init_lock = threading.Lock()

# Cache instances (None when disabled)
_llm_cache = None
_asr_cache = None
_tts_cache = None
_translate_cache = None
_version_state_cache = None


def init_caches(cache_dir: str, enabled: bool = False) -> None:
    """初始化缓存系统。
    
    Args:
        cache_dir: 缓存目录路径
        enabled: 是否启用 diskcache。False 时不创建任何 Cache 实例。
    
    幂等+线程安全：多线程调用时只执行一次。
    """
    global _cache_enabled, _cache_initialized
    global _llm_cache, _asr_cache, _tts_cache, _translate_cache, _version_state_cache
    
    # 快速路径：已初始化过
    if _cache_initialized:
        return
    
    with _cache_init_lock:
        if _cache_initialized:
            return
        
        _cache_enabled = enabled
        
        if not enabled:
            logger.debug("Cache 已禁用（storage.cache_enabled=false），跳过 diskcache 初始化")
            _cache_initialized = True
            return
        
        # 延迟导入 diskcache（禁用时完全不需要）
        from diskcache import Cache
        
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Close existing caches if any
        for c in [_llm_cache, _asr_cache, _tts_cache, _translate_cache, _version_state_cache]:
            if c:
                try:
                    c.close()
                except Exception:
                    pass
                
        _llm_cache = Cache(str(cache_path / "llm_translation"), timeout=_CACHE_SQLITE_TIMEOUT)
        _asr_cache = Cache(str(cache_path / "asr_results"), tag_index=True, timeout=_CACHE_SQLITE_TIMEOUT)
        _tts_cache = Cache(str(cache_path / "tts_audio"), timeout=_CACHE_SQLITE_TIMEOUT)
        _translate_cache = Cache(str(cache_path / "translate_results"), timeout=_CACHE_SQLITE_TIMEOUT)
        _version_state_cache = Cache(str(cache_path / "version_state"), timeout=_CACHE_SQLITE_TIMEOUT)
        
        logger.info(f"Cache 已启用，目录: {cache_path}")
        _cache_initialized = True


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


def get_llm_cache():
    """Get LLM translation cache instance (None if disabled)."""
    return _llm_cache


def get_asr_cache():
    """Get ASR results cache instance (None if disabled)."""
    return _asr_cache


def get_translate_cache():
    """Get translate cache instance (None if disabled)."""
    return _translate_cache


def get_tts_cache():
    """Get TTS audio cache instance (None if disabled)."""
    return _tts_cache


def get_version_state_cache():
    """Get version check state cache instance (None if disabled)."""
    return _version_state_cache


def memoize(cache_instance_getter, **kwargs):
    """Decorator to cache function results with global switch support.

    Cache 禁用时直接调用原函数，零开销。
    Cache 启用但 SQLite 失败时自动 fallback 到直接调用。

    Args:
        cache_instance_getter: Function that returns a Cache instance (e.g., get_llm_cache)
        **kwargs: Arguments passed to cache.memoize() (expire, typed, etc.)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            if not _cache_enabled:
                return func(*args, **kw)
            
            cache_instance = cache_instance_getter()
            if cache_instance is None:
                return func(*args, **kw)
            
            try:
                return cache_instance.memoize(**kwargs)(func)(*args, **kw)
            except sqlite3.OperationalError as cache_err:
                logger.warning(
                    f"Cache SQLite 错误，fallback 到直接调用 {func.__name__}: {cache_err}"
                )
                return func(*args, **kw)

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
