"""cache 工具契约测试。"""

import sqlite3
from dataclasses import dataclass

from vat.utils import cache as cache_module


@dataclass
class _DemoData:
    value: int


class TestGenerateCacheKeyContracts:
    def test_generate_cache_key_handles_dataclass_and_dict_order_stably(self):
        key_a = cache_module.generate_cache_key({"a": 1, "b": _DemoData(value=2)})
        key_b = cache_module.generate_cache_key({"b": _DemoData(value=2), "a": 1})

        assert key_a == key_b


class TestMemoizeContracts:
    def test_memoize_calls_original_when_cache_disabled(self, monkeypatch):
        monkeypatch.setattr(cache_module, "_cache_enabled", False)
        calls = []

        @cache_module.memoize(lambda: None)
        def func(x):
            calls.append(x)
            return x * 2

        assert func(3) == 6
        assert calls == [3]

    def test_memoize_falls_back_when_cache_memoize_raises_sqlite_error(self, monkeypatch):
        monkeypatch.setattr(cache_module, "_cache_enabled", True)

        class _BrokenCache:
            def memoize(self, **kwargs):
                raise sqlite3.OperationalError("database locked")

        calls = []

        @cache_module.memoize(lambda: _BrokenCache())
        def func(x):
            calls.append(x)
            return x + 1

        assert func(4) == 5
        assert calls == [4]


class TestCacheInitAndSwitchContracts:
    def test_enable_disable_cache_switches_flag(self, monkeypatch):
        monkeypatch.setattr(cache_module, "_cache_enabled", False)

        cache_module.enable_cache()
        assert cache_module.is_cache_enabled() is True

        cache_module.disable_cache()
        assert cache_module.is_cache_enabled() is False

    def test_init_caches_disabled_sets_initialized_without_creating_instances(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cache_module, "_cache_initialized", False)
        monkeypatch.setattr(cache_module, "_cache_enabled", False)
        monkeypatch.setattr(cache_module, "_llm_cache", None)
        monkeypatch.setattr(cache_module, "_translate_cache", None)

        cache_module.init_caches(str(tmp_path), enabled=False)

        assert cache_module._cache_initialized is True
        assert cache_module._cache_enabled is False
        assert cache_module.get_llm_cache() is None
        assert cache_module.get_translate_cache() is None

    def test_init_caches_enabled_creates_named_cache_instances(self, monkeypatch, tmp_path):
        created = []

        class _FakeCache:
            def __init__(self, path, **kwargs):
                created.append(path)

            def close(self):
                return None

        monkeypatch.setattr(cache_module, "_cache_initialized", False)
        monkeypatch.setattr(cache_module, "_cache_enabled", False)
        monkeypatch.setattr(cache_module, "_llm_cache", None)
        monkeypatch.setattr(cache_module, "_asr_cache", None)
        monkeypatch.setattr(cache_module, "_tts_cache", None)
        monkeypatch.setattr(cache_module, "_translate_cache", None)
        monkeypatch.setattr(cache_module, "_version_state_cache", None)

        import sys
        import types

        fake_diskcache = types.SimpleNamespace(Cache=_FakeCache)
        monkeypatch.setitem(sys.modules, "diskcache", fake_diskcache)

        cache_module.init_caches(str(tmp_path), enabled=True)

        assert cache_module._cache_initialized is True
        assert cache_module._cache_enabled is True
        assert len(created) == 5
        assert cache_module.get_translate_cache() is not None

    def test_getters_return_none_when_uninitialized(self, monkeypatch):
        monkeypatch.setattr(cache_module, "_llm_cache", None)
        monkeypatch.setattr(cache_module, "_asr_cache", None)
        monkeypatch.setattr(cache_module, "_tts_cache", None)
        monkeypatch.setattr(cache_module, "_translate_cache", None)
        monkeypatch.setattr(cache_module, "_version_state_cache", None)

        assert cache_module.get_llm_cache() is None
        assert cache_module.get_asr_cache() is None
        assert cache_module.get_tts_cache() is None
        assert cache_module.get_translate_cache() is None
        assert cache_module.get_version_state_cache() is None
