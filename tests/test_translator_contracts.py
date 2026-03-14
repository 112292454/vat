"""翻译契约测试。"""

import sqlite3
from types import SimpleNamespace

import pytest

from vat.asr.asr_data import ASRDataSeg


@pytest.fixture
def translator():
    from vat.translator.llm_translator import LLMTranslator
    from vat.translator.types import TargetLanguage

    return LLMTranslator(
        thread_num=1,
        batch_num=5,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        output_dir="/tmp/test_translator_contracts",
        model="test-model",
        custom_translate_prompt="",
        is_reflect=False,
        api_key="test-key",
        base_url="http://localhost:1234/v1",
    )


class TestSetSegmentsTranslatedTextContracts:
    def test_sets_translated_text_by_index(self, translator):
        segments = [
            ASRDataSeg(text="a", start_time=0, end_time=100),
            ASRDataSeg(text="b", start_time=100, end_time=200),
        ]
        translated = [
            SimpleNamespace(index=2, translated_text="乙"),
            SimpleNamespace(index=1, translated_text="甲"),
        ]

        result = translator._set_segments_translated_text(segments, translated)

        assert result[0].translated_text == "甲"
        assert result[1].translated_text == "乙"

    def test_missing_translation_raises_instead_of_silent_partial_result(self, translator):
        segments = [
            ASRDataSeg(text="a", start_time=0, end_time=100),
            ASRDataSeg(text="b", start_time=100, end_time=200),
            ASRDataSeg(text="c", start_time=200, end_time=300),
        ]
        translated = [
            SimpleNamespace(index=1, translated_text="甲"),
            SimpleNamespace(index=3, translated_text="丙"),
        ]

        with pytest.raises(RuntimeError, match="缺少翻译"):
            translator._set_segments_translated_text(segments, translated)


class TestTranslatorCacheKeyContracts:
    def _make_chunk(self):
        from vat.translator.base import SubtitleProcessData

        return [
            SubtitleProcessData(index=1, original_text="おはよう"),
            SubtitleProcessData(index=2, original_text="こんにちは"),
        ]

    def test_custom_prompt_changes_cache_key(self, translator):
        chunk = self._make_chunk()
        base_key = translator._get_cache_key(chunk)

        translator.custom_prompt = "speaker=fubuki"
        changed_key = translator._get_cache_key(chunk)

        assert changed_key != base_key

    def test_reflect_mode_changes_cache_key(self, translator):
        chunk = self._make_chunk()
        base_key = translator._get_cache_key(chunk)

        translator.is_reflect = True
        changed_key = translator._get_cache_key(chunk)

        assert changed_key != base_key

    def test_context_payload_changes_cache_key_when_context_enabled(self, translator):
        chunk = self._make_chunk()
        translator.enable_context = True
        translator._previous_batch_result = {"1": "上一句"}
        key_a = translator._get_cache_key(chunk)

        translator._previous_batch_result = {"1": "另一句"}
        key_b = translator._get_cache_key(chunk)

        assert key_b != key_a


class TestSafeTranslateChunkContracts:
    def _make_chunk(self):
        from vat.translator.base import SubtitleProcessData

        return [SubtitleProcessData(index=1, original_text="おはよう")]

    def test_cache_hit_returns_cached_result_without_calling_translate(self, translator, monkeypatch):
        chunk = self._make_chunk()
        cached = [SimpleNamespace(index=1, translated_text="缓存命中")]

        class FakeCache:
            def get(self, key, default=None):
                return cached

        translator._cache = FakeCache()
        monkeypatch.setattr("vat.translator.base.is_cache_enabled", lambda: True)
        translator._translate_chunk = lambda _chunk: (_ for _ in ()).throw(AssertionError("should not translate"))

        result = translator._safe_translate_chunk(chunk)

        assert result is cached
        assert translator._cache_hit_count == 1

    def test_cache_miss_writes_result_and_calls_update_callback(self, translator, monkeypatch):
        chunk = self._make_chunk()
        writes = []
        updates = []

        class FakeCache:
            def get(self, key, default=None):
                return None

            def set(self, key, value, expire=None):
                writes.append((key, value, expire))

        expected = [SimpleNamespace(index=1, translated_text="翻译结果")]
        translator._cache = FakeCache()
        translator.update_callback = lambda result: updates.append(result)
        translator._translate_chunk = lambda _chunk: expected
        monkeypatch.setattr("vat.translator.base.is_cache_enabled", lambda: True)

        result = translator._safe_translate_chunk(chunk)

        assert result is expected
        assert updates == [expected]
        assert len(writes) == 1
        assert writes[0][1] is expected

    def test_sqlite_error_on_cache_get_falls_back_to_translate(self, translator, monkeypatch):
        chunk = self._make_chunk()

        class FakeCache:
            def get(self, key, default=None):
                raise sqlite3.OperationalError("database is locked")

            def set(self, key, value, expire=None):
                return None

        expected = [SimpleNamespace(index=1, translated_text="回退结果")]
        translator._cache = FakeCache()
        translator._translate_chunk = lambda _chunk: expected
        monkeypatch.setattr("vat.translator.base.is_cache_enabled", lambda: True)

        result = translator._safe_translate_chunk(chunk)

        assert result is expected

    def test_sqlite_error_on_cache_set_does_not_fail_translation(self, translator, monkeypatch):
        chunk = self._make_chunk()

        class FakeCache:
            def get(self, key, default=None):
                return None

            def set(self, key, value, expire=None):
                raise sqlite3.OperationalError("database is locked")

        expected = [SimpleNamespace(index=1, translated_text="写缓存失败也返回")]
        translator._cache = FakeCache()
        translator._translate_chunk = lambda _chunk: expected
        monkeypatch.setattr("vat.translator.base.is_cache_enabled", lambda: True)

        result = translator._safe_translate_chunk(chunk)

        assert result is expected
