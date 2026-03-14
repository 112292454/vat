"""翻译契约测试。"""

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
