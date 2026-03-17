"""video_info_translator 契约测试。"""

import json
from types import SimpleNamespace

import pytest

from vat.llm.video_info_translator import TranslatedVideoInfo, VideoInfoTranslator


class TestTranslatedVideoInfoContracts:
    def test_roundtrip_dict_conversion(self):
        info = TranslatedVideoInfo(
            original_title="原标题",
            original_description="原简介",
            original_tags=["a"],
            title_translated="翻译标题",
            description_summary="摘要",
            description_translated="翻译简介",
            tags_translated=["标签A"],
            tags_generated=["标签B"],
            recommended_tid=21,
            recommended_tid_name="日常",
            tid_reason="reason",
        )

        restored = TranslatedVideoInfo.from_dict(info.to_dict())

        assert restored == info

    def test_from_dict_accepts_legacy_description_optimized_field(self):
        restored = TranslatedVideoInfo.from_dict({
            "original_title": "原标题",
            "original_description": "原简介",
            "original_tags": [],
            "title_translated": "翻译标题",
            "description_optimized": "旧摘要字段",
            "description_translated": "翻译简介",
            "tags_translated": [],
            "tags_generated": [],
            "recommended_tid": 21,
            "recommended_tid_name": "日常",
            "tid_reason": "reason",
        })

        assert restored.description_summary == "旧摘要字段"


class TestVideoInfoTranslatorHelpers:
    def test_get_client_is_lazy_and_reuses_cached_client(self, monkeypatch):
        translator = VideoInfoTranslator()
        created = []
        fake_client = object()
        monkeypatch.setattr(
            "vat.llm.video_info_translator.get_or_create_client",
            lambda api_key, base_url, proxy: created.append((api_key, base_url, proxy)) or fake_client,
        )

        first = translator._get_client()
        second = translator._get_client()

        assert first is fake_client
        assert second is fake_client
        assert created == [("", "", "")]

    def test_strip_uploader_prefix_removes_only_matching_prefix(self):
        translator = VideoInfoTranslator()

        assert translator._strip_uploader_prefix("【白上吹雪】杂谈直播", "白上吹雪") == "杂谈直播"

    def test_strip_uploader_prefix_removes_loose_bracket_match(self):
        translator = VideoInfoTranslator()

        assert translator._strip_uploader_prefix("[Official-白上吹雪-Ch] 杂谈直播", "白上吹雪") == "杂谈直播"

    def test_strip_uploader_prefix_preserves_non_uploader_markers(self):
        translator = VideoInfoTranslator()

        assert translator._strip_uploader_prefix("【3DLIVE】演唱会", "白上吹雪") == "【3DLIVE】演唱会"

    def test_normalize_translations_rewrites_non_mainland_terms(self):
        translator = VideoInfoTranslator()

        assert translator._normalize_translations("马力欧和寶可夢") == "马里奥和宝可梦"

    def test_build_zones_info_groups_subzones(self):
        translator = VideoInfoTranslator()

        zones_info = translator._build_zones_info()

        assert "【生活区】" in zones_info
        assert "21: 日常" in zones_info


class TestVideoInfoTranslatorTranslate:
    def test_translate_parses_json_markdown_and_normalizes_output(self, monkeypatch):
        translator = VideoInfoTranslator(model="test-model")

        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    message=SimpleNamespace(
                                        content="""```json
{
  "title_translated": "【白上吹雪】马力欧与寶可夢",
  "description_summary": "摘要",
  "description_translated": "马力欧的简介",
  "tags_translated": ["寶可夢"],
  "tags_generated": ["VTuber"],
  "recommended_tid": 21,
  "recommended_tid_name": "日常",
  "tid_reason": "reason"
}
```"""
                                    )
                                )
                            ]
                        )

        monkeypatch.setattr(translator, "_get_client", lambda: _FakeClient())

        result = translator.translate(
            title="原标题",
            description="原简介",
            tags=["tag1"],
            uploader="白上吹雪",
        )

        assert result.title_translated == "马里奥与宝可梦"
        assert result.description_translated == "马里奥的简介"
        assert result.tags_generated == ["VTuber"]

    def test_translate_raises_when_title_empty(self):
        translator = VideoInfoTranslator(model="test-model")

        try:
            translator.translate(title="", description="desc", tags=[], uploader="主播")
        except ValueError as exc:
            assert "title 为空" in str(exc)
        else:
            raise AssertionError("translate 应在 title 为空时抛错")

    def test_translate_retries_json_decode_error_then_succeeds(self, monkeypatch):
        translator = VideoInfoTranslator(model="test-model")
        responses = iter([
            "not json",
            """```json
            {
              "title_translated": "翻译标题",
              "description_summary": "摘要",
              "description_translated": "翻译简介",
              "tags_translated": [],
              "tags_generated": [],
              "recommended_tid": 21,
              "recommended_tid_name": "日常",
              "tid_reason": "reason"
            }
            ```""",
        ])
        sleep_calls = []

        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
                        )

        monkeypatch.setattr(translator, "_get_client", lambda: _FakeClient())
        monkeypatch.setattr("time.sleep", lambda seconds: sleep_calls.append(seconds))

        result = translator.translate(
            title="原标题",
            description="原简介",
            tags=[],
            uploader="主播",
        )

        assert result.title_translated == "翻译标题"
        assert sleep_calls == [1]

    def test_translate_retries_network_error_then_succeeds(self, monkeypatch):
        translator = VideoInfoTranslator(model="test-model")
        attempts = {"count": 0}
        sleep_calls = []

        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        attempts["count"] += 1
                        if attempts["count"] == 1:
                            raise RuntimeError("connection reset")
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                                "title_translated": "翻译标题",
                                "description_summary": "摘要",
                                "description_translated": "翻译简介",
                                "tags_translated": [],
                                "tags_generated": [],
                                "recommended_tid": 21,
                                "recommended_tid_name": "日常",
                                "tid_reason": "reason",
                            }, ensure_ascii=False)))]
                        )

        monkeypatch.setattr(translator, "_get_client", lambda: _FakeClient())
        monkeypatch.setattr("time.sleep", lambda seconds: sleep_calls.append(seconds))

        result = translator.translate("原标题", "原简介", [], uploader="主播")

        assert result.title_translated == "翻译标题"
        assert sleep_calls == [2]

    def test_translate_raises_when_llm_response_missing_title_translated(self, monkeypatch):
        translator = VideoInfoTranslator(model="test-model")

        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                                "description_summary": "摘要",
                                "description_translated": "翻译简介",
                            }, ensure_ascii=False)))]
                        )

        monkeypatch.setattr(translator, "_get_client", lambda: _FakeClient())

        with pytest.raises(RuntimeError, match="视频信息翻译最终失败"):
            translator.translate("原标题", "原简介", [], uploader="主播")

    def test_translate_accepts_plain_fenced_code_block(self, monkeypatch):
        translator = VideoInfoTranslator(model="test-model")

        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content="""```
{
  "title_translated": "翻译标题",
  "description_summary": "摘要",
  "description_translated": "翻译简介",
  "tags_translated": [],
  "tags_generated": [],
  "recommended_tid": 21,
  "recommended_tid_name": "日常",
  "tid_reason": "reason"
}
```"""))]
                        )

        monkeypatch.setattr(translator, "_get_client", lambda: _FakeClient())

        result = translator.translate("原标题", "原简介", [], uploader="主播")

        assert result.title_translated == "翻译标题"

    def test_translate_uses_default_tid_fallback_when_llm_omits_tid_fields(self, monkeypatch):
        translator = VideoInfoTranslator(model="test-model")

        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                                "title_translated": "翻译标题",
                                "description_summary": "摘要",
                                "description_translated": "翻译简介",
                                "tags_translated": [],
                                "tags_generated": [],
                            }, ensure_ascii=False)))]
                        )

        monkeypatch.setattr(translator, "_get_client", lambda: _FakeClient())

        result = translator.translate("原标题", "原简介", [], uploader="主播", default_tid=17)

        assert result.recommended_tid == 17
        assert result.recommended_tid_name == "单机游戏"
        assert result.tid_reason == "默认分区"

    def test_translate_does_not_retry_non_network_non_json_error(self, monkeypatch):
        translator = VideoInfoTranslator(model="test-model")
        attempts = {"count": 0}

        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        attempts["count"] += 1
                        raise RuntimeError("quota exceeded")

        monkeypatch.setattr(translator, "_get_client", lambda: _FakeClient())

        with pytest.raises(RuntimeError, match="视频信息翻译最终失败"):
            translator.translate("原标题", "原简介", [], uploader="主播")

        assert attempts["count"] == 1

    def test_translate_allows_empty_uploader_and_still_returns_result(self, monkeypatch):
        translator = VideoInfoTranslator(model="test-model")

        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        return SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                                "title_translated": "翻译标题",
                                "description_summary": "摘要",
                                "description_translated": "翻译简介",
                                "tags_translated": [],
                                "tags_generated": [],
                                "recommended_tid": 21,
                                "recommended_tid_name": "日常",
                                "tid_reason": "reason",
                            }, ensure_ascii=False)))]
                        )

        monkeypatch.setattr(translator, "_get_client", lambda: _FakeClient())

        result = translator.translate("原标题", "原简介", [], uploader="")

        assert result.title_translated == "翻译标题"
