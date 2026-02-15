"""翻译器错误处理测试

验证不可重试错误（地区限制、认证失败等）能正确 fail-fast，
不会静默吞掉错误导致步骤被标记为"完成"。
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

import openai


# ─── Fixtures ───────────────────────────────────────────────────────

def _make_chunk(n: int = 5, start_idx: int = 1) -> list:
    """生成测试用的字幕 chunk"""
    from vat.subtitle_utils.entities import SubtitleProcessData
    return [
        SubtitleProcessData(
            index=start_idx + i,
            original_text=f"テスト字幕 {start_idx + i}",
        )
        for i in range(n)
    ]


def _make_bad_request_error() -> openai.BadRequestError:
    """构造 BadRequestError（地区限制）"""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.headers = {}
    mock_response.json.return_value = {
        "error": {
            "code": 400,
            "message": "User location is not supported for the API use.",
            "status": "FAILED_PRECONDITION",
        }
    }
    return openai.BadRequestError(
        message="User location is not supported for the API use.",
        response=mock_response,
        body={"error": {"message": "User location is not supported"}},
    )


def _make_auth_error() -> openai.AuthenticationError:
    """构造 AuthenticationError"""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}
    return openai.AuthenticationError(
        message="Invalid API key",
        response=mock_response,
        body={"error": {"message": "Invalid API key"}},
    )


@pytest.fixture
def translator():
    """创建 LLMTranslator 实例（mock 掉实际 API 调用）"""
    from vat.translator.llm_translator import LLMTranslator
    from vat.translator.types import TargetLanguage

    t = LLMTranslator(
        thread_num=1,
        batch_num=5,
        target_language=TargetLanguage.SIMPLIFIED_CHINESE,
        output_dir="/tmp/test_translator",
        model="test-model",
        custom_translate_prompt="",
        is_reflect=False,
        api_key="test-key",
        base_url="http://localhost:1234/v1",
    )
    return t


# ─── _translate_chunk: BadRequestError fail-fast ────────────────────

class TestTranslateChunkFailFast:
    """_translate_chunk 对不可重试错误应 fail-fast，不降级到单条翻译"""

    def test_bad_request_error_raises_immediately(self, translator):
        """BadRequestError（地区限制）应立即抛出，不降级"""
        chunk = _make_chunk()

        with patch("vat.translator.llm_translator.call_llm") as mock_llm:
            mock_llm.side_effect = _make_bad_request_error()
            with pytest.raises(RuntimeError, match="API 请求被拒绝"):
                translator._translate_chunk(chunk)

    def test_bad_request_error_does_not_call_single_translate(self, translator):
        """BadRequestError 不应触发降级到 _translate_chunk_single"""
        chunk = _make_chunk()

        with patch("vat.translator.llm_translator.call_llm") as mock_llm, \
             patch.object(translator, "_translate_chunk_single") as mock_single:
            mock_llm.side_effect = _make_bad_request_error()
            with pytest.raises(RuntimeError):
                translator._translate_chunk(chunk)
            mock_single.assert_not_called()

    def test_auth_error_raises_immediately(self, translator):
        """AuthenticationError 应立即抛出"""
        chunk = _make_chunk()

        with patch("vat.translator.llm_translator.call_llm") as mock_llm:
            mock_llm.side_effect = _make_auth_error()
            with pytest.raises(RuntimeError, match="API 认证失败"):
                translator._translate_chunk(chunk)


# ─── _translate_chunk_single: 全部失败时抛异常 ──────────────────────

class TestTranslateChunkSingleFailure:
    """单条翻译零容忍：任何一条失败即 raise"""

    def test_any_failure_raises(self, translator):
        """任何单条翻译失败即立报错，不允许部分翻译缺失"""
        chunk = _make_chunk(3)

        with patch("vat.translator.llm_translator.call_llm") as mock_llm:
            mock_llm.side_effect = Exception("API error")
            with pytest.raises(RuntimeError, match="翻译失败"):
                translator._translate_chunk_single(chunk)

    def test_first_failure_stops_immediately(self, translator):
        """第一条失败后不应继续尝试后续条目"""
        chunk = _make_chunk(3)

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("API error")
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "翻译结果"
            return mock_resp

        with patch("vat.translator.llm_translator.call_llm") as mock_llm:
            mock_llm.side_effect = side_effect
            with pytest.raises(RuntimeError, match="翻译失败"):
                translator._translate_chunk_single(chunk)
            # 第 1 次调用就失败，不应再调用第 2、3 次
            assert call_count == 1


# ─── _parallel_translate: 失败比例检查 ──────────────────────────────

class TestParallelTranslateZeroTolerance:
    """翻译零容忍：任何批次失败即整体失败"""

    def test_all_chunks_failed_raises(self, translator):
        """所有批次失败 → 抛出异常"""
        chunks = [_make_chunk(3, start_idx=i * 3 + 1) for i in range(4)]

        with patch.object(translator, "_safe_translate_chunk") as mock_translate:
            mock_translate.side_effect = RuntimeError("API 请求被拒绝")
            with pytest.raises(RuntimeError, match="翻译批次.*失败"):
                translator._parallel_translate(chunks)

    def test_single_chunk_failed_raises(self, translator):
        """哪怕只有 1 个批次失败也应整体失败"""
        chunks = [_make_chunk(3, start_idx=i * 3 + 1) for i in range(4)]

        call_count = 0
        def side_effect(chunk):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return chunk

        with patch.object(translator, "_safe_translate_chunk") as mock_translate:
            mock_translate.side_effect = side_effect
            with pytest.raises(RuntimeError, match="翻译批次.*失败"):
                translator._parallel_translate(chunks)

    def test_all_success_returns_result(self, translator):
        """全部成功 → 正常返回"""
        chunks = [_make_chunk(3, start_idx=i * 3 + 1) for i in range(4)]

        with patch.object(translator, "_safe_translate_chunk") as mock_translate:
            mock_translate.side_effect = lambda chunk: chunk
            result = translator._parallel_translate(chunks)
            assert len(result) == 12  # 4 chunks * 3 items


# ─── _optimize_chunk: 不可重试错误 fail-fast ────────────────────────

class TestOptimizeChunkFailFast:
    """_optimize_chunk 对不可重试错误应 fail-fast"""

    def test_bad_request_error_raises(self, translator):
        """BadRequestError 应立即抛出"""
        chunk = {"1": "テスト", "2": "字幕"}

        with patch("vat.translator.llm_translator.call_llm") as mock_llm:
            mock_llm.side_effect = _make_bad_request_error()
            with pytest.raises(RuntimeError, match="优化 API 不可用"):
                translator._optimize_chunk(chunk)

    def test_auth_error_raises(self, translator):
        """AuthenticationError 应立即抛出"""
        chunk = {"1": "テスト", "2": "字幕"}

        with patch("vat.translator.llm_translator.call_llm") as mock_llm:
            mock_llm.side_effect = _make_auth_error()
            with pytest.raises(RuntimeError, match="优化 API 不可用"):
                translator._optimize_chunk(chunk)
