import queue
from types import SimpleNamespace

import pytest

import vat.asr.chunked_asr as chunked_asr_module
import vat.asr.model_source as model_source_module
import vat.asr.whisper_wrapper as whisper_wrapper_module

from vat.asr.chunked_asr import ChunkedASR, _load_model_with_memory_check
from vat.asr.whisper_wrapper import WhisperASR


class _FakeLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def test_load_model_with_memory_check_reports_network_or_cache_guidance(monkeypatch):
    class FakeWhisperASR:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def _ensure_model_loaded(self):
            raise RuntimeError("ConnectError: [Errno 104] Connection reset by peer")

    monkeypatch.setattr("vat.utils.gpu.check_gpu_free_memory", lambda gpu_id, min_free_memory_mb: True)
    monkeypatch.setattr(whisper_wrapper_module, "WhisperASR", FakeWhisperASR)

    with pytest.raises(RuntimeError, match="首次 ASR 需要可用网络下载 Whisper 模型"):
        _load_model_with_memory_check(
            gpu_id=0,
            whisper_kwargs={
                "model_name": "large-v3",
                "download_root": "/tmp/isolated-models",
            },
            min_free_memory_mb=8000,
            worker_logger=_FakeLogger(),
        )


def test_asr_chunks_multiprocess_fails_fast_on_worker_model_init_error(monkeypatch, tmp_path):
    class FakeQueue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

        def get(self, timeout=None):
            if not self.items:
                raise queue.Empty
            return self.items.pop(0)

    class FakeProcess:
        def __init__(self, target, args, name):
            self.args = args
            self.name = name
            self._alive = False

        def start(self):
            result_queue = self.args[2]
            result_queue.put({
                "chunk_idx": -1,
                "segments": [],
                "error": "Whisper 模型加载失败（model=large-v3）。首次 ASR 需要可用网络下载 Whisper 模型，或预先将模型缓存放到 /tmp/isolated-models。",
                "fatal": True,
                "gpu_id": 0,
            })

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            return None

        def terminate(self):
            return None

        def kill(self):
            return None

        def close(self):
            return None

    class FakeContext:
        def Queue(self):
            return FakeQueue()

        def Process(self, target, args, name):
            return FakeProcess(target, args, name)

    whisper_instance = SimpleNamespace(
        model_name="large-v3",
        compute_type="float32",
        language="ja",
        vad_filter=False,
        beam_size=5,
        download_root="/tmp/isolated-models",
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=[0.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        initial_prompt="",
        repetition_penalty=1.0,
        hallucination_silence_threshold=2,
        vad_threshold=0.02,
        vad_min_speech_duration_ms=30,
        vad_max_speech_duration_s=9999.0,
        vad_min_silence_duration_ms=20,
        vad_speech_pad_ms=5000,
        chunk_length_sec=600,
        chunk_overlap_sec=10,
        use_pipeline=False,
        enable_diarization=False,
        enable_punctuation=False,
        pipeline_batch_size=8,
        pipeline_chunk_length=30,
        num_speakers=1,
        min_speakers=1,
        max_speakers=2,
    )

    audio_path = tmp_path / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    chunked = ChunkedASR(
        asr_class=object,
        audio_path=str(audio_path),
        asr_kwargs={"whisper_asr": whisper_instance, "language": "ja"},
        chunk_length=600,
        chunk_overlap=10,
        chunk_concurrency=1,
    )

    monkeypatch.setattr(chunked_asr_module.multiprocessing, "get_context", lambda method: FakeContext())

    with pytest.raises(RuntimeError, match="首次 ASR 需要可用网络下载 Whisper 模型"):
        chunked._asr_chunks_multiprocess(
            chunks=[(b"fake", 0)],
            callback=None,
            gpu_count=1,
        )


def test_asr_chunks_multiprocess_preflights_hf_connectivity_when_cache_missing(monkeypatch, tmp_path):
    whisper_instance = SimpleNamespace(
        model_name="large-v3",
        compute_type="float32",
        language="ja",
        vad_filter=False,
        beam_size=5,
        download_root="/tmp/isolated-models",
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=[0.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        initial_prompt="",
        repetition_penalty=1.0,
        hallucination_silence_threshold=2,
        vad_threshold=0.02,
        vad_min_speech_duration_ms=30,
        vad_max_speech_duration_s=9999.0,
        vad_min_silence_duration_ms=20,
        vad_speech_pad_ms=5000,
        chunk_length_sec=600,
        chunk_overlap_sec=10,
        use_pipeline=False,
        enable_diarization=False,
        enable_punctuation=False,
        pipeline_batch_size=8,
        pipeline_chunk_length=30,
        num_speakers=1,
        min_speakers=1,
        max_speakers=2,
    )

    audio_path = tmp_path / "demo.mp3"
    audio_path.write_bytes(b"fake-audio")
    chunked = ChunkedASR(
        asr_class=object,
        audio_path=str(audio_path),
        asr_kwargs={"whisper_asr": whisper_instance, "language": "ja"},
        chunk_length=600,
        chunk_overlap=10,
        chunk_concurrency=1,
    )

    monkeypatch.setattr(model_source_module, "ensure_whisper_model_source_available", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Whisper 模型加载失败（model=large-v3）。首次 ASR 需要可用网络下载 Whisper 模型，或预先将模型缓存放到 /tmp/isolated-models。原始错误: RuntimeError: 无法连接 HuggingFace，且本地模型缓存不存在")))
    monkeypatch.setattr(
        chunked_asr_module.multiprocessing,
        "get_context",
        lambda method: (_ for _ in ()).throw(AssertionError("不应在 preflight 失败时启动 worker")),
    )

    with pytest.raises(RuntimeError, match="本地模型缓存不存在"):
        chunked._asr_chunks_multiprocess(
            chunks=[(b"fake", 0)],
            callback=None,
            gpu_count=1,
        )


def test_whisper_model_load_preflights_single_gpu_path(monkeypatch):
    asr = WhisperASR.__new__(WhisperASR)
    asr.model_name = "large-v3"
    asr.device = "cpu"
    asr.compute_type = "float32"
    asr.download_root = "/tmp/isolated-models"
    asr.gpu_id = None
    asr.model = None

    monkeypatch.setattr(
        whisper_wrapper_module,
        "ensure_whisper_model_source_available",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Whisper 模型加载失败（model=large-v3）。首次 ASR 需要可用网络下载 Whisper 模型，或预先将模型缓存放到 /tmp/isolated-models。原始错误: RuntimeError: 无法连接 HuggingFace，且本地模型缓存不存在")
        ),
    )
    monkeypatch.setattr(
        whisper_wrapper_module,
        "WhisperModel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("preflight 失败时不应尝试加载 WhisperModel")),
    )

    with pytest.raises(RuntimeError, match="首次 ASR 需要可用网络下载 Whisper 模型"):
        asr._load_faster_whisper_model()


def test_has_local_whisper_cache_checks_required_files(monkeypatch, tmp_path):
    cache_dir = tmp_path / "whisper"
    cache_dir.mkdir()
    (cache_dir / "model.bin").write_text("x", encoding="utf-8")

    assert model_source_module.has_local_whisper_cache("large-v3", str(cache_dir)) is False

    (cache_dir / "config.json").write_text("{}", encoding="utf-8")
    assert model_source_module.has_local_whisper_cache("large-v3", str(cache_dir)) is True


def test_has_local_whisper_cache_accepts_hf_cache_layout_under_download_root(tmp_path):
    cache_dir = tmp_path / "whisper-cache"
    snapshot_dir = cache_dir / "models--Systran--faster-whisper-large-v3" / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "model.bin").write_text("x", encoding="utf-8")
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    assert model_source_module.has_local_whisper_cache("large-v3", str(cache_dir)) is True


def test_has_local_whisper_cache_treats_cached_no_exist_as_missing(monkeypatch):
    from huggingface_hub.file_download import _CACHED_NO_EXIST

    calls = []

    def fake_try_to_load_from_cache(repo_id, filename, cache_dir=None, revision=None, repo_type=None):
        calls.append((repo_id, filename))
        return _CACHED_NO_EXIST

    monkeypatch.setattr(
        "huggingface_hub.file_download.try_to_load_from_cache",
        fake_try_to_load_from_cache,
    )

    assert model_source_module.has_local_whisper_cache("large-v3", None) is False
    assert calls == [("Systran/faster-whisper-large-v3", "model.bin")]
