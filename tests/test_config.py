"""
config.py 单元测试

测试配置加载、字段默认值、新增字段（download_delay 等）。
"""
import pytest
from vat.config import (
    Config, YouTubeDownloaderConfig, GPUConfig, DownloaderConfig,
)


def _minimal_config_dict(**overrides):
    """构造最小化但完整的配置字典，用于 Config.from_dict"""
    data = {
        'storage': {
            'work_dir': '/tmp/work',
            'output_dir': '/tmp/output',
            'database_path': '/tmp/db.db',
            'models_dir': '/tmp/models',
            'resource_dir': 'resources',
            'fonts_dir': 'fonts',
            'subtitle_style_dir': 'styles',
            'cache_dir': '/tmp/cache',
        },
        'downloader': {'youtube': {'format': 'best', 'max_workers': 1}},
        'asr': {
            'backend': 'faster-whisper', 'model': 'large-v3', 'language': 'ja',
            'device': 'auto', 'compute_type': 'float16', 'vad_filter': False,
            'beam_size': 5, 'models_subdir': 'whisper', 'word_timestamps': True,
            'condition_on_previous_text': False, 'temperature': [0.0],
            'compression_ratio_threshold': 2.4, 'log_prob_threshold': -1.0,
            'no_speech_threshold': 0.6, 'initial_prompt': '',
            'repetition_penalty': 1.0, 'hallucination_silence_threshold': None,
            'vad_threshold': 0.5, 'vad_min_speech_duration_ms': 250,
            'vad_max_speech_duration_s': 30, 'vad_min_silence_duration_ms': 100,
            'vad_speech_pad_ms': 30, 'enable_chunked': False,
            'chunk_length_sec': 300, 'chunk_overlap_sec': 10,
            'chunk_concurrency': 1,
            'split': {
                'enable': True, 'mode': 'sentence',
                'max_words_cjk': 30, 'max_words_english': 15,
                'min_words_cjk': 5, 'min_words_english': 3,
                'model': 'gpt-4', 'enable_chunking': False,
                'chunk_size_sentences': 50, 'chunk_overlap_sentences': 2,
                'chunk_min_threshold': 100,
            },
        },
        'translator': {
            'backend_type': 'llm', 'source_language': 'ja',
            'target_language': 'zh-cn', 'skip_translate': False,
            'llm': {
                'model': 'gpt-4', 'enable_reflect': True, 'batch_size': 10,
                'thread_num': 3, 'custom_prompt': '', 'enable_context': True,
                'optimize': {'enable': True, 'custom_prompt': ''},
            },
            'local': {
                'model_filename': 'model.gguf', 'backend': 'sakura',
                'n_gpu_layers': 35, 'context_size': 4096,
            },
        },
        'embedder': {
            'subtitle_formats': ['srt'], 'embed_mode': 'hard',
            'output_container': 'mp4', 'video_codec': 'libx265',
            'audio_codec': 'copy', 'crf': 23, 'preset': 'medium',
            'use_gpu': True, 'subtitle_style': 'default',
        },
        'uploader': {'bilibili': {'cookies_file': '', 'line': 'AUTO', 'threads': 3}},
        'gpu': {'device': 'auto', 'allow_cpu_fallback': False, 'min_free_memory_mb': 2000},
        'concurrency': {'gpu_devices': [0], 'max_concurrent_per_gpu': 1},
        'logging': {'level': 'INFO', 'file': 'vat.log', 'format': '%(message)s'},
        'llm': {'api_key': '', 'base_url': ''},
        'proxy': {'http_proxy': ''},
    }
    # 递归合并 overrides
    def _merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                _merge(base[k], v)
            else:
                base[k] = v
    _merge(data, overrides)
    return data


class TestConfigFromDict:
    """Config.from_dict 基本加载"""

    def test_loads_without_error(self):
        config = Config.from_dict(_minimal_config_dict())
        assert config is not None

    def test_gpu_config(self):
        config = Config.from_dict(_minimal_config_dict())
        assert config.gpu.device == "auto"
        assert config.gpu.allow_cpu_fallback is False
        assert config.gpu.min_free_memory_mb == 2000

    def test_storage_paths(self):
        config = Config.from_dict(_minimal_config_dict())
        assert config.storage.work_dir == '/tmp/work'
        assert config.storage.database_path == '/tmp/db.db'


class TestYouTubeDownloaderConfig:
    """YouTubeDownloaderConfig 字段测试"""

    def test_default_download_delay(self):
        """download_delay 默认值为 0"""
        cfg = YouTubeDownloaderConfig(format='best', max_workers=1)
        assert cfg.download_delay == 0

    def test_explicit_download_delay(self):
        cfg = YouTubeDownloaderConfig(format='best', max_workers=1, download_delay=30)
        assert cfg.download_delay == 30

    def test_default_cookies_file(self):
        cfg = YouTubeDownloaderConfig(format='best', max_workers=1)
        assert cfg.cookies_file == ""

    def test_default_remote_components(self):
        cfg = YouTubeDownloaderConfig(format='best', max_workers=1)
        assert cfg.remote_components == []

    def test_explicit_remote_components(self):
        cfg = YouTubeDownloaderConfig(
            format='best', max_workers=1,
            remote_components=["ejs:github"],
        )
        assert cfg.remote_components == ["ejs:github"]

    def test_default_subtitle_languages(self):
        cfg = YouTubeDownloaderConfig(format='best', max_workers=1)
        assert cfg.subtitle_languages == ["ja", "zh", "en"]

    def test_download_delay_from_full_config(self):
        """通过 Config.from_dict 加载 download_delay"""
        data = _minimal_config_dict(
            downloader={'youtube': {'format': 'best', 'max_workers': 1, 'download_delay': 15}}
        )
        config = Config.from_dict(data)
        assert config.downloader.youtube.download_delay == 15

    def test_cookies_from_full_config(self):
        data = _minimal_config_dict(
            downloader={'youtube': {
                'format': 'best', 'max_workers': 1,
                'cookies_file': '/path/to/cookies.txt',
                'remote_components': ['ejs:github'],
            }}
        )
        config = Config.from_dict(data)
        assert config.downloader.youtube.cookies_file == '/path/to/cookies.txt'
        assert config.downloader.youtube.remote_components == ['ejs:github']


class TestConfigLoadFromFile:
    """从真实配置文件加载"""

    def test_load_default_config(self):
        """default.yaml 应能正常加载且包含新字段"""
        from vat.config import load_config
        config = load_config()
        # download_delay 在 default.yaml 中配置为 30（用户已修改）
        assert config.downloader.youtube.download_delay > 0
        assert isinstance(config.downloader.youtube.cookies_file, str)
        assert isinstance(config.downloader.youtube.remote_components, list)
