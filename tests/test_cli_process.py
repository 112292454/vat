from types import SimpleNamespace
from unittest.mock import MagicMock

from click.testing import CliRunner

from vat.cli.commands import process as process_cmd, translate as translate_cmd, parse_stages, status as status_cmd
from vat.config import Config
from vat.models import TaskStep


def _minimal_config():
    return Config.from_dict(
        {
            "storage": {
                "work_dir": "/tmp/work",
                "output_dir": "/tmp/output",
                "database_path": "/tmp/db.db",
                "models_dir": "/tmp/models",
                "resource_dir": "resources",
                "fonts_dir": "fonts",
                "subtitle_style_dir": "styles",
                "cache_dir": "/tmp/cache",
            },
            "downloader": {"youtube": {"format": "best", "max_workers": 1}},
            "asr": {
                "backend": "faster-whisper",
                "model": "large-v3",
                "language": "ja",
                "device": "auto",
                "compute_type": "float16",
                "vad_filter": False,
                "beam_size": 5,
                "models_subdir": "whisper",
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "temperature": [0.0],
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "initial_prompt": "",
                "repetition_penalty": 1.0,
                "hallucination_silence_threshold": None,
                "vad_threshold": 0.5,
                "vad_min_speech_duration_ms": 250,
                "vad_max_speech_duration_s": 30,
                "vad_min_silence_duration_ms": 100,
                "vad_speech_pad_ms": 30,
                "enable_chunked": False,
                "chunk_length_sec": 300,
                "chunk_overlap_sec": 10,
                "chunk_concurrency": 1,
                "split": {
                    "enable": True,
                    "mode": "sentence",
                    "max_words_cjk": 30,
                    "max_words_english": 15,
                    "min_words_cjk": 5,
                    "min_words_english": 3,
                    "model": "gpt-4",
                    "enable_chunking": False,
                    "chunk_size_sentences": 50,
                    "chunk_overlap_sentences": 2,
                    "chunk_min_threshold": 100,
                },
            },
            "translator": {
                "backend_type": "llm",
                "source_language": "ja",
                "target_language": "zh-cn",
                "skip_translate": False,
                "llm": {
                    "model": "gpt-4",
                    "enable_reflect": True,
                    "batch_size": 10,
                    "thread_num": 3,
                    "custom_prompt": "",
                    "enable_context": True,
                    "optimize": {"enable": True, "custom_prompt": ""},
                },
                "local": {
                    "model_filename": "model.gguf",
                    "backend": "sakura",
                    "n_gpu_layers": 35,
                    "context_size": 4096,
                },
            },
            "embedder": {
                "subtitle_formats": ["srt"],
                "embed_mode": "hard",
                "output_container": "mp4",
                "video_codec": "libx265",
                "audio_codec": "copy",
                "crf": 23,
                "preset": "medium",
                "use_gpu": True,
                "subtitle_style": "default",
            },
            "uploader": {
                "bilibili": {"cookies_file": "", "line": "AUTO", "threads": 3}
            },
            "gpu": {
                "device": "auto",
                "allow_cpu_fallback": False,
                "min_free_memory_mb": 2000,
            },
            "concurrency": {"gpu_devices": [0], "max_concurrent_per_gpu": 1},
            "logging": {"level": "INFO", "file": "vat.log", "format": "%(message)s"},
            "llm": {"api_key": "", "base_url": ""},
            "proxy": {"http_proxy": ""},
        }
    )


class TestProcessCommandConfigIsolation:
    def test_process_command_passes_independent_config_per_video(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.get_video.side_effect = lambda vid: SimpleNamespace(id=vid, title=f"title-{vid}")

        seen_config_ids = []

        class FakeProcessor:
            def __init__(self, *, video_id, config, **kwargs):
                seen_config_ids.append(id(config))

            def process(self, steps):
                return True

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr("vat.cli.commands.VideoProcessor", FakeProcessor)

        runner = CliRunner()
        result = runner.invoke(
            process_cmd,
            ["-v", "v1", "-v", "v2", "-s", "whisper", "-c", "2"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert len(seen_config_ids) == 2
        assert len(set(seen_config_ids)) == 2

    def test_playlist_prompt_context_does_not_mutate_cached_global_config(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.get_video.return_value = SimpleNamespace(id="v1", title="title-v1")

        captured_prompts = []

        class FakeProcessor:
            def __init__(self, *, config, **kwargs):
                captured_prompts.append(
                    (
                        config.translator.llm.custom_prompt,
                        config.translator.llm.optimize.custom_prompt,
                    )
                )

            def process(self, steps):
                return True

        fake_playlist_service = MagicMock()
        fake_playlist_service.get_playlist.return_value = SimpleNamespace(
            id="PL1",
            title="playlist",
            metadata={
                "custom_prompt_translate": "fubuki",
                "custom_prompt_optimize": "fubuki",
            },
        )

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr("vat.cli.commands.PlaylistService", lambda db: fake_playlist_service)
        monkeypatch.setattr("vat.cli.commands.VideoProcessor", FakeProcessor)

        original_translate_prompt = config.translator.llm.custom_prompt
        original_optimize_prompt = config.translator.llm.optimize.custom_prompt

        runner = CliRunner()
        result = runner.invoke(
            process_cmd,
            ["-v", "v1", "-p", "PL1", "-s", "whisper"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert captured_prompts
        translate_prompt, optimize_prompt = captured_prompts[0]
        assert translate_prompt != original_translate_prompt
        assert optimize_prompt != original_optimize_prompt
        assert config.translator.llm.custom_prompt == original_translate_prompt
        assert config.translator.llm.optimize.custom_prompt == original_optimize_prompt


class TestCliHelpContracts:
    def test_process_help_clarifies_translate_group_must_be_explicit(self):
        runner = CliRunner()
        result = runner.invoke(process_cmd, ["--help"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "optimize,translate" in result.output

    def test_translate_help_clarifies_it_only_runs_translate_stage(self):
        runner = CliRunner()
        result = runner.invoke(translate_cmd, ["--help"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "仅执行 translate 阶段" in result.output


class TestParseStagesContracts:
    def test_translate_is_single_stage_not_group(self):
        assert parse_stages("translate") == [TaskStep.TRANSLATE]

    def test_asr_group_expands_and_deduplicates(self):
        assert parse_stages("asr,translate,asr") == [
            TaskStep.WHISPER,
            TaskStep.SPLIT,
            TaskStep.TRANSLATE,
        ]

    def test_invalid_stage_raises_valueerror(self):
        try:
            parse_stages("whisper,unknown-stage")
        except ValueError as exc:
            assert "未知" in str(exc)
        else:
            raise AssertionError("parse_stages 应对未知阶段抛出 ValueError")


class TestProcessStageContracts:
    def test_process_command_delegates_normal_path_to_shared_batch_runner(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.get_video.return_value = SimpleNamespace(id="v1", title="title-v1")
        observed = {}

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr(
            "vat.cli.commands.run_video_batch",
            lambda **kwargs: observed.update(kwargs) or SimpleNamespace(failed_video_ids=[], stopped_early=False),
        )

        result = CliRunner().invoke(
            process_cmd,
            ["-v", "v1", "-s", "whisper", "-c", "2", "-g", "cuda:1", "--fail-fast"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert observed["video_ids"] == ["v1"]
        assert observed["steps"] == ["whisper"]
        assert observed["concurrency"] == 2
        assert observed["gpu_id"] == 1
        assert observed["fail_fast"] is True

    def test_process_translate_stage_reaches_processor_as_single_stage(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.get_video.return_value = SimpleNamespace(id="v1", title="title-v1")
        observed = {}

        class FakeProcessor:
            def __init__(self, **kwargs):
                pass

            def process(self, steps):
                observed["steps"] = steps
                return True

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr("vat.cli.commands.VideoProcessor", FakeProcessor)

        result = CliRunner().invoke(
            process_cmd,
            ["-v", "v1", "-s", "translate"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert observed["steps"] == ["translate"]

    def test_process_asr_group_reaches_processor_as_whisper_then_split(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.get_video.return_value = SimpleNamespace(id="v1", title="title-v1")
        observed = {}

        class FakeProcessor:
            def __init__(self, **kwargs):
                pass

            def process(self, steps):
                observed["steps"] = steps
                return True

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr("vat.cli.commands.VideoProcessor", FakeProcessor)

        result = CliRunner().invoke(
            process_cmd,
            ["-v", "v1", "-s", "asr"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert observed["steps"] == ["whisper", "split"]

    def test_process_force_invalidates_downstream_from_earliest_requested_stage(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.get_video.return_value = SimpleNamespace(id="v1", title="title-v1")

        class FakeProcessor:
            def __init__(self, **kwargs):
                pass

            def process(self, steps):
                return True

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr("vat.cli.commands.VideoProcessor", FakeProcessor)

        result = CliRunner().invoke(
            process_cmd,
            ["-v", "v1", "-s", "translate,embed", "--force"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        fake_db.invalidate_downstream_tasks.assert_called_once_with("v1", TaskStep.TRANSLATE)

    def test_process_force_dry_run_does_not_invalidate_downstream_tasks(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.get_video.return_value = SimpleNamespace(id="v1", title="title-v1")

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)

        result = CliRunner().invoke(
            process_cmd,
            ["-v", "v1", "-s", "translate", "--force", "--dry-run"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        fake_db.invalidate_downstream_tasks.assert_not_called()

    def test_process_force_invalid_upload_cron_does_not_invalidate_downstream_tasks(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)

        result = CliRunner().invoke(
            process_cmd,
            ["-v", "v1", "-s", "upload", "--force", "--upload-cron", "invalid"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "无效的 cron 表达式" in result.output
        fake_db.invalidate_downstream_tasks.assert_not_called()

    def test_process_rejects_invalid_stage_name(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)

        result = CliRunner().invoke(
            process_cmd,
            ["-v", "v1", "-s", "download,unknown-stage"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "无效的阶段参数" in result.output


class TestTranslateCommandContracts:
    def test_translate_all_only_selects_split_done_and_not_translated_videos(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.list_videos.return_value = [
            SimpleNamespace(id="v-split-pending"),
            SimpleNamespace(id="v-translated"),
            SimpleNamespace(id="v-no-split"),
        ]

        def is_step_completed(video_id, step):
            table = {
                ("v-split-pending", TaskStep.SPLIT): True,
                ("v-split-pending", TaskStep.TRANSLATE): False,
                ("v-translated", TaskStep.SPLIT): True,
                ("v-translated", TaskStep.TRANSLATE): True,
                ("v-no-split", TaskStep.SPLIT): False,
                ("v-no-split", TaskStep.TRANSLATE): False,
            }
            return table.get((video_id, step), False)

        fake_db.is_step_completed.side_effect = is_step_completed
        scheduled = {}

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr(
            "vat.cli.commands.schedule_videos",
            lambda cfg, video_ids, steps, use_multi_gpu, force: scheduled.update(
                {"video_ids": video_ids, "steps": steps, "force": force}
            ),
        )

        result = CliRunner().invoke(
            translate_cmd,
            ["--all"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert scheduled == {
            "video_ids": ["v-split-pending"],
            "steps": ["translate"],
            "force": False,
        }

    def test_translate_all_force_reincludes_already_translated_videos(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.list_videos.return_value = [
            SimpleNamespace(id="v-split-pending"),
            SimpleNamespace(id="v-translated"),
            SimpleNamespace(id="v-no-split"),
        ]

        def is_step_completed(video_id, step):
            table = {
                ("v-split-pending", TaskStep.SPLIT): True,
                ("v-split-pending", TaskStep.TRANSLATE): False,
                ("v-translated", TaskStep.SPLIT): True,
                ("v-translated", TaskStep.TRANSLATE): True,
                ("v-no-split", TaskStep.SPLIT): False,
                ("v-no-split", TaskStep.TRANSLATE): False,
            }
            return table.get((video_id, step), False)

        fake_db.is_step_completed.side_effect = is_step_completed
        scheduled = {}

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr(
            "vat.cli.commands.schedule_videos",
            lambda cfg, video_ids, steps, use_multi_gpu, force: scheduled.update(
                {"video_ids": video_ids, "steps": steps, "force": force}
            ),
        )

        result = CliRunner().invoke(
            translate_cmd,
            ["--all", "--force"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert scheduled == {
            "video_ids": ["v-split-pending", "v-translated"],
            "steps": ["translate"],
            "force": True,
        }

    def test_translate_backend_online_maps_to_llm_backend_type(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        scheduled = {}

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr(
            "vat.cli.commands.schedule_videos",
            lambda cfg, video_ids, steps, use_multi_gpu, force: scheduled.update(
                {
                    "backend_type": cfg.translator.backend_type,
                    "video_ids": video_ids,
                    "steps": steps,
                }
            ),
        )

        runner = CliRunner()
        result = runner.invoke(
            translate_cmd,
            ["-v", "v1", "--backend", "online"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert scheduled["backend_type"] == "llm"
        assert scheduled["video_ids"] == ["v1"]
        assert scheduled["steps"] == ["translate"]
        assert config.translator.backend_type == "llm"

    def test_translate_backend_local_maps_to_local_backend_type_without_mutating_cached_config(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        scheduled = {}

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.get_logger", lambda: MagicMock())
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)
        monkeypatch.setattr(
            "vat.cli.commands.schedule_videos",
            lambda cfg, video_ids, steps, use_multi_gpu, force: scheduled.update(
                {
                    "backend_type": cfg.translator.backend_type,
                    "video_ids": video_ids,
                }
            ),
        )

        runner = CliRunner()
        result = runner.invoke(
            translate_cmd,
            ["-v", "v1", "--backend", "local"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert scheduled["backend_type"] == "local"
        # 缓存配置对象不应被本次 CLI 调用改脏
        assert config.translator.backend_type == "llm"

    def test_translate_rejects_unsupported_hybrid_backend(self):
        runner = CliRunner()
        result = runner.invoke(translate_cmd, ["--backend", "hybrid"], catch_exceptions=False)

        assert result.exit_code != 0
        assert "hybrid" in result.output


class TestStatusCommandContracts:
    def test_status_pending_filter_excludes_fully_skipped_video(self, monkeypatch):
        config = _minimal_config()
        fake_db = MagicMock()
        fake_db.list_videos.return_value = [
            SimpleNamespace(id="v-skip", title="跳过视频", source_type=SimpleNamespace(value="youtube")),
        ]
        fake_db.get_tasks.return_value = [
            SimpleNamespace(id=i + 1, step=step, status=SimpleNamespace(value="skipped"))
            for i, step in enumerate(TaskStep)
        ]

        monkeypatch.setattr("vat.cli.commands.get_config", lambda path: config)
        monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: fake_db)

        result = CliRunner().invoke(
            status_cmd,
            ["--pending"],
            obj={"config_path": "unused.yaml"},
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        assert "跳过视频" not in result.output
