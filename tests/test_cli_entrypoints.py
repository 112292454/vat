from pathlib import Path
import builtins
import importlib
import sys
from types import SimpleNamespace

import yaml
from click.testing import CliRunner

from vat.cli.commands import cli, init as init_cmd, web as web_cmd, watch as watch_cmd
from vat.config import write_starter_config


def test_write_starter_config_writes_sanitized_starter_config(tmp_path):
    output_path = tmp_path / "config.yaml"

    write_starter_config(str(output_path))
    data = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert data["storage"]["work_dir"] == "./work"
    assert data["storage"]["output_dir"] == "./data/videos"
    assert data["storage"]["database_path"] == "./data/database.db"
    assert data["storage"]["models_dir"] == "./models"
    assert data["storage"]["resource_dir"] == "vat/resources"
    assert data["storage"]["fonts_dir"] == "vat/resources/fonts"
    assert data["storage"]["subtitle_style_dir"] == "vat/resources/subtitle_style"
    assert data["downloader"]["youtube"]["cookies_file"] == ""
    assert data["uploader"]["bilibili"]["cookies_file"] == ""
    assert data["uploader"]["bilibili"]["default_tags"] == []
    assert data["uploader"]["bilibili"]["season_id"] is None
    assert data["uploader"]["bilibili"]["templates"]["title"] == "${translated_title}"
    assert data["uploader"]["bilibili"]["templates"]["description"] == "${translated_desc}"
    assert data["uploader"]["bilibili"]["templates"]["custom_vars"] == {}
    assert data["proxy"]["http_proxy"] == ""
    assert data["llm"]["provider"] == "openai_compatible"
    assert data["llm"]["api_key"] == "${VAT_LLM_APIKEY}"
    assert data["llm"]["base_url"] == "https://api.openai.com/v1"
    assert data["llm"]["credentials_path"] == ""
    assert data["llm"]["project_id"] == ""
    raw_text = output_path.read_text(encoding="utf-8")
    assert "_initialized" not in raw_text
    assert "/local/gzy/4090/vat" not in raw_text
    assert "/home/gzy/py/vat" not in raw_text
    assert "1047066927" not in raw_text
    assert "全熟" not in raw_text


def test_init_command_delegates_to_write_starter_config(monkeypatch):
    observed = {}

    def fake_write_starter_config(output):
        observed["output"] = output

    monkeypatch.setattr("vat.cli.commands.write_starter_config", fake_write_starter_config)

    result = CliRunner().invoke(
        init_cmd,
        ["-o", "config/config.yaml"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert observed == {"output": "config/config.yaml"}


def test_init_command_prints_minimum_next_step_guidance(monkeypatch):
    monkeypatch.setattr("vat.cli.commands.write_starter_config", lambda output: None)

    result = CliRunner().invoke(
        init_cmd,
        ["-o", "config/config.yaml"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "storage.work_dir / output_dir / database_path / models_dir" in result.output
    assert "VAT_LLM_APIKEY" in result.output
    assert "HuggingFace" in result.output
    assert "cookies 可先留空" in result.output
    assert "storage.models_dir/asr.models_subdir" in result.output


def test_init_command_returns_nonzero_on_write_failure(monkeypatch):
    monkeypatch.setattr(
        "vat.cli.commands.write_starter_config",
        lambda output: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = CliRunner().invoke(
        init_cmd,
        ["-o", "config/config.yaml"],
        catch_exceptions=False,
    )

    assert result.exit_code != 0
    assert "创建配置文件失败: boom" in result.output


def test_web_command_delegates_to_run_server_with_group_config(monkeypatch):
    observed = {}

    def fake_run_server(*, host=None, port=None, config_path=None):
        observed.update({
            "host": host,
            "port": port,
            "config_path": config_path,
        })

    monkeypatch.setattr("vat.web.app.run_server", fake_run_server)

    result = CliRunner().invoke(
        web_cmd,
        ["--host", "127.0.0.1", "--port", "18080"],
        obj={"config_path": "config/dev.yaml"},
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert observed == {
        "host": "127.0.0.1",
        "port": 18080,
        "config_path": "config/dev.yaml",
    }


def test_watch_command_passes_group_config_to_process_job_submitter(monkeypatch):
    observed = {}

    fake_config = SimpleNamespace(
        storage=SimpleNamespace(database_path="db.sqlite3", output_dir="outputs"),
        watch=SimpleNamespace(
            default_interval=60,
            default_stages="all",
            default_concurrency=1,
        ),
    )

    class FakePlaylistService:
        def __init__(self, db):
            observed["playlist_service_db"] = db

        def get_playlist(self, playlist_id):
            return SimpleNamespace(id=playlist_id, title=f"Playlist {playlist_id}")

    class FakeWatchService:
        def __init__(self, **kwargs):
            observed["watch_service_kwargs"] = kwargs

        def run(self):
            observed["watch_run_called"] = True

    def fake_build_process_job_submitter(config, config_path=None):
        observed["submitter_config"] = config
        observed["submitter_config_path"] = config_path
        return object()

    monkeypatch.setattr("vat.cli.commands.get_config", lambda path: fake_config)
    monkeypatch.setattr(
        "vat.cli.commands.get_logger",
        lambda: SimpleNamespace(info=lambda *a, **k: None),
    )
    monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: object())
    monkeypatch.setattr("vat.cli.commands._get_playlist_service_cls", lambda: FakePlaylistService)
    monkeypatch.setattr(
        "vat.services.process_job_submitter.build_process_job_submitter",
        fake_build_process_job_submitter,
    )
    monkeypatch.setattr("vat.services.watch_service.WatchService", FakeWatchService)

    result = CliRunner().invoke(
        watch_cmd,
        ["--playlist", "PL_A", "--once"],
        obj={"config_path": "config/dev.yaml"},
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert observed["submitter_config"] is fake_config
    assert observed["submitter_config_path"] == "config/dev.yaml"
    assert observed["watch_service_kwargs"]["process_job_submitter"] is not None
    assert observed["watch_run_called"] is True


def test_cli_help_does_not_require_pipeline_or_service_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith(("vat.pipeline", "vat.downloaders", "vat.services")):
            raise AssertionError(f"unexpected eager import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("vat.cli.commands", None)

    commands = importlib.import_module("vat.cli.commands")
    result = CliRunner().invoke(commands.cli, ["--help"], catch_exceptions=False)

    assert result.exit_code == 0
    assert "web" in result.output


def test_pipeline_local_happy_path_does_not_eager_import_torch(monkeypatch, tmp_path):
    video_file = tmp_path / "demo.mp4"
    video_file.write_bytes(b"fake")

    original_import = builtins.__import__
    observed = {}

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for name in [
        "vat.pipeline.executor",
        "vat.pipeline",
        "vat.asr.vocal_separation.separator",
        "vat.asr.vocal_separation",
        "vat.asr",
    ]:
        sys.modules.pop(name, None)
    monkeypatch.setattr(
        "vat.cli.commands.get_config",
        lambda path: SimpleNamespace(
            storage=SimpleNamespace(database_path="db.sqlite3", output_dir="outputs"),
            concurrency=SimpleNamespace(gpu_devices=[0]),
        ),
    )
    monkeypatch.setattr("vat.cli.commands.get_logger", lambda: SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None))
    monkeypatch.setattr("vat.cli.commands.Database", lambda *args, **kwargs: object())

    pipeline_module = importlib.import_module("vat.pipeline")
    monkeypatch.setattr(pipeline_module, "detect_source_type", lambda src: SimpleNamespace(value="local"))
    monkeypatch.setattr(
        pipeline_module,
        "create_video_from_source",
        lambda src, db, source_type, title="": observed.setdefault("video_id", "vid-local"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "schedule_videos",
        lambda config, video_ids, steps, use_multi_gpu=False, force=False: observed.update(
            {"video_ids": video_ids, "steps": steps, "force": force}
        ),
    )

    result = CliRunner().invoke(
        cli,
        ["pipeline", "--url", str(video_file), "--title", "demo"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert observed["video_ids"] == ["vid-local"]
    assert observed["steps"] == ["download", "asr", "translate", "embed"]
