from pathlib import Path
from types import SimpleNamespace

import uvicorn
import yaml

from vat.config import write_starter_config
from vat.web.app import run_server
from vat.web.deps import get_db, get_web_config, reset_web_runtime_state, set_web_config_path
from vat.web.routes import bilibili, playlists, tasks


def _write_web_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "web-config.yaml"
    write_starter_config(str(config_path))
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data["storage"]["database_path"] = str(tmp_path / "custom.db")
    data["storage"]["output_dir"] = str(tmp_path / "outputs")
    data["storage"]["work_dir"] = str(tmp_path / "work")
    data["uploader"]["bilibili"]["cookies_file"] = "cookies/custom.json"
    data["web"]["host"] = "127.0.0.1"
    data["web"]["port"] = 18081
    config_path.write_text(
        yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def test_run_server_propagates_group_config_to_web_runtime(monkeypatch, tmp_path):
    config_path = _write_web_config(tmp_path)

    reset_web_runtime_state()
    tasks._job_manager = None

    observed = {}

    class FakeJobManager:
        def __init__(self, db_path, log_dir, config_path=None):
            observed["job_manager_db_path"] = db_path
            observed["job_manager_log_dir"] = log_dir
            observed["job_manager_config_path"] = config_path

    class FakePlaylistService:
        def __init__(self, db, config=None):
            observed["playlist_service_db"] = db
            observed["playlist_service_config_db_path"] = config.storage.database_path

    def fake_uvicorn_run(app, host, port):
        observed["uvicorn_host"] = host
        observed["uvicorn_port"] = port

    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)
    monkeypatch.setattr(tasks, "JobManager", FakeJobManager)
    monkeypatch.setattr(playlists, "PlaylistService", FakePlaylistService)

    run_server(config_path=str(config_path))

    config = get_web_config()
    db = get_db()
    tasks.get_job_manager()
    playlists.get_playlist_service(db=db)

    assert config.storage.database_path == str(tmp_path / "custom.db")
    assert config.storage.output_dir == str(tmp_path / "outputs")
    assert str(db.db_path) == str(tmp_path / "custom.db")
    assert db.output_base_dir == str(tmp_path / "outputs")
    assert observed["job_manager_db_path"] == str(tmp_path / "custom.db")
    assert observed["job_manager_log_dir"] == str(tmp_path / "job_logs")
    assert observed["job_manager_config_path"] == str(config_path)
    assert observed["playlist_service_config_db_path"] == str(tmp_path / "custom.db")
    assert observed["uvicorn_host"] == "127.0.0.1"
    assert observed["uvicorn_port"] == 18081
    assert bilibili._get_cookies_path() == Path.cwd() / "cookies" / "custom.json"


def test_set_web_config_path_resets_cached_singletons(monkeypatch, tmp_path):
    config_path = _write_web_config(tmp_path)

    reset_web_runtime_state()
    tasks._job_manager = object()
    set_web_config_path(str(config_path))

    assert tasks._job_manager is None
