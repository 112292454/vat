"""playlists API 与 JobManager 交互契约测试。"""

from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from vat.web.jobs import JobStatus, WebJob
from vat.web.routes.playlists import router, get_db, get_playlist_service


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


@pytest.fixture
def app():
    return FastAPI()


@pytest.fixture
def client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _make_job(job_id, status=JobStatus.RUNNING, error=None):
    return WebJob(
        job_id=job_id,
        video_ids=[],
        steps=["sync-playlist"],
        gpu_device="auto",
        force=False,
        status=status,
        pid=4321 if status == JobStatus.RUNNING else None,
        log_file=None,
        progress=0.0,
        error=error,
        created_at=datetime(2026, 3, 14, 12, 0, 0),
        started_at=None,
        finished_at=None,
        task_type="sync-playlist",
        task_params={},
        cancel_requested=False,
    )


class _FakeJobManager:
    def __init__(self):
        self.jobs = {}
        self.submit_calls = []
        self.update_calls = []
        self.log_lines = {}

    def submit_job(self, **kwargs):
        self.submit_calls.append(kwargs)
        return f"job-{len(self.submit_calls)}"

    def update_job_status(self, job_id):
        self.update_calls.append(job_id)

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def get_log_content(self, job_id, tail_lines=3):
        return self.log_lines.get(job_id, [])


class _FakeDb:
    def __init__(self):
        self.playlists = {}

    def get_playlist(self, playlist_id):
        return self.playlists.get(playlist_id)

    def list_playlists(self):
        return list(self.playlists.values())

    def update_playlist(self, playlist_id, metadata=None):
        self.playlists[playlist_id].metadata = metadata


class _FakePlaylistService:
    def __init__(self, playlist=None):
        self.playlist = playlist

    def get_playlist(self, playlist_id):
        if self.playlist and self.playlist.id == playlist_id:
            return self.playlist
        return None

    def get_playlist_videos(self, playlist_id):
        return []

    def get_playlist_progress(self, playlist_id):
        return {"completed": 0, "total": 0}

    def delete_playlist(self, playlist_id, delete_videos=False):
        return {"deleted_videos": 0, "deleted_relations": 0, "delete_videos": delete_videos}


def _make_playlist(playlist_id="PL1"):
    return SimpleNamespace(
        id=playlist_id,
        title=f"Playlist {playlist_id}",
        source_url=f"https://youtube.com/playlist?list={playlist_id}",
        channel="channel",
        channel_id="channel-id",
        video_count=3,
        last_synced_at=None,
        metadata={},
    )


class TestPlaylistJobRoutes:
    @pytest.mark.anyio
    async def test_add_playlist_resolves_id_and_submits_sync_job(self, app, client, monkeypatch):
        fake_db = _FakeDb()
        job_manager = _FakeJobManager()
        fake_config = SimpleNamespace(
            downloader=SimpleNamespace(
                youtube=SimpleNamespace(
                    format="best",
                    cookies_file=None,
                    remote_components=False,
                )
            ),
            get_stage_proxy=lambda _stage: None,
        )

        class _FakeDownloader:
            def __init__(self, **_kwargs):
                pass

            def get_playlist_info(self, url):
                return {"id": "UC123"}

        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: fake_db
        monkeypatch.setattr("vat.web.routes.playlists._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.playlists._sync_status", {})
        monkeypatch.setattr("vat.web.routes.playlists.resolve_playlist_id", lambda url, playlist_id: f"{playlist_id}-videos")
        monkeypatch.setattr("vat.config.load_config", lambda: fake_config)
        monkeypatch.setattr("vat.downloaders.YouTubeDownloader", _FakeDownloader)

        async with client as ac:
            response = await ac.post(
                "/api/playlists",
                json={"url": "https://www.youtube.com/@test/videos"},
            )

        assert response.status_code == 200
        assert response.json()["playlist_id"] == "UC123-videos"
        assert job_manager.submit_calls == [{
            "video_ids": [],
            "steps": ["sync-playlist"],
            "task_type": "sync-playlist",
            "task_params": {
                "playlist_id": "UC123-videos",
                "url": "https://www.youtube.com/@test/videos",
            },
        }]

    @pytest.mark.anyio
    async def test_add_playlist_does_not_resubmit_running_sync_job(self, app, client, monkeypatch):
        fake_db = _FakeDb()
        job_manager = _FakeJobManager()
        job_manager.jobs["job-running"] = _make_job("job-running", status=JobStatus.RUNNING)
        fake_config = SimpleNamespace(
            downloader=SimpleNamespace(
                youtube=SimpleNamespace(
                    format="best",
                    cookies_file=None,
                    remote_components=False,
                )
            ),
            get_stage_proxy=lambda _stage: None,
        )

        class _FakeDownloader:
            def __init__(self, **_kwargs):
                pass

            def get_playlist_info(self, url):
                return {"id": "UC123"}

        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: fake_db
        monkeypatch.setattr("vat.web.routes.playlists._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.playlists._sync_status", {"UC123-videos": {"job_id": "job-running"}})
        monkeypatch.setattr("vat.web.routes.playlists.resolve_playlist_id", lambda url, playlist_id: f"{playlist_id}-videos")
        monkeypatch.setattr("vat.config.load_config", lambda: fake_config)
        monkeypatch.setattr("vat.downloaders.YouTubeDownloader", _FakeDownloader)

        async with client as ac:
            response = await ac.post(
                "/api/playlists",
                json={"url": "https://www.youtube.com/@test/videos"},
            )

        assert response.status_code == 200
        assert response.json()["message"] == "同步已在进行中"
        assert job_manager.update_calls == ["job-running"]
        assert job_manager.submit_calls == []

    @pytest.mark.anyio
    async def test_sync_playlist_submits_background_job(self, app, client, monkeypatch):
        fake_db = _FakeDb()
        fake_db.playlists["PL1"] = _make_playlist("PL1")
        job_manager = _FakeJobManager()

        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: fake_db
        monkeypatch.setattr("vat.web.routes.playlists._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.playlists._sync_status", {})

        async with client as ac:
            response = await ac.post("/api/playlists/PL1/sync", json={})

        assert response.status_code == 200
        assert response.json()["message"] == "已启动后台同步"
        assert job_manager.submit_calls == [{
            "video_ids": [],
            "steps": ["sync-playlist"],
            "task_type": "sync-playlist",
            "task_params": {"playlist_id": "PL1"},
        }]

    @pytest.mark.anyio
    async def test_sync_playlist_does_not_resubmit_running_job(self, app, client, monkeypatch):
        fake_db = _FakeDb()
        fake_db.playlists["PL1"] = _make_playlist("PL1")
        job_manager = _FakeJobManager()
        job_manager.jobs["job-running"] = _make_job("job-running", status=JobStatus.RUNNING)

        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: fake_db
        monkeypatch.setattr("vat.web.routes.playlists._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.playlists._sync_status", {"PL1": {"job_id": "job-running"}})

        async with client as ac:
            response = await ac.post("/api/playlists/PL1/sync", json={})

        assert response.status_code == 200
        assert response.json()["message"] == "同步已在进行中"
        assert job_manager.update_calls == ["job-running"]
        assert job_manager.submit_calls == []

    @pytest.mark.anyio
    async def test_get_sync_status_maps_running_job_and_prefers_job_error(self, app, client, monkeypatch):
        job_manager = _FakeJobManager()
        job_manager.jobs["job-1"] = _make_job("job-1", status=JobStatus.FAILED, error="job failed")
        job_manager.log_lines["job-1"] = ["older", "latest log line"]

        app.include_router(router)
        monkeypatch.setattr("vat.web.routes.playlists._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.playlists._sync_status", {"PL1": {"job_id": "job-1"}})

        async with client as ac:
            response = await ac.get("/api/playlists/PL1/sync-status")

        assert response.status_code == 200
        assert response.json() == {
            "status": "failed",
            "message": "job failed",
            "job_id": "job-1",
        }
        assert job_manager.update_calls == ["job-1"]

    @pytest.mark.anyio
    async def test_refresh_playlist_passes_force_flags_to_job_manager(self, app, client, monkeypatch):
        fake_db = _FakeDb()
        fake_db.playlists["PL1"] = _make_playlist("PL1")
        job_manager = _FakeJobManager()

        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: fake_db
        monkeypatch.setattr("vat.web.routes.playlists._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.playlists._refresh_status", {})

        async with client as ac:
            response = await ac.post(
                "/api/playlists/PL1/refresh",
                json={"force_refetch": True, "force_retranslate": True},
            )

        assert response.status_code == 200
        assert response.json()["status"] == "started"
        assert job_manager.submit_calls == [{
            "video_ids": [],
            "steps": ["refresh-playlist"],
            "task_type": "refresh-playlist",
            "task_params": {
                "playlist_id": "PL1",
                "force_refetch": True,
                "force_retranslate": True,
            },
        }]

    @pytest.mark.anyio
    async def test_refresh_status_remaps_syncing_to_refreshing(self, app, client, monkeypatch):
        job_manager = _FakeJobManager()
        job_manager.jobs["job-refresh"] = _make_job("job-refresh", status=JobStatus.RUNNING)

        app.include_router(router)
        monkeypatch.setattr("vat.web.routes.playlists._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.playlists._refresh_status", {"PL1": {"job_id": "job-refresh"}})

        async with client as ac:
            response = await ac.get("/api/playlists/PL1/refresh-status")

        assert response.status_code == 200
        assert response.json()["status"] == "refreshing"
        assert response.json()["job_id"] == "job-refresh"

    @pytest.mark.anyio
    async def test_retranslate_playlist_submits_background_job(self, app, client, monkeypatch):
        playlist = _make_playlist("PL1")
        service = _FakePlaylistService(playlist=playlist)
        job_manager = _FakeJobManager()

        app.include_router(router)
        app.dependency_overrides[get_playlist_service] = lambda: service
        monkeypatch.setattr("vat.web.routes.playlists._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.playlists._retranslate_status", {})

        async with client as ac:
            response = await ac.post("/api/playlists/PL1/retranslate")

        assert response.status_code == 200
        assert response.json()["status"] == "started"
        assert job_manager.submit_calls == [{
            "video_ids": [],
            "steps": ["retranslate-playlist"],
            "task_type": "retranslate-playlist",
            "task_params": {"playlist_id": "PL1"},
        }]
