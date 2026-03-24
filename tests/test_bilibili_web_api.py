"""B站 Web API 契约测试。"""

import os
import tempfile
from types import SimpleNamespace
from datetime import datetime

import httpx
import pytest
from fastapi import FastAPI

os.environ["HOME"] = os.path.join(tempfile.gettempdir(), "vat-test-home")

from vat.web.routes.bilibili import router
from vat.web.jobs import JobStatus, WebJob


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


class _FakeDatabase:
    def __init__(self, db_path, output_base_dir=None):
        self.db_path = db_path
        self.output_base_dir = output_base_dir


class _FakeJobManager:
    def __init__(self):
        self.jobs = {}
        self.update_calls = []
        self.log_lines = {}

    def update_job_status(self, job_id):
        self.update_calls.append(job_id)

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def get_log_content(self, job_id, tail_lines=3):
        return self.log_lines.get(job_id, [])

    def find_latest_job(self, task_type, task_params_subset=None, statuses=None, limit=200):
        expected = task_params_subset or {}
        allowed = set(statuses or [])
        for job in self.jobs.values():
            if job.task_type != task_type:
                continue
            if allowed and job.status not in allowed:
                continue
            params = job.task_params or {}
            if all(params.get(key) == value for key, value in expected.items()):
                return job
        return None


def _make_job(job_id, *, task_type, task_params, status=JobStatus.RUNNING, error=None):
    return WebJob(
        job_id=job_id,
        video_ids=[],
        steps=[task_type],
        gpu_device="auto",
        force=False,
        status=status,
        pid=4321 if status == JobStatus.RUNNING else None,
        log_file=None,
        progress=0.5,
        error=error,
        created_at=datetime(2026, 3, 24, 12, 0, 0),
        started_at=None,
        finished_at=None,
        task_type=task_type,
        task_params=task_params,
        cancel_requested=False,
    )


class TestResyncSeasonInfoRoute:
    @pytest.mark.anyio
    async def test_dispatches_batch_resync_via_threadpool(self, app, client, monkeypatch):
        fake_config = SimpleNamespace(
            storage=SimpleNamespace(database_path="db.sqlite3", output_dir="outputs"),
        )
        fake_uploader = object()
        threadpool_call = {}
        fake_db = _FakeDatabase("db.sqlite3", "outputs")

        app.include_router(router)
        monkeypatch.setattr("vat.web.routes.bilibili.load_config", lambda: fake_config)
        monkeypatch.setattr("vat.web.routes.bilibili.get_db", lambda: fake_db)
        monkeypatch.setattr("vat.web.routes.bilibili._get_uploader", lambda with_upload_params=False: fake_uploader)

        def _should_not_be_called_directly(*args, **kwargs):
            raise AssertionError("resync_season_video_infos 应通过线程池调度，而不是直接在 async 路由中调用")

        async def _fake_run_in_threadpool(func, *args, **kwargs):
            threadpool_call["func"] = func
            threadpool_call["args"] = args
            threadpool_call["kwargs"] = kwargs
            return {
                "success": True,
                "season_id": 42,
                "refreshed": 1,
                "failed": 0,
                "skipped": 0,
                "details": [],
                "message": "ok",
            }

        monkeypatch.setattr(
            "vat.web.routes.bilibili.resync_season_video_infos",
            _should_not_be_called_directly,
            raising=False,
        )
        monkeypatch.setattr("vat.web.routes.bilibili.run_in_threadpool", _fake_run_in_threadpool)

        async with client as ac:
            response = await ac.post("/bilibili/season/42/resync-info")

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert threadpool_call["func"] is _should_not_be_called_directly
        assert threadpool_call["kwargs"]["season_id"] == 42
        assert threadpool_call["kwargs"]["delay_seconds"] == 1.0

    @pytest.mark.anyio
    async def test_returns_aggregate_counts_for_completed_batch(self, app, client, monkeypatch):
        fake_config = SimpleNamespace(
            storage=SimpleNamespace(database_path="db.sqlite3", output_dir="outputs"),
        )
        fake_uploader = object()

        app.include_router(router)
        monkeypatch.setattr("vat.web.routes.bilibili.load_config", lambda: fake_config)
        monkeypatch.setattr("vat.web.routes.bilibili.Database", _FakeDatabase)
        monkeypatch.setattr("vat.web.routes.bilibili._get_uploader", lambda with_upload_params=False: fake_uploader)
        monkeypatch.setattr(
            "vat.web.routes.bilibili.run_in_threadpool",
            lambda func, *args, **kwargs: __import__("asyncio").sleep(0, result=func(*args, **kwargs)),
        )
        monkeypatch.setattr(
            "vat.web.routes.bilibili.resync_season_video_infos",
            lambda db, uploader, config, season_id, delay_seconds=1.0: {
                "success": True,
                "season_id": season_id,
                "refreshed": 2,
                "failed": 1,
                "skipped": 0,
                "details": [],
                "message": "合集 42 元信息同步完成：成功 2，失败 1，跳过 0",
            },
            raising=False,
        )

        async with client as ac:
            response = await ac.post("/bilibili/season/42/resync-info")

        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "season_id": 42,
            "refreshed": 2,
            "failed": 1,
            "skipped": 0,
            "details": [],
            "message": "合集 42 元信息同步完成：成功 2，失败 1，跳过 0",
        }

    @pytest.mark.anyio
    async def test_returns_error_when_batch_setup_fails(self, app, client, monkeypatch):
        fake_config = SimpleNamespace(
            storage=SimpleNamespace(database_path="db.sqlite3", output_dir="outputs"),
        )

        app.include_router(router)
        monkeypatch.setattr("vat.web.routes.bilibili.load_config", lambda: fake_config)
        monkeypatch.setattr("vat.web.routes.bilibili.Database", _FakeDatabase)
        monkeypatch.setattr("vat.web.routes.bilibili._get_uploader", lambda with_upload_params=False: object())
        monkeypatch.setattr(
            "vat.web.routes.bilibili.run_in_threadpool",
            lambda func, *args, **kwargs: __import__("asyncio").sleep(0, result=func(*args, **kwargs)),
        )
        monkeypatch.setattr(
            "vat.web.routes.bilibili.resync_season_video_infos",
            lambda db, uploader, config, season_id, delay_seconds=1.0: {
                "success": False,
                "season_id": season_id,
                "refreshed": 0,
                "failed": 0,
                "skipped": 0,
                "details": [],
                "message": "无法获取合集信息",
            },
            raising=False,
        )

        async with client as ac:
            response = await ac.post("/bilibili/season/42/resync-info")

        assert response.status_code == 200
        assert response.json() == {
            "success": False,
            "season_id": 42,
            "error": "无法获取合集信息",
            "refreshed": 0,
            "failed": 0,
            "skipped": 0,
            "details": [],
        }


class TestSyncSeasonTitlesRoute:
    @pytest.mark.anyio
    async def test_dispatches_season_title_sync_via_threadpool(self, app, client, monkeypatch):
        fake_config = SimpleNamespace(
            storage=SimpleNamespace(database_path="db.sqlite3", output_dir="outputs"),
        )
        fake_uploader = object()
        threadpool_call = {}
        fake_db = _FakeDatabase("db.sqlite3", "outputs")

        app.include_router(router)
        monkeypatch.setattr("vat.web.routes.bilibili.load_config", lambda: fake_config)
        monkeypatch.setattr("vat.web.routes.bilibili.get_db", lambda: fake_db)
        monkeypatch.setattr("vat.web.routes.bilibili._get_uploader", lambda with_upload_params=False: fake_uploader)

        def _should_not_be_called_directly(*args, **kwargs):
            raise AssertionError("sync_season_episode_titles_with_recovery 应通过线程池调度，而不是直接在 async 路由中调用")

        async def _fake_run_in_threadpool(func, *args, **kwargs):
            threadpool_call["func"] = func
            threadpool_call["args"] = args
            threadpool_call["kwargs"] = kwargs
            return {
                "success": True,
                "updated": 2,
                "skipped": 1,
                "details": [],
            }

        monkeypatch.setattr(
            "vat.web.routes.bilibili.sync_season_episode_titles_with_recovery",
            _should_not_be_called_directly,
            raising=False,
        )
        monkeypatch.setattr("vat.web.routes.bilibili.run_in_threadpool", _fake_run_in_threadpool)

        async with client as ac:
            response = await ac.post("/bilibili/season/42/sync-titles")

        assert response.status_code == 200
        assert response.json() == {
            "success": True,
            "updated": 2,
            "skipped": 1,
            "message": "合集 42 标题同步完成：2 个已更新，1 个已同步",
        }
        assert threadpool_call["func"] is _should_not_be_called_directly
        assert threadpool_call["args"][0] is fake_db
        assert threadpool_call["args"][1] is fake_uploader
        assert threadpool_call["args"][2] == 42


class TestBilibiliJobStatusRoutes:
    @pytest.mark.anyio
    async def test_fix_status_can_resolve_job_without_legacy_memory_state(self, app, client, monkeypatch):
        job_manager = _FakeJobManager()
        job_manager.jobs["job-fix"] = _make_job(
            "job-fix",
            task_type="fix-violation",
            task_params={"aid": 123},
            status=JobStatus.RUNNING,
        )

        app.include_router(router)
        monkeypatch.setattr("vat.web.routes.bilibili._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.bilibili._fix_tasks", {})

        async with client as ac:
            response = await ac.get("/bilibili/fix/123/status")

        assert response.status_code == 200
        assert response.json()["status"] == "masking"
        assert response.json()["job_id"] == "job-fix"
        assert job_manager.update_calls == ["job-fix"]

    @pytest.mark.anyio
    async def test_season_sync_status_can_resolve_job_without_legacy_memory_state(self, app, client, monkeypatch):
        job_manager = _FakeJobManager()
        job_manager.jobs["job-sync"] = _make_job(
            "job-sync",
            task_type="season-sync",
            task_params={"playlist_id": "PL1"},
            status=JobStatus.FAILED,
            error="sync failed",
        )

        app.include_router(router)
        monkeypatch.setattr("vat.web.routes.bilibili._get_job_manager", lambda: job_manager)
        monkeypatch.setattr("vat.web.routes.bilibili._sync_tasks", {})

        async with client as ac:
            response = await ac.get("/bilibili/season-sync/PL1/status")

        assert response.status_code == 200
        assert response.json()["status"] == "failed"
        assert response.json()["job_id"] == "job-sync"
        assert response.json()["message"] == "sync failed"
        assert job_manager.update_calls == ["job-sync"]
