"""B站 Web API 契约测试。"""

import os
import tempfile
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

os.environ["HOME"] = os.path.join(tempfile.gettempdir(), "vat-test-home")

from vat.web.routes.bilibili import router


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


class TestResyncSeasonInfoRoute:
    @pytest.mark.anyio
    async def test_dispatches_batch_resync_via_threadpool(self, app, client, monkeypatch):
        fake_config = SimpleNamespace(
            storage=SimpleNamespace(database_path="db.sqlite3", output_dir="outputs"),
        )
        fake_uploader = object()
        threadpool_call = {}

        app.include_router(router)
        monkeypatch.setattr("vat.web.routes.bilibili.load_config", lambda: fake_config)
        monkeypatch.setattr("vat.web.routes.bilibili.Database", _FakeDatabase)
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
