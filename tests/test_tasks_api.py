"""tasks API 路由契约测试。"""

from datetime import datetime
import httpx
import pytest
from fastapi import FastAPI

from vat.models import TaskStep
from vat.web.jobs import JobStatus, WebJob
from vat.web.routes.tasks import router, get_job_manager, parse_steps


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


class _FakeJobManager:
    def __init__(self, cancel_result=True):
        self.cancel_result = cancel_result
        self.cancel_calls = []
        self.jobs = {}
        self.update_calls = []
        self.delete_calls = []
        self.submit_calls = []
        self.running_video_ids = set()

    def cancel_job(self, task_id):
        self.cancel_calls.append(task_id)
        return self.cancel_result

    def get_job(self, task_id):
        return self.jobs.get(task_id)

    def update_job_status(self, task_id):
        self.update_calls.append(task_id)

    def delete_job(self, task_id):
        self.delete_calls.append(task_id)
        return True

    def submit_job(self, **kwargs):
        self.submit_calls.append(kwargs)
        return "retry-new-task"

    def list_jobs(self, limit=50):
        return list(self.jobs.values())[:limit]

    def get_running_video_ids(self):
        return self.running_video_ids


def _make_job(task_id, status=JobStatus.COMPLETED, **overrides):
    data = dict(
        job_id=task_id,
        video_ids=["v1"],
        steps=["download"],
        gpu_device="auto",
        force=False,
        status=status,
        pid=1234 if status == JobStatus.RUNNING else None,
        log_file=None,
        progress=0.0,
        error=None,
        created_at=datetime(2026, 3, 14, 12, 0, 0),
        started_at=None,
        finished_at=None,
        upload_cron=None,
        concurrency=1,
        fail_fast=False,
        task_type="process",
        task_params={},
        cancel_requested=False,
    )
    data.update(overrides)
    return WebJob(**data)


class _FakeDb:
    def __init__(self):
        self.invalidate_calls = []
        self.completed_steps = {}
        self.videos = {}

    def invalidate_downstream_tasks(self, video_id, first_step):
        self.invalidate_calls.append((video_id, first_step))

    def is_step_completed(self, video_id, step):
        return self.completed_steps.get((video_id, step), False)

    def get_video(self, video_id):
        return self.videos.get(video_id)


class TestParseSteps:
    def test_parse_steps_expands_groups_and_deduplicates(self):
        assert parse_steps(["asr", "split", "translate", "translate"]) == [
            "whisper",
            "split",
            "translate",
        ]

    def test_parse_steps_keeps_unknown_step_names_for_later_validation(self):
        assert parse_steps(["download", "mystery-step"]) == ["download", "mystery-step"]


class TestExecuteTaskApi:
    @pytest.mark.anyio
    async def test_execute_task_rejects_upload_cron_for_non_upload_steps(self, app, client):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager

        payload = {
            "video_ids": ["v1"],
            "steps": ["download"],
            "upload_cron": "0 8 * * *",
        }

        async with client as ac:
            response = await ac.post("/api/tasks/execute", json=payload)

        assert response.status_code == 400
        assert "定时上传仅可用于 upload 阶段" in response.json()["detail"]
        assert job_manager.submit_calls == []

    @pytest.mark.anyio
    async def test_execute_task_rejects_running_video_conflicts(self, app, client):
        job_manager = _FakeJobManager()
        job_manager.running_video_ids = {"v1"}
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager

        payload = {
            "video_ids": ["v1", "v2"],
            "steps": ["download"],
        }

        async with client as ac:
            response = await ac.post("/api/tasks/execute", json=payload)

        assert response.status_code == 409
        assert "v1" in response.json()["detail"]
        assert job_manager.submit_calls == []

    @pytest.mark.anyio
    async def test_execute_task_force_invalidates_from_earliest_requested_step(self, app, client, monkeypatch):
        job_manager = _FakeJobManager()
        fake_db = _FakeDb()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        monkeypatch.setattr("vat.web.deps.get_db", lambda: fake_db)

        payload = {
            "video_ids": ["v1", "v2"],
            "steps": ["translate", "download"],
            "force": True,
        }

        async with client as ac:
            response = await ac.post("/api/tasks/execute", json=payload)

        assert response.status_code == 200
        assert fake_db.invalidate_calls == [
            ("v1", TaskStep.DOWNLOAD),
            ("v2", TaskStep.DOWNLOAD),
        ]
        assert job_manager.submit_calls[0]["steps"] == ["translate", "download"]

    @pytest.mark.anyio
    async def test_execute_task_returns_cli_preview_when_requested(self, app, client, monkeypatch):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        monkeypatch.setattr(
            "vat.web.routes.tasks._generate_cli_command",
            lambda request: f"cli:{','.join(request.steps)}",
        )

        payload = {
            "video_ids": ["v1"],
            "steps": ["asr"],
            "generate_cli": True,
            "concurrency": 2,
        }

        async with client as ac:
            response = await ac.post("/api/tasks/execute", json=payload)

        assert response.status_code == 200
        assert response.json()["status"] == "submitted"
        assert response.json()["cli_command"] == "cli:asr"
        assert job_manager.submit_calls[0]["steps"] == ["whisper", "split"]


class TestCancelTaskApi:
    @pytest.mark.anyio
    async def test_cancel_task_uses_to_thread_and_returns_cancelling(self, app, client, monkeypatch):
        job_manager = _FakeJobManager(cancel_result=True)
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager

        to_thread_calls = []

        async def fake_to_thread(func, *args, **kwargs):
            to_thread_calls.append((func, args, kwargs))
            return func(*args, **kwargs)

        monkeypatch.setattr("vat.web.routes.tasks.asyncio.to_thread", fake_to_thread)

        async with client as ac:
            response = await ac.post("/api/tasks/job-123/cancel")

        assert response.status_code == 200
        assert response.json() == {"status": "cancelling", "task_id": "job-123"}
        assert job_manager.cancel_calls == ["job-123"]
        assert len(to_thread_calls) == 1
        assert to_thread_calls[0][1] == ("job-123",)

    @pytest.mark.anyio
    async def test_cancel_task_returns_400_when_manager_rejects(self, app, client, monkeypatch):
        job_manager = _FakeJobManager(cancel_result=False)
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager

        async def fake_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        monkeypatch.setattr("vat.web.routes.tasks.asyncio.to_thread", fake_to_thread)

        async with client as ac:
            response = await ac.post("/api/tasks/job-404/cancel")

        assert response.status_code == 400
        assert response.json()["detail"] == "Task cannot be cancelled"
        assert job_manager.cancel_calls == ["job-404"]


class TestTaskQueryApi:
    @pytest.mark.anyio
    async def test_get_task_refreshes_status_before_returning(self, app, client):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        job_manager.jobs["job-1"] = _make_job("job-1", status=JobStatus.RUNNING, pid=4321)

        async with client as ac:
            response = await ac.get("/api/tasks/job-1")

        assert response.status_code == 200
        assert response.json()["task_id"] == "job-1"
        assert job_manager.update_calls == ["job-1"]

    @pytest.mark.anyio
    async def test_list_tasks_returns_serialized_jobs(self, app, client):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        job_manager.jobs["job-a"] = _make_job("job-a")
        job_manager.jobs["job-b"] = _make_job("job-b", status=JobStatus.FAILED)

        async with client as ac:
            response = await ac.get("/api/tasks")

        assert response.status_code == 200
        assert [item["task_id"] for item in response.json()] == ["job-a", "job-b"]


class TestDeleteTaskApi:
    @pytest.mark.anyio
    async def test_delete_task_refreshes_stale_running_job_before_rejecting(self, app, client):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        job_manager.jobs["job-stale"] = _make_job("job-stale", status=JobStatus.RUNNING, pid=4321)

        def refresh_to_completed(task_id):
            job_manager.update_calls.append(task_id)
            job_manager.jobs[task_id] = _make_job("job-stale", status=JobStatus.COMPLETED)

        job_manager.update_job_status = refresh_to_completed

        async with client as ac:
            response = await ac.delete("/api/tasks/job-stale")

        assert response.status_code == 200
        assert response.json() == {"status": "deleted", "task_id": "job-stale"}
        assert job_manager.delete_calls == ["job-stale"]
        assert job_manager.update_calls == ["job-stale"]

    @pytest.mark.anyio
    async def test_delete_task_rejects_running_job(self, app, client):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        job_manager.jobs["job-running"] = _make_job("job-running", status=JobStatus.RUNNING, pid=4321)

        async with client as ac:
            response = await ac.delete("/api/tasks/job-running")

        assert response.status_code == 400
        assert response.json()["detail"] == "Cannot delete running task. Cancel it first."
        assert job_manager.delete_calls == []
        assert job_manager.update_calls == ["job-running"]

    @pytest.mark.anyio
    async def test_delete_task_deletes_terminal_job(self, app, client):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        job_manager.jobs["job-done"] = _make_job("job-done", status=JobStatus.CANCELLED)

        async with client as ac:
            response = await ac.delete("/api/tasks/job-done")

        assert response.status_code == 200
        assert response.json() == {"status": "deleted", "task_id": "job-done"}
        assert job_manager.delete_calls == ["job-done"]
        assert job_manager.update_calls == ["job-done"]


class TestRetryTaskApi:
    @pytest.mark.anyio
    async def test_retry_task_refreshes_stale_running_job_before_rejecting(self, app, client):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        job_manager.jobs["job-stale"] = _make_job("job-stale", status=JobStatus.RUNNING, pid=4321)

        def refresh_to_failed(task_id):
            job_manager.update_calls.append(task_id)
            job_manager.jobs[task_id] = _make_job("job-stale", status=JobStatus.FAILED)

        job_manager.update_job_status = refresh_to_failed

        async with client as ac:
            response = await ac.post("/api/tasks/job-stale/retry")

        assert response.status_code == 200
        assert response.json() == {
            "status": "submitted",
            "new_task_id": "retry-new-task",
            "original_task_id": "job-stale",
        }
        assert job_manager.update_calls == ["job-stale"]
        assert job_manager.submit_calls == [{
            "video_ids": ["v1"],
            "steps": ["download"],
            "gpu_device": "auto",
            "force": False,
            "concurrency": 1,
            "upload_cron": None,
            "fail_fast": False,
            "task_type": "process",
            "task_params": {},
        }]

    @pytest.mark.anyio
    async def test_retry_task_rejects_running_job(self, app, client):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        job_manager.jobs["job-running"] = _make_job("job-running", status=JobStatus.RUNNING, pid=4321)

        async with client as ac:
            response = await ac.post("/api/tasks/job-running/retry")

        assert response.status_code == 400
        assert response.json()["detail"] == "Task is still running"
        assert job_manager.submit_calls == []
        assert job_manager.update_calls == ["job-running"]

    @pytest.mark.anyio
    async def test_retry_task_resubmits_original_job_parameters(self, app, client):
        job_manager = _FakeJobManager()
        app.include_router(router)
        app.dependency_overrides[get_job_manager] = lambda: job_manager
        job_manager.jobs["job-original"] = _make_job(
            "job-original",
            status=JobStatus.FAILED,
            video_ids=["v1", "v2"],
            steps=["download", "translate"],
            gpu_device="cuda:1",
            force=True,
            upload_cron="0 8 * * *",
            concurrency=3,
            fail_fast=True,
            task_type="process",
            task_params={"playlist_id": "pl-1", "upload_mode": "dtime"},
        )

        async with client as ac:
            response = await ac.post("/api/tasks/job-original/retry")

        assert response.status_code == 200
        assert response.json() == {
            "status": "submitted",
            "new_task_id": "retry-new-task",
            "original_task_id": "job-original",
        }
        assert job_manager.update_calls == ["job-original"]
        assert job_manager.submit_calls == [{
            "video_ids": ["v1", "v2"],
            "steps": ["download", "translate"],
            "gpu_device": "cuda:1",
            "force": True,
            "concurrency": 3,
            "upload_cron": "0 8 * * *",
            "fail_fast": True,
            "task_type": "process",
            "task_params": {"playlist_id": "pl-1", "upload_mode": "dtime"},
        }]
