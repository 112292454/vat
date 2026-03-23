"""videos API 路由契约测试。"""

import httpx
import pytest
from fastapi import FastAPI

from vat.models import Video, Task, TaskStep, TaskStatus, SourceType
from vat.web.routes.videos import router
from vat.web.deps import get_db


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


class _FakeDb:
    def __init__(self):
        self.videos = []
        self.tasks = {}
        self.progress = {}

    def list_videos(self, playlist_id=None):
        return self.videos

    def get_tasks(self, video_id):
        return self.tasks.get(video_id, [])

    def batch_get_video_progress(self, video_ids=None):
        return {vid: self.progress.get(vid, {"progress": 0}) for vid in (video_ids or [])}

    def get_video_playlists(self, video_id):
        return []

    def get_video(self, video_id):
        for video in self.videos:
            if video.id == video_id:
                return video
        return None


@pytest.fixture
def app():
    return FastAPI()


@pytest.fixture
def client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


class TestVideosApiProgress:
    @pytest.mark.anyio
    async def test_list_videos_uses_batch_progress_and_counts_skipped(self, app, client):
        fake_db = _FakeDb()
        video = Video(
            id="v1",
            source_type=SourceType.YOUTUBE,
            source_url="https://youtube.com/watch?v=v1",
            title="Video 1",
            metadata={},
        )
        fake_db.videos = [video]
        fake_db.tasks["v1"] = [
            Task(video_id="v1", step=TaskStep.DOWNLOAD, status=TaskStatus.COMPLETED),
            Task(video_id="v1", step=TaskStep.WHISPER, status=TaskStatus.SKIPPED),
        ]
        fake_db.progress["v1"] = {"progress": 28}

        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: fake_db

        async with client as ac:
            response = await ac.get("/api/videos")

        assert response.status_code == 200
        payload = response.json()
        assert payload["total"] == 1
        assert payload["videos"][0]["progress"] == 0.28
        assert payload["videos"][0]["tasks"][1]["status"] == "skipped"
