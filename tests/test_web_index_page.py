"""web 首页页面契约测试。"""

from datetime import datetime

from fastapi.testclient import TestClient

from vat.models import DEFAULT_STAGE_SEQUENCE, SourceType, Video


class _FakeDb:
    def __init__(self):
        now = datetime(2026, 4, 5, 12, 0, 0)
        self.videos = [
            Video(
                id="v-processing",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=v-processing",
                title="处理中视频",
                created_at=now,
                metadata={},
            ),
            Video(
                id="v-todo",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=v-todo",
                title="待处理队列视频",
                created_at=now,
                metadata={},
            ),
            Video(
                id="v-normal",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=v-normal",
                title="普通视频",
                created_at=now,
                metadata={},
            ),
        ]

    def list_videos_paginated(self, **kwargs):
        exclude_video_ids = kwargs.get("exclude_video_ids") or set()
        videos = [video for video in self.videos if video.id not in exclude_video_ids]
        return {
            "videos": videos,
            "total": len(videos),
            "page": kwargs.get("page", 1),
            "total_pages": 1,
        }

    def list_playlists(self):
        return []

    def batch_get_video_progress(self, video_ids):
        return {
            video_id: {
                "progress": 0,
                "task_status": {
                    step.value: {"status": "pending", "error": None}
                    for step in DEFAULT_STAGE_SEQUENCE
                },
                "has_failed": False,
                "has_running": False,
            }
            for video_id in video_ids
        }

    def get_statistics(self):
        return {
            "total_videos": len(self.videos),
            "completed_videos": 0,
            "partial_completed_videos": 0,
            "running_videos": 0,
            "failed_videos": 0,
        }


class _FakeJobManager:
    def get_running_video_ids(self):
        return {"v-processing"}

    def get_running_and_queued_video_ids(self):
        return {"v-processing", "v-todo"}


def test_index_hide_processing_also_hides_videos_queued_in_running_job_task_params(monkeypatch):
    from vat.web import app as web_app_module
    from vat.web.routes import tasks as tasks_routes

    monkeypatch.setattr(web_app_module, "get_db", lambda: _FakeDb())
    monkeypatch.setattr(tasks_routes, "get_job_manager", lambda: _FakeJobManager())

    with TestClient(web_app_module.app) as client:
        response = client.get("/?hide_processing=1")

    assert response.status_code == 200
    assert "处理中视频" not in response.text
    assert "待处理队列视频" not in response.text
    assert "普通视频" in response.text
