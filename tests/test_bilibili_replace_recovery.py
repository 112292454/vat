"""replace_video 恢复模型契约测试。"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from vat.database import Database
from vat.models import Video, SourceType
from vat.services.bilibili_workflows import replace_video_with_recovery


def _make_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    return db, path


class _FakeUploader:
    def __init__(self, edit_ok=True):
        self.edit_ok = edit_ok
        self.context_calls = []
        self.upload_calls = []
        self.edit_calls = []

    def _load_replace_video_context(self, aid):
        self.context_calls.append(aid)
        return {
            "session": SimpleNamespace(),
            "bili_jct": "csrf-token",
            "archive": {
                "title": "原稿标题",
                "desc": "简介",
                "tag": "标签",
                "tid": 21,
                "copyright": 1,
                "source": "",
                "cover": "cover.jpg",
            },
            "old_videos": [{"title": "P1", "desc": "分P简介", "filename": "oldfile"}],
        }

    def _upload_replacement_file(self, new_video_path, old_video):
        self.upload_calls.append((str(new_video_path), old_video))
        return "uploaded-file-name"

    def _apply_replace_edit(self, *, session, bili_jct, aid, archive, old_videos, new_filename):
        self.edit_calls.append(
            {
                "aid": aid,
                "bili_jct": bili_jct,
                "new_filename": new_filename,
            }
        )
        return self.edit_ok


class TestReplaceVideoRecovery:
    def test_records_failed_state_after_upload_when_edit_fails(self, tmp_path):
        db, db_path = _make_db()
        try:
            video = Video(
                id="vid-replace",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=vid-replace",
                title="Video",
                metadata={"bilibili_aid": 12345},
            )
            db.add_video(video)

            uploader = _FakeUploader(edit_ok=False)
            new_video = tmp_path / "replacement.mp4"
            new_video.write_bytes(b"00")

            result = replace_video_with_recovery(db, uploader, 12345, new_video)

            assert result["success"] is False
            refreshed = db.get_video("vid-replace")
            op = refreshed.metadata["bilibili_ops"]["replace_video"]
            assert op["state"] == "failed"
            assert op["last_successful_state"] == "file_uploaded"
            assert op["uploaded_filename"] == "uploaded-file-name"
        finally:
            os.unlink(db_path)

    def test_records_verified_state_on_success(self, tmp_path):
        db, db_path = _make_db()
        try:
            video = Video(
                id="vid-replace-ok",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=vid-replace-ok",
                title="Video",
                metadata={"bilibili_aid": 54321},
            )
            db.add_video(video)

            uploader = _FakeUploader(edit_ok=True)
            new_video = tmp_path / "replacement.mp4"
            new_video.write_bytes(b"00")

            result = replace_video_with_recovery(db, uploader, 54321, new_video)

            assert result["success"] is True
            refreshed = db.get_video("vid-replace-ok")
            op = refreshed.metadata["bilibili_ops"]["replace_video"]
            assert op["state"] == "verified"
            assert op["last_successful_state"] == "edit_applied"
            assert op["uploaded_filename"] == "uploaded-file-name"
        finally:
            os.unlink(db_path)
