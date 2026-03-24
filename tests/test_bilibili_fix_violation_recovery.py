"""fix_violation 恢复模型契约测试。"""

import os
import tempfile
from unittest.mock import MagicMock

from vat.database import Database
from vat.models import SourceType, Video
from vat.services.bilibili_workflows import fix_violation_with_recovery


def _make_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    return db, path


class _FakeUploader:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def fix_violation(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.result)


class TestFixViolationRecovery:
    def test_skips_persistence_when_db_has_no_real_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        uploader = _FakeUploader(
            {
                "success": True,
                "new_ranges": [(100, 110)],
                "all_ranges": [(100, 110)],
                "masked_path": None,
                "source": "local",
                "message": "修复完成，已重新提交审核",
            }
        )
        fake_db = MagicMock()

        result = fix_violation_with_recovery(
            fake_db,
            uploader,
            aid=12345,
            dry_run=False,
        )

        assert result["success"] is True
        assert list(tmp_path.iterdir()) == []

    def test_records_masked_stage_when_upload_replace_fails(self, tmp_path):
        db, db_path = _make_db()
        try:
            video = Video(
                id="vid-fix-fail",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=fix-fail",
                title="Video",
                metadata={"bilibili_aid": 12345},
            )
            db.add_video(video)

            uploader = _FakeUploader(
                {
                    "success": False,
                    "new_ranges": [(100, 110)],
                    "all_ranges": [(100, 110), (200, 210)],
                    "masked_path": str(tmp_path / "masked.mp4"),
                    "source": "local",
                    "message": "上传替换失败，遮罩文件已保留",
                }
            )

            result = fix_violation_with_recovery(
                db,
                uploader,
                aid=12345,
                video_path=tmp_path / "source.mp4",
                dry_run=False,
            )

            assert result["success"] is False
            refreshed = db.get_video("vid-fix-fail")
            op = refreshed.metadata["bilibili_ops"]["fix_violation"]
            assert op["state"] == "failed"
            assert op["last_successful_state"] == "masked_video_ready"
            assert op["all_ranges"] == [[100, 110], [200, 210]]
            assert op["masked_path"].endswith("masked.mp4")
        finally:
            os.unlink(db_path)

    def test_records_replacement_submitted_on_success(self, tmp_path):
        db, db_path = _make_db()
        try:
            video = Video(
                id="vid-fix-ok",
                source_type=SourceType.YOUTUBE,
                source_url="https://youtube.com/watch?v=fix-ok",
                title="Video",
                metadata={"bilibili_aid": 54321},
            )
            db.add_video(video)

            uploader = _FakeUploader(
                {
                    "success": True,
                    "new_ranges": [(100, 110)],
                    "all_ranges": [(100, 110)],
                    "masked_path": None,
                    "source": "local",
                    "message": "修复完成，已重新提交审核",
                    "upload_duration": 321,
                }
            )

            result = fix_violation_with_recovery(
                db,
                uploader,
                aid=54321,
                video_path=tmp_path / "source.mp4",
                dry_run=False,
            )

            assert result["success"] is True
            refreshed = db.get_video("vid-fix-ok")
            op = refreshed.metadata["bilibili_ops"]["fix_violation"]
            assert op["state"] == "replacement_submitted"
            assert op["last_successful_state"] == "replacement_submitted"
            assert op["all_ranges"] == [[100, 110]]
            assert op["upload_duration"] == 321
        finally:
            os.unlink(db_path)
