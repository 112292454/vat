"""Bilibili 合集批量刷新元信息契约测试。"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ["HOME"] = os.path.join(tempfile.gettempdir(), "vat-test-home")

from vat.uploaders import bilibili as bilibili_module


def _make_uploader():
    uploader = SimpleNamespace()
    uploader.get_season_episodes = MagicMock()
    return uploader


class TestResyncSeasonVideoInfos:
    def test_returns_error_when_season_episodes_unavailable(self):
        uploader = _make_uploader()
        uploader.get_season_episodes.return_value = None

        result = bilibili_module.resync_season_video_infos(
            db=object(),
            uploader=uploader,
            config=object(),
            season_id=42,
        )

        assert result == {
            "success": False,
            "season_id": 42,
            "refreshed": 0,
            "failed": 0,
            "skipped": 0,
            "details": [],
            "message": "无法获取合集信息",
        }

    def test_aggregates_results_and_sleeps_between_items(self, monkeypatch):
        uploader = _make_uploader()
        uploader.get_season_episodes.return_value = {
            "section_id": 9,
            "episodes": [
                {"aid": 101, "title": "A"},
                {"aid": 102, "title": "B"},
                {"id": 3, "title": "缺少 aid"},
            ],
        }

        resync_calls = []
        sleep_calls = []

        def _fake_resync(db, uploader, config, aid, callback=None):
            resync_calls.append(aid)
            if aid == 101:
                return {"success": True, "title": "新标题A", "message": "ok"}
            return {"success": False, "title": "", "message": "DB 中未找到"}

        monkeypatch.setattr(bilibili_module, "resync_video_info", _fake_resync)
        monkeypatch.setattr(bilibili_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

        result = bilibili_module.resync_season_video_infos(
            db=object(),
            uploader=uploader,
            config=object(),
            season_id=42,
            delay_seconds=1.5,
        )

        assert result["success"] is True
        assert result["season_id"] == 42
        assert result["refreshed"] == 1
        assert result["failed"] == 1
        assert result["skipped"] == 1
        assert result["message"] == "合集 42 元信息同步完成：成功 1，失败 1，跳过 1"
        assert resync_calls == [101, 102]
        assert sleep_calls == [1.5, 1.5]
        assert result["details"] == [
            {"aid": 101, "success": True, "title": "新标题A", "message": "ok"},
            {"aid": 102, "success": False, "title": "", "message": "DB 中未找到"},
            {"aid": None, "success": False, "title": "", "message": "合集条目缺少 aid，已跳过"},
        ]
