"""sync_season_episode_titles 恢复模型契约测试。"""

import os
import tempfile

from vat.database import Database
from vat.models import Playlist, SourceType, Video
from vat.services.bilibili_workflows import sync_season_episode_titles_with_recovery


def _make_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    return db, path


def _seed_playlist(db: Database, *, playlist_id: str, season_id: int) -> None:
    playlist = Playlist(
        id=playlist_id,
        title="Playlist",
        source_url="https://youtube.com/playlist?list=PL_TEST",
        metadata={"upload_config": {"season_id": season_id}},
    )
    db.add_playlist(playlist)

    for index, aid in enumerate((301, 302, 303), start=1):
        video = Video(
            id=f"vid-{aid}",
            source_type=SourceType.YOUTUBE,
            source_url=f"https://youtube.com/watch?v={aid}",
            title=f"Video {aid}",
            metadata={"bilibili_aid": aid, "bilibili_target_season_id": season_id},
        )
        db.add_video(video)
        db.add_video_to_playlist(video.id, playlist_id, playlist_index=index, upload_order_index=index)


class _FakeSeasonUploader:
    def __init__(self, *, readd_results, sort_ok=True):
        self.readd_results = list(readd_results)
        self.sort_ok = sort_ok
        self.remove_calls = []
        self.add_calls = []
        self.sort_calls = []

    def get_season_episodes(self, season_id):
        return {
            "section_id": 10,
            "episodes": [
                {"id": 1, "aid": 301, "title": "Old1", "archiveTitle": "New1"},
                {"id": 2, "aid": 302, "title": "Old2", "archiveTitle": "New2"},
                {"id": 3, "aid": 303, "title": "Stable", "archiveTitle": "Stable"},
            ],
        }

    def remove_from_season(self, aids, season_id):
        self.remove_calls.append((list(aids), season_id))
        return True

    def add_to_season(self, aid, season_id):
        self.add_calls.append((aid, season_id))
        return self.readd_results.pop(0)

    def sort_season_episodes(self, season_id, aids_in_order):
        self.sort_calls.append((season_id, list(aids_in_order)))
        return self.sort_ok


class TestSyncSeasonEpisodeTitlesRecovery:
    def test_records_partial_readd_failure_in_playlist_metadata(self, monkeypatch):
        monkeypatch.setattr("vat.services.bilibili_workflows.time.sleep", lambda _seconds: None)

        db, db_path = _make_db()
        try:
            _seed_playlist(db, playlist_id="PL-sync", season_id=42)
            uploader = _FakeSeasonUploader(readd_results=[True, False])

            result = sync_season_episode_titles_with_recovery(
                db,
                uploader,
                season_id=42,
                delay_seconds=0.1,
            )

            assert result["success"] is False
            assert result["failed"] == [302]

            refreshed = db.get_playlist("PL-sync")
            op = refreshed.metadata["bilibili_ops"]["sync_season_episode_titles"]
            assert op["season_id"] == 42
            assert op["state"] == "failed"
            assert op["last_successful_state"] == "episodes_readded"
            assert op["original_order"] == [301, 302, 303]
            assert op["need_update_aids"] == [301, 302]
            assert op["readded_aids"] == [301]
            assert op["failed_aids"] == [302]
            assert uploader.sort_calls == []
        finally:
            os.unlink(db_path)

    def test_records_verified_state_after_order_restore(self, monkeypatch):
        monkeypatch.setattr("vat.services.bilibili_workflows.time.sleep", lambda _seconds: None)

        db, db_path = _make_db()
        try:
            _seed_playlist(db, playlist_id="PL-sync-ok", season_id=42)
            uploader = _FakeSeasonUploader(readd_results=[True, True], sort_ok=True)

            result = sync_season_episode_titles_with_recovery(
                db,
                uploader,
                season_id=42,
                delay_seconds=0.1,
            )

            assert result["success"] is True
            assert result["updated"] == 2
            assert result["skipped"] == 1

            refreshed = db.get_playlist("PL-sync-ok")
            op = refreshed.metadata["bilibili_ops"]["sync_season_episode_titles"]
            assert op["season_id"] == 42
            assert op["state"] == "verified"
            assert op["last_successful_state"] == "order_restored"
            assert op["original_order"] == [301, 302, 303]
            assert op["need_update_aids"] == [301, 302]
            assert op["readded_aids"] == [301, 302]
            assert op["failed_aids"] == []
            assert uploader.sort_calls == [(42, [301, 302, 303])]
        finally:
            os.unlink(db_path)
