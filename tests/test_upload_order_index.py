"""upload_order_index 单元测试

验证：
1. _assign_indices_to_new_videos: 增量式分配（只处理 index=0 的新视频，从 max+1 开始）
2. sync 时 update_video_playlist_info 不丢失 upload_order_index
3. backfill_upload_order_index: 全量重排（手动修复工具）
4. 索引只存在 playlist_videos 表中，不写 video.metadata
"""

import os
import tempfile
import pytest
from vat.database import Database
from vat.models import Video, Playlist, SourceType


@pytest.fixture
def db():
    """创建临时数据库，测试结束后删除"""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    database = Database(path)
    yield database
    os.unlink(path)


def _add_playlist(db, playlist_id="PL_TEST", title="Test Playlist"):
    pl = Playlist(id=playlist_id, title=title, source_url=f"https://youtube.com/playlist?list={playlist_id}")
    db.add_playlist(pl)
    return pl


def _add_video_to_playlist(db, video_id, playlist_id, playlist_index, upload_date, title=None):
    """辅助：创建视频并关联到 playlist"""
    v = Video(
        id=video_id,
        source_type=SourceType.YOUTUBE,
        source_url=f"https://youtube.com/watch?v={video_id}",
        title=title or video_id,
        playlist_id=playlist_id,
        playlist_index=playlist_index,
        metadata={'upload_date': upload_date},
    )
    db.add_video(v)
    db.add_video_to_playlist(video_id, playlist_id, playlist_index)
    return v


def _get_order_index(db, playlist_id, video_id):
    """从 playlist_videos 关联表读取 upload_order_index"""
    pv = db.get_playlist_video_info(playlist_id, video_id)
    return pv['upload_order_index'] if pv else 0


def _get_service(db):
    """构造 PlaylistService（只注入 db，不需要 downloader）"""
    from vat.services.playlist_service import PlaylistService
    return PlaylistService(db=db, downloader=None)


class TestAssignIndicesToNewVideos:
    """_assign_indices_to_new_videos 增量式分配"""

    def test_all_new_videos_get_indices(self, db):
        """所有新视频（index=0）按 upload_date 分配，1=最旧"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v_new", "PL_TEST", 1, "20250301")
        _add_video_to_playlist(db, "v_mid", "PL_TEST", 2, "20240601")
        _add_video_to_playlist(db, "v_old", "PL_TEST", 3, "20230101")

        service = _get_service(db)
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

        assert _get_order_index(db, "PL_TEST", "v_old") == 1
        assert _get_order_index(db, "PL_TEST", "v_mid") == 2
        assert _get_order_index(db, "PL_TEST", "v_new") == 3

    def test_only_new_videos_assigned(self, db):
        """已有索引的视频不变，只给 index=0 的新视频分配"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v_old", "PL_TEST", 2, "20230101")
        _add_video_to_playlist(db, "v_exist", "PL_TEST", 1, "20240601")

        # 手动给 v_old 和 v_exist 设置已有索引
        db.update_playlist_video_order_index("PL_TEST", "v_old", 1)
        db.update_playlist_video_order_index("PL_TEST", "v_exist", 2)

        # 添加一个新视频（index=0）
        _add_video_to_playlist(db, "v_new", "PL_TEST", 3, "20250101")

        service = _get_service(db)
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

        # 已有的不变
        assert _get_order_index(db, "PL_TEST", "v_old") == 1
        assert _get_order_index(db, "PL_TEST", "v_exist") == 2
        # 新的从 max+1=3 开始
        assert _get_order_index(db, "PL_TEST", "v_new") == 3

    def test_does_not_touch_existing_indices(self, db):
        """即使已有索引"看起来不对"（有缝隙），也不重排"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 1, "20230101")
        _add_video_to_playlist(db, "v2", "PL_TEST", 2, "20240101")

        # 故意设置有缝隙的索引：1, 5（跳了 2,3,4）
        db.update_playlist_video_order_index("PL_TEST", "v1", 1)
        db.update_playlist_video_order_index("PL_TEST", "v2", 5)

        service = _get_service(db)
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

        # 索引不变（没有新视频要分配）
        assert _get_order_index(db, "PL_TEST", "v1") == 1
        assert _get_order_index(db, "PL_TEST", "v2") == 5

    def test_multiple_new_videos_sorted_by_date(self, db):
        """多个新视频按 upload_date 排序，从 max+1 开始"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v_exist", "PL_TEST", 1, "20220101")
        db.update_playlist_video_order_index("PL_TEST", "v_exist", 10)

        # 3 个新视频，乱序添加
        _add_video_to_playlist(db, "v_c", "PL_TEST", 2, "20250101")
        _add_video_to_playlist(db, "v_a", "PL_TEST", 3, "20230101")
        _add_video_to_playlist(db, "v_b", "PL_TEST", 4, "20240601")

        service = _get_service(db)
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

        assert _get_order_index(db, "PL_TEST", "v_exist") == 10  # 不变
        assert _get_order_index(db, "PL_TEST", "v_a") == 11  # 最旧的新视频
        assert _get_order_index(db, "PL_TEST", "v_b") == 12
        assert _get_order_index(db, "PL_TEST", "v_c") == 13  # 最新的新视频

    def test_idempotent(self, db):
        """重复调用不改变已分配的索引"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 2, "20230101")
        _add_video_to_playlist(db, "v2", "PL_TEST", 1, "20250101")

        service = _get_service(db)
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

        idx1_first = _get_order_index(db, "PL_TEST", "v1")
        idx2_first = _get_order_index(db, "PL_TEST", "v2")

        # 第二次调用不应改变
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

        assert _get_order_index(db, "PL_TEST", "v1") == idx1_first
        assert _get_order_index(db, "PL_TEST", "v2") == idx2_first

    def test_empty_playlist(self, db):
        """空 playlist 不报错"""
        _add_playlist(db)
        service = _get_service(db)
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

    def test_does_not_write_to_video_metadata(self, db):
        """索引只写 playlist_videos 表，不写 video.metadata"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 1, "20230101")

        service = _get_service(db)
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

        # playlist_videos 表有索引
        assert _get_order_index(db, "PL_TEST", "v1") == 1
        # video.metadata 不应有 upload_order_index
        meta = db.get_video("v1").metadata
        assert 'upload_order_index' not in meta


class TestUpdateVideoPlaylistInfoPreservesIndex:
    """update_video_playlist_info 不丢失 upload_order_index（Bug 1 回归测试）"""

    def test_update_playlist_index_preserves_order_index(self, db):
        """更新 playlist_index 时不抹掉 upload_order_index"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 5, "20230101")

        db.update_playlist_video_order_index("PL_TEST", "v1", 42)
        db.update_video_playlist_info("v1", "PL_TEST", 99)

        pv_info = db.get_playlist_video_info("PL_TEST", "v1")
        assert pv_info['upload_order_index'] == 42
        assert pv_info['playlist_index'] == 99

    def test_multiple_syncs_preserve_order_index(self, db):
        """多次 sync 都不丢失 upload_order_index"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 3, "20230101")
        db.update_playlist_video_order_index("PL_TEST", "v1", 1)

        for new_pl_idx in [2, 5, 1]:
            db.update_video_playlist_info("v1", "PL_TEST", new_pl_idx)
            pv_info = db.get_playlist_video_info("PL_TEST", "v1")
            assert pv_info['upload_order_index'] == 1, \
                f"upload_order_index 丢失！playlist_index={new_pl_idx}"


class TestBackfillUploadOrderIndex:
    """backfill_upload_order_index 全量重排（手动修复工具）"""

    def test_backfill_overwrites_wrong_indices(self, db):
        """backfill 覆盖错误索引"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 2, "20230101")
        _add_video_to_playlist(db, "v2", "PL_TEST", 1, "20250101")

        # 设置错误的索引
        db.update_playlist_video_order_index("PL_TEST", "v1", 99)

        service = _get_service(db)
        result = service.backfill_upload_order_index("PL_TEST")

        assert result['updated'] >= 1
        assert _get_order_index(db, "PL_TEST", "v1") == 1
        assert _get_order_index(db, "PL_TEST", "v2") == 2

    def test_backfill_fills_missing(self, db):
        """backfill 填充缺失的索引"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 2, "20230101")
        _add_video_to_playlist(db, "v2", "PL_TEST", 1, "20250101")

        service = _get_service(db)
        result = service.backfill_upload_order_index("PL_TEST")

        assert result['updated'] == 2
        assert _get_order_index(db, "PL_TEST", "v1") == 1
        assert _get_order_index(db, "PL_TEST", "v2") == 2

    def test_backfill_returns_changed_videos(self, db):
        """backfill 返回变更列表"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 2, "20230101")
        _add_video_to_playlist(db, "v2", "PL_TEST", 1, "20250101")
        db.update_playlist_video_order_index("PL_TEST", "v1", 99)

        service = _get_service(db)
        result = service.backfill_upload_order_index("PL_TEST")

        assert 'changed_videos' in result
        # v1: 99->1, v2: 0->2
        changed_ids = {vid for vid, _, _ in result['changed_videos']}
        assert 'v1' in changed_ids

    def test_backfill_does_not_write_to_video_metadata(self, db):
        """backfill 只写 playlist_videos 表，不写 video.metadata"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 1, "20230101")

        service = _get_service(db)
        service.backfill_upload_order_index("PL_TEST")

        assert _get_order_index(db, "PL_TEST", "v1") == 1
        meta = db.get_video("v1").metadata
        assert 'upload_order_index' not in meta


class TestPlaylistIndexVsUploadOrderIndex:
    """验证 playlist_index（YouTube 逆序）和 upload_order_index（时间正序）语义不混淆"""

    def test_youtube_index_is_reverse_of_upload_order(self, db):
        """YouTube playlist_index 1=最新，upload_order_index 1=最旧"""
        _add_playlist(db)
        _add_video_to_playlist(db, "newest", "PL_TEST", 1, "20250301")
        _add_video_to_playlist(db, "middle", "PL_TEST", 2, "20240601")
        _add_video_to_playlist(db, "oldest", "PL_TEST", 3, "20230101")

        service = _get_service(db)
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

        assert _get_order_index(db, "PL_TEST", "oldest") == 1
        assert _get_order_index(db, "PL_TEST", "newest") == 3

        pv_oldest = db.get_playlist_video_info("PL_TEST", "oldest")
        pv_newest = db.get_playlist_video_info("PL_TEST", "newest")
        assert pv_oldest['playlist_index'] == 3
        assert pv_newest['playlist_index'] == 1

    def test_large_playlist_all_indexed(self, db):
        """大 playlist（50 个视频）所有视频都获得正确的连续索引"""
        _add_playlist(db)
        for i in range(50):
            date = f"2024{(i // 28 + 1):02d}{(i % 28 + 1):02d}"
            _add_video_to_playlist(db, f"v_{i:03d}", "PL_TEST", 50 - i, date)

        service = _get_service(db)
        service._assign_indices_to_new_videos("PL_TEST", lambda x: None)

        indices = []
        for i in range(50):
            idx = _get_order_index(db, "PL_TEST", f"v_{i:03d}")
            assert idx > 0, f"v_{i:03d} 缺少 upload_order_index"
            indices.append(idx)

        assert sorted(indices) == list(range(1, 51)), "索引不是 1~50 的连续整数"
