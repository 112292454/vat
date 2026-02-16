"""upload_order_index 单元测试

验证：
1. 全量重分配按 upload_date 正确排序（1=最旧, N=最新）
2. sync 时 update_video_playlist_info 不丢失 upload_order_index
3. 新视频加入后重分配正确插入
4. backfill 全量覆盖错误索引
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


def _get_service(db):
    """构造 PlaylistService（只注入 db，不需要 downloader）"""
    from vat.services.playlist_service import PlaylistService
    return PlaylistService(db=db, downloader=None)


class TestReassignUploadOrderIndices:
    """_reassign_upload_order_indices 全量重分配"""

    def test_basic_chronological_order(self, db):
        """按 upload_date 正确分配 1=最旧, N=最新"""
        _add_playlist(db)
        # 故意乱序添加
        _add_video_to_playlist(db, "v_new", "PL_TEST", 1, "20250301")   # 最新，YouTube idx=1
        _add_video_to_playlist(db, "v_mid", "PL_TEST", 2, "20240601")   # 中间
        _add_video_to_playlist(db, "v_old", "PL_TEST", 3, "20230101")   # 最旧，YouTube idx=3

        service = _get_service(db)
        messages = []
        service._reassign_upload_order_indices("PL_TEST", messages.append)

        # 验证：最旧=1, 中间=2, 最新=3
        v_old = db.get_video("v_old")
        v_mid = db.get_video("v_mid")
        v_new = db.get_video("v_new")
        assert v_old.metadata['upload_order_index'] == 1
        assert v_mid.metadata['upload_order_index'] == 2
        assert v_new.metadata['upload_order_index'] == 3

    def test_overwrites_wrong_indices(self, db):
        """覆盖已有的错误索引"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 2, "20230101")
        _add_video_to_playlist(db, "v2", "PL_TEST", 1, "20250101")

        # 手动设置错误的 upload_order_index
        meta1 = db.get_video("v1").metadata
        meta1['upload_order_index'] = 99  # 错误值
        db.update_video("v1", metadata=meta1)

        service = _get_service(db)
        service._reassign_upload_order_indices("PL_TEST", lambda x: None)

        assert db.get_video("v1").metadata['upload_order_index'] == 1  # 最旧→1
        assert db.get_video("v2").metadata['upload_order_index'] == 2  # 最新→2

    def test_idempotent(self, db):
        """重复调用不改变正确的索引"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 2, "20230101")
        _add_video_to_playlist(db, "v2", "PL_TEST", 1, "20250101")

        service = _get_service(db)
        service._reassign_upload_order_indices("PL_TEST", lambda x: None)

        # 第二次调用不应有更新
        messages = []
        service._reassign_upload_order_indices("PL_TEST", messages.append)
        # 如果没有更新，不会产生回调
        assert not any("更新" in m for m in messages)

    def test_new_video_interleaved(self, db):
        """新视频（时间上在已有视频之间）正确插入"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v_old", "PL_TEST", 3, "20230101")
        _add_video_to_playlist(db, "v_new", "PL_TEST", 1, "20250101")

        service = _get_service(db)
        service._reassign_upload_order_indices("PL_TEST", lambda x: None)
        assert db.get_video("v_old").metadata['upload_order_index'] == 1
        assert db.get_video("v_new").metadata['upload_order_index'] == 2

        # 添加一个时间上在中间的视频
        _add_video_to_playlist(db, "v_mid", "PL_TEST", 2, "20240601")
        service._reassign_upload_order_indices("PL_TEST", lambda x: None)

        assert db.get_video("v_old").metadata['upload_order_index'] == 1
        assert db.get_video("v_mid").metadata['upload_order_index'] == 2  # 正确插入
        assert db.get_video("v_new").metadata['upload_order_index'] == 3

    def test_same_date_stable_order(self, db):
        """相同 upload_date 的视频排序稳定"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v_a", "PL_TEST", 3, "20240101")
        _add_video_to_playlist(db, "v_b", "PL_TEST", 2, "20240101")
        _add_video_to_playlist(db, "v_c", "PL_TEST", 1, "20250101")

        service = _get_service(db)
        service._reassign_upload_order_indices("PL_TEST", lambda x: None)

        # 相同日期的两个视频应该有连续索引（具体顺序不重要，但不能跳号）
        idx_a = db.get_video("v_a").metadata['upload_order_index']
        idx_b = db.get_video("v_b").metadata['upload_order_index']
        idx_c = db.get_video("v_c").metadata['upload_order_index']
        assert {idx_a, idx_b} == {1, 2}  # 两个同日期的占据 1 和 2
        assert idx_c == 3  # 最新的是 3

    def test_empty_playlist(self, db):
        """空 playlist 不报错"""
        _add_playlist(db)
        service = _get_service(db)
        service._reassign_upload_order_indices("PL_TEST", lambda x: None)  # 不应报错


class TestUpdateVideoPlaylistInfoPreservesIndex:
    """update_video_playlist_info 不丢失 upload_order_index（Bug 1 回归测试）"""

    def test_update_playlist_index_preserves_order_index(self, db):
        """更新 playlist_index 时不抹掉 upload_order_index"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 5, "20230101")

        # 设置 upload_order_index
        db.update_playlist_video_order_index("PL_TEST", "v1", 42)

        # 模拟 sync 更新 playlist_index（YouTube 每次 sync 都变）
        db.update_video_playlist_info("v1", "PL_TEST", 99)

        # upload_order_index 应该保留
        pv_info = db.get_playlist_video_info("PL_TEST", "v1")
        assert pv_info['upload_order_index'] == 42
        assert pv_info['playlist_index'] == 99  # playlist_index 已更新

    def test_multiple_syncs_preserve_order_index(self, db):
        """多次 sync 都不丢失 upload_order_index"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 3, "20230101")
        db.update_playlist_video_order_index("PL_TEST", "v1", 1)

        # 模拟 3 次 sync，playlist_index 每次都不同
        for new_pl_idx in [2, 5, 1]:
            db.update_video_playlist_info("v1", "PL_TEST", new_pl_idx)
            pv_info = db.get_playlist_video_info("PL_TEST", "v1")
            assert pv_info['upload_order_index'] == 1, \
                f"upload_order_index 丢失！playlist_index={new_pl_idx}"


class TestBackfillUploadOrderIndex:
    """backfill_upload_order_index 全量重分配"""

    def test_backfill_overwrites_wrong_indices(self, db):
        """backfill 覆盖错误索引"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 2, "20230101")
        _add_video_to_playlist(db, "v2", "PL_TEST", 1, "20250101")

        # 设置错误的索引
        meta = db.get_video("v1").metadata
        meta['upload_order_index'] = 100
        db.update_video("v1", metadata=meta)

        service = _get_service(db)
        result = service.backfill_upload_order_index("PL_TEST")

        assert result['updated'] >= 1
        assert db.get_video("v1").metadata['upload_order_index'] == 1
        assert db.get_video("v2").metadata['upload_order_index'] == 2

    def test_backfill_fills_missing(self, db):
        """backfill 填充缺失的索引"""
        _add_playlist(db)
        _add_video_to_playlist(db, "v1", "PL_TEST", 2, "20230101")
        _add_video_to_playlist(db, "v2", "PL_TEST", 1, "20250101")
        # 不设置 upload_order_index，模拟旧数据

        service = _get_service(db)
        result = service.backfill_upload_order_index("PL_TEST")

        assert result['updated'] == 2
        assert db.get_video("v1").metadata['upload_order_index'] == 1
        assert db.get_video("v2").metadata['upload_order_index'] == 2


class TestPlaylistIndexVsUploadOrderIndex:
    """验证 playlist_index（YouTube 逆序）和 upload_order_index（时间正序）语义不混淆"""

    def test_youtube_index_is_reverse_of_upload_order(self, db):
        """YouTube playlist_index 1=最新，upload_order_index 1=最旧"""
        _add_playlist(db)
        # YouTube: index=1 是最新，index=3 是最旧
        _add_video_to_playlist(db, "newest", "PL_TEST", 1, "20250301")
        _add_video_to_playlist(db, "middle", "PL_TEST", 2, "20240601")
        _add_video_to_playlist(db, "oldest", "PL_TEST", 3, "20230101")

        service = _get_service(db)
        service._reassign_upload_order_indices("PL_TEST", lambda x: None)

        # upload_order_index 应该与 playlist_index 相反
        oldest = db.get_video("oldest")
        newest = db.get_video("newest")
        assert oldest.metadata['upload_order_index'] == 1   # 最旧=1
        assert newest.metadata['upload_order_index'] == 3   # 最新=3
        # 而 YouTube 的 playlist_index 是相反的
        pv_oldest = db.get_playlist_video_info("PL_TEST", "oldest")
        pv_newest = db.get_playlist_video_info("PL_TEST", "newest")
        assert pv_oldest['playlist_index'] == 3  # YouTube: 最旧=最大
        assert pv_newest['playlist_index'] == 1  # YouTube: 最新=1

    def test_large_playlist_all_indexed(self, db):
        """大 playlist（50 个视频）所有视频都获得正确的连续索引"""
        _add_playlist(db)
        for i in range(50):
            date = f"2024{(i // 28 + 1):02d}{(i % 28 + 1):02d}"  # 2024-01-01 ~ 2024-02-22
            _add_video_to_playlist(db, f"v_{i:03d}", "PL_TEST", 50 - i, date)

        service = _get_service(db)
        service._reassign_upload_order_indices("PL_TEST", lambda x: None)

        # 验证：所有视频都有索引，且索引是 1~50 的连续整数
        indices = []
        for i in range(50):
            meta = db.get_video(f"v_{i:03d}").metadata
            assert 'upload_order_index' in meta, f"v_{i:03d} 缺少 upload_order_index"
            indices.append(meta['upload_order_index'])

        assert sorted(indices) == list(range(1, 51)), "索引不是 1~50 的连续整数"

        # 验证按日期排序正确
        videos_sorted = sorted(
            [(db.get_video(f"v_{i:03d}").metadata.get('upload_date', ''),
              db.get_video(f"v_{i:03d}").metadata['upload_order_index'])
             for i in range(50)],
            key=lambda x: x[0]
        )
        prev_idx = 0
        for date, idx in videos_sorted:
            assert idx > prev_idx, f"索引未按日期递增: date={date}, idx={idx}, prev={prev_idx}"
            prev_idx = idx
