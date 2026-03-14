"""B站合集排序相关契约测试。"""

from unittest.mock import MagicMock

from vat.uploaders.bilibili import BilibiliUploader


def _make_uploader():
    uploader = BilibiliUploader.__new__(BilibiliUploader)
    uploader.cookie_data = {"bili_jct": "csrf-token"}
    uploader.get_season_episodes = MagicMock()
    uploader._get_authenticated_session = MagicMock()
    return uploader


class TestExtractTitleIndex:
    def test_extracts_trailing_index(self):
        assert BilibiliUploader._extract_title_index("标题 | #42") == 42

    def test_returns_none_when_no_suffix_index(self):
        assert BilibiliUploader._extract_title_index("普通标题") is None


class TestAutoSortSeason:
    def test_skip_sort_when_newly_added_is_already_max_index(self):
        uploader = _make_uploader()
        uploader.get_season_episodes.return_value = {
            "episodes": [
                {"id": 1, "aid": 101, "title": "旧视频 | #1"},
                {"id": 2, "aid": 102, "title": "新视频 | #2"},
            ]
        }
        uploader.sort_season_episodes = MagicMock()

        result = uploader.auto_sort_season(55, newly_added_aid=102)

        assert result is True
        uploader.sort_season_episodes.assert_not_called()

    def test_calls_sort_when_current_order_differs_from_title_index_order(self):
        uploader = _make_uploader()
        uploader.get_season_episodes.return_value = {
            "episodes": [
                {"id": 10, "aid": 300, "title": "第三话 | #3"},
                {"id": 11, "aid": 100, "title": "第一话 | #1"},
                {"id": 12, "aid": 200, "title": "第二话 | #2"},
            ]
        }
        uploader.sort_season_episodes = MagicMock(return_value=True)

        result = uploader.auto_sort_season(56)

        assert result is True
        uploader.sort_season_episodes.assert_called_once_with(56, [100, 200, 300])

    def test_noop_when_current_order_already_correct(self):
        uploader = _make_uploader()
        uploader.get_season_episodes.return_value = {
            "episodes": [
                {"id": 10, "aid": 100, "title": "第一话 | #1"},
                {"id": 11, "aid": 200, "title": "第二话 | #2"},
            ]
        }
        uploader.sort_season_episodes = MagicMock()

        result = uploader.auto_sort_season(57)

        assert result is True
        uploader.sort_season_episodes.assert_not_called()


class TestSortSeasonEpisodes:
    def test_returns_false_when_requested_aid_missing(self):
        uploader = _make_uploader()
        uploader.get_season_episodes.return_value = {
            "section_id": 9,
            "episodes": [
                {"id": 1, "aid": 100, "title": "A"},
            ],
        }

        assert uploader.sort_season_episodes(66, [100, 200]) is False
        uploader._get_authenticated_session.assert_not_called()

    def test_appends_unlisted_existing_episodes_to_tail(self):
        uploader = _make_uploader()
        uploader.get_season_episodes.return_value = {
            "section_id": 9,
            "episodes": [
                {"id": 1, "aid": 100, "title": "A"},
                {"id": 2, "aid": 200, "title": "B"},
                {"id": 3, "aid": 300, "title": "C"},
            ],
        }
        session = MagicMock()
        session.post.return_value.json.return_value = {"code": 0}
        uploader._get_authenticated_session.return_value = session

        result = uploader.sort_season_episodes(66, [300, 100])

        assert result is True
        payload = session.post.call_args.kwargs["json"]
        assert payload["sorts"] == [
            {"id": 3, "sort": 1},
            {"id": 1, "sort": 2},
            {"id": 2, "sort": 3},
        ]
