"""resolve_playlist_id: YouTube channel tab URL → 带后缀的 playlist ID"""

import pytest
from vat.services.playlist_service import resolve_playlist_id


# ── channel tab URL（/@handle 格式）──────────────────────────

@pytest.mark.parametrize("tab,suffix", [
    ("shorts", "-shorts"),
    ("videos", "-videos"),
    ("streams", "-streams"),
])
def test_handle_tab_url(tab, suffix):
    url = f"https://www.youtube.com/@ShirakamiFubuki/{tab}"
    assert resolve_playlist_id(url, "UCxxx") == f"UCxxx{suffix}"


# ── channel tab URL（/channel/UCxxx 格式）────────────────────

def test_channel_id_tab_url():
    url = "https://www.youtube.com/channel/UCdn5BQ06XqgXoAxIhbqw5Rg/shorts"
    assert resolve_playlist_id(url, "UCdn5BQ06XqgXoAxIhbqw5Rg") == "UCdn5BQ06XqgXoAxIhbqw5Rg-shorts"


# ── 已有后缀不重复追加 ───────────────────────────────────────

def test_no_double_suffix():
    url = "https://www.youtube.com/@ShirakamiFubuki/shorts"
    assert resolve_playlist_id(url, "UCxxx-shorts") == "UCxxx-shorts"


# ── 非 tab URL 不追加 ───────────────────────────────────────

def test_regular_playlist_url():
    url = "https://www.youtube.com/playlist?list=PLxxxxxxx"
    assert resolve_playlist_id(url, "PLxxxxxxx") == "PLxxxxxxx"


def test_channel_home_url():
    url = "https://www.youtube.com/@ShirakamiFubuki"
    assert resolve_playlist_id(url, "UCxxx") == "UCxxx"


# ── URL 带查询参数 / fragment ────────────────────────────────

def test_tab_url_with_query():
    url = "https://www.youtube.com/@ShirakamiFubuki/videos?view=0&sort=dd"
    assert resolve_playlist_id(url, "UCxxx") == "UCxxx-videos"


def test_tab_url_with_fragment():
    url = "https://www.youtube.com/@ShirakamiFubuki/streams#section"
    assert resolve_playlist_id(url, "UCxxx") == "UCxxx-streams"
