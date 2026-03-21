# Watch Metadata Investigation

## 原始目标

排查当前 `watch`/playlist sync 模式里，新视频是如何写入数据库的；确认为什么像 `DsKpoGf1qWA` 这类视频会缺少 `thumbnail`、`upload_date` 等完整元信息，并判断这是否由 download 失败、预约阶段、或其他 yt-dlp 提取问题导致。

## 当前计划

1. 还原 `watch -> sync_playlist -> add_video/update_video` 的实际写库路径。
2. 对本地数据库中的异常样本做取证，确认缺字段时数据库里留下了什么状态。
3. 直接复现 `get_playlist_info` / `get_video_info` 行为，确认是预约态失败还是提取客户端问题。
4. 用测试锁定修复：失败后持续重抓元信息，避免插值记录永久卡住。

## 当前状态

- 已确认 `watch` 本身不直接写 `videos`，而是调用 `PlaylistService.sync_playlist()`。
- 已确认 `sync_playlist()` 对“新视频”是先写基础记录，再并发调用 `get_video_info()` 补 `upload_date`/`thumbnail`。
- 已确认 `DsKpoGf1qWA` 在数据库中没有任何 `tasks`，说明“缺封面/缺完整信息”发生在 download 之前。
- 已确认 YouTube streams tab 的 flat entries 本身经常只带 `title/duration/live_status`，`thumbnail`/`upload_date`/`uploader` 为空，必须依赖后续 `get_video_info()` 补齐。
- 已确认当前默认 client 下，`get_video_info(DsKpoGf1qWA)` 报 `The page needs to be reloaded.`；切到 `mweb` client 后可成功拿到 `upload_date`、`thumbnail`、`uploader`。
- 已确认真正的预约视频（如 `lqm48iNoKOY`）在 `mweb` 下仍会返回 upcoming 错误，这类视频需要后续轮次继续重抓。
- 已确认当前 `sync_playlist()` 只会重抓 `upload_date` 为空的旧视频；一旦第一次失败后写入了 `upload_date_interpolated=true`，后续轮次就不会再重抓，因此半成品记录会永久残留。
- 已实现修复：
  - `get_video_info()` 在默认 client 命中 `The page needs to be reloaded` 时自动回退到 `mweb` client。
  - `sync_playlist()` 会对 `upload_date_interpolated=true`、`thumbnail=''`、`live_status in {is_upcoming,is_live}` 的旧视频继续补抓元信息。
  - 成功补抓后会回写最新 `live_status`，并清除 `upload_date_interpolated`。
- 已验证：
  - 本地对 `DsKpoGf1qWA`、`8MDlT-1PjAY`、`Oecin2VQi-k` 直接运行 `get_video_info()`，均可通过 `mweb` fallback 拿到真实 `upload_date`、`thumbnail`、`uploader`。
  - 回归测试 `tests/test_services.py tests/test_watch_service.py tests/test_youtube_downloader.py -q` 全部通过（`79 passed`）。
- 已进一步确认：
  - 当前数据库里 `members-only / Join this channel` 原因的条目数为 `0`，日志里出现的会员限定视频只是 sync 阶段临时探测到的 playlist entries，并未落库。
  - 当前实现会在 `_classify_pruned_unavailable_videos()` 中先剔除这类新视频，再进入真正的 `add_video()` / `add_video_to_playlist()` 写库阶段。
  - 当前数据库里真正残留的 `unavailable=1` 只有 3 条，原因都是旧的“预约/首播视频，无实际内容”标记，而不是会员限定。
  - 其中 `Jg8t3EtnHjU` 已经 7 个阶段全部完成，说明这 3 条是历史脏数据：旧状态没有在后续成功处理后被正确清理。
  - 用户明确要求：cookie 问题不允许通过降级逻辑掩盖，后续如继续处理应按配置错误/环境错误正面修复。
- 已定位 cookie 根因：
  - `cookies/www.youtube.com_cookies.txt` 的静态内容本身是合法 Netscape 文件。
  - yt-dlp 在 `YoutubeDL.__exit__()` 中会对 `cookiefile` 调 `cookiejar.save()`，即每次调用结束都会把 cookies 回写到源文件。
  - `sync_playlist()` / metadata 回填对 `get_video_info()` 使用多线程并发；多个 `YoutubeDL(...)` 同时写同一个 cookie 文件，会导致其他线程偶发读到半写入内容，从而报“does not look like a Netscape format cookies file”。
  - 修复方式不是降级为无 cookie，而是为每次 `get_video_info()` 调用生成隔离的临时 cookie 副本，调用后删除。
- 已完成数据库修正：
  - 为 `MDwkJVqui_M`、`rKMhl43RHo0` 增加 `processing_notes` 备注：主播误设了多年后的预约时间，保留条目，不按会员限定处理。
  - 清除了 `Jg8t3EtnHjU` 上历史遗留的 `unavailable` / `unavailable_reason`，让它按普通视频对待。
  - 使用修复后的 `get_video_info()` 对剩余不完整视频做定点回填，`92/92` 成功，数据库中不完整视频从 `95` 降到 `3`。
  - 现存仅剩 3 条特殊保留项：`lqm48iNoKOY`、`MDwkJVqui_M`、`rKMhl43RHo0`。
- 已继续确认 download 路径：
  - 真正的 `download()` 之前会先经过 `_extract_info_with_retry()`；此前这里仍可能因默认 client 触发 `The page needs to be reloaded` 而失败。
  - 本地实验表明：默认 download 选项对 `DsKpoGf1qWA` 会失败，但为下载路径加入 `player_client = [web, web_safari, tv, mweb]` 后，`yt-dlp.download(..., simulate=True)` 可成功返回 `0`。
  - 因此已把相同修复扩展到 download 路径：
    - `_get_ydl_opts()` 默认携带稳健 client 组合
    - `_extract_info_with_retry()`、`_download_with_retry()`、`_wait_for_stream_end()` 都使用隔离 cookie 副本
  - 回归测试已更新并通过，相关测试总数提升到 `82 passed`。

## 下一步

1. 如果希望将来继续自动回填，可在后续把 `lqm48iNoKOY` 也加上同类人工备注。
2. 如需进一步减少 mweb 的 `GVS PO Token` warning，可单独研究是否要配置 PO Token；这不影响当前 metadata 回填结果。
