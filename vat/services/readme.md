# VAT 模块文档：Services（服务层）

> 提供 Playlist 级别的业务逻辑，衔接 Database 和 Downloader，是 CLI/WebUI 的核心业务层。

---

## 1. 模块组成

| 文件 | 职责 |
|------|------|
| `playlist_service.py` | Playlist 增量同步、视频排序、元信息刷新、批量重翻译等 |
| `watch_service.py` | Watch 模式主循环：监控 Playlist 新视频、筛选、提交处理任务、重试管理 |

---

## 2. PlaylistService

### 2.1 核心流程：sync_playlist

```
playlist_url (YouTube)
    │
    ├─ 1. downloader.get_playlist_info() → 获取 playlist 元数据 + entries
    │
    ├─ 2. 增量比对：识别新增视频 vs 已存在视频
    │      ├─ 新视频：先仅在内存中暂存，不立即写库
    │      └─ 已存在：记录待更新的 playlist_index / 待补抓的元信息
    │
    ├─ 3. 并行获取 upload_date / 可用性判定（10 并发）
    │      ├─ 成功：更新 metadata + 异步提交 LLM 翻译
    │      └─ 失败：日期插值 + 标记状态
    │          ├─ unavailable：永久不可用
    │          │   ├─ 已存在视频：保留视频本体，按现有语义标记 unavailable
    │          │   ├─ 本次新发现视频：直接从当前 playlist / DB 结果集中剔除，不分配索引
    │          │   └─ 历史中断残留且 `upload_order_index=0`：直接清理残留记录
    │          └─ error：临时失败（下次 sync 可重试）
    │
    ├─ 4. 统一写入 DB（只写允许保留的视频）
    │
    ├─ 5. 分配 upload_order_index（增量式，1=最旧）
    │
    └─ 返回 SyncResult(new_videos, existing_videos, total)
```

### 2.2 方法一览

#### Playlist 同步与管理

| 方法 | 说明 |
|------|------|
| `sync_playlist(url, ...)` | 增量同步 Playlist（核心方法） |
| `list_playlists()` | 列出所有 Playlist |
| `get_playlist(id)` | 获取单个 Playlist |
| `delete_playlist(id, delete_videos)` | 删除 Playlist（可选删除关联视频） |

#### 视频查询

| 方法 | 说明 |
|------|------|
| `get_playlist_videos(id, order_by)` | 获取 Playlist 下所有视频（支持按 upload_date/playlist_index/created_at 排序） |
| `get_pending_videos(id, target_step)` | 获取待处理视频（可按阶段过滤） |
| `get_completed_videos(id)` | 获取已完成所有阶段的视频 |

#### 批量操作

| 方法 | 说明 |
|------|------|
| `retranslate_videos(id)` | 重新翻译所有视频的标题/简介（更新翻译逻辑后使用） |
| `refresh_videos(id)` | 刷新视频元信息（封面、时长、日期等） |
| `backfill_upload_order_index(id)` | 全量重分配上传顺序索引（手动修复工具） |

---

## 3. 关键设计

### 3.1 upload_order_index vs playlist_index

| | `playlist_index` | `upload_order_index` |
|---|---|---|
| 语义 | YouTube 返回的顺序（1=最新） | 按 upload_date 排序（1=最旧） |
| 稳定性 | 每次 sync 可能变化 | 只增不改（增量分配） |
| 用途 | 仅用于对比新旧视频 | 上传到 B 站时的标题编号 `#N` |

### 3.2 日期插值

当 `get_video_info` 获取 upload_date 失败时，根据 Playlist 中已知日期的位置关系进行线性插值：

- **两侧有已知日期**：取中间值
- **仅有更旧日期**：近期发布，偏向今天（`today - min(7, gap/10)` 天）
- **仅有更新日期**：最旧视频，`newer_date - 1` 天

插值日期标记 `upload_date_interpolated: true`，成功获取真实日期时自动清除。

### 3.3 异步翻译

`_submit_translate_task` 将视频信息翻译提交到全局线程池（10 并发），避免阻塞 sync 主流程。翻译完成后结果存入 `video.metadata['translated']` 和 `video.metadata['_video_info']`。

---

## 4. CLI 命令

```bash
# 添加并同步 Playlist
vat playlist add <url>

# 手动同步
vat playlist sync <playlist_id>

# 刷新视频元信息
vat playlist refresh <playlist_id>

# 重新翻译所有视频信息
vat playlist retranslate <playlist_id>
```

---

## 5. 关键代码索引

| 组件 | 文件位置 | 调用者 |
|------|----------|--------|
| `PlaylistService` | `vat/services/playlist_service.py` | CLI `playlist` 命令组、WebUI playlist 路由 |
| `SyncResult` | `vat/services/playlist_service.py` | CLI/WebUI 同步结果展示 |
| `VideoInfoTranslator` | `vat/llm/video_info_translator.py` | `_submit_translate_task` 异步调用 |
| `YouTubeDownloader` | `vat/downloaders/youtube.py` | `get_playlist_info`、`get_video_info` |
| `WatchService` | `vat/services/watch_service.py` | CLI `watch` 命令、WebUI Watch Tab |

---

## 6. WatchService

自动监控 YouTube Playlist，发现新视频后提交全流程处理任务。详见 `docs/WATCH_MODE_SPEC.md`。

### 核心流程

```
watch 启动
  │
  ├─ for each playlist:
  │     1. sync_playlist() → 获取新视频列表
  │     2. _get_processable_videos() → 排除已完成/运行中/不可用/超重试上限
  │     3. _get_retry_candidates() → 之前失败的视频（未超重试上限）
  │     4. _submit_process_job() → subprocess.Popen 启动 vat process
  │     5. _record_round() → 记录本轮结果到 watch_rounds 表
  │
  ├─ sleep(interval)
  └─ 循环（除非 --once）
```

### 关键设计

- **不直接执行处理**：Watch 只负责发现和提交，处理由 `vat process` 子进程执行
- **重试机制**：失败视频在后续轮次自动重试，受 `max_retries` 限制
- **冲突检测**：启动时检查同 playlist 的已有 session，防止重复监控
- **状态追踪**：通过 `watch_sessions` + `watch_rounds` 表记录每轮结果
