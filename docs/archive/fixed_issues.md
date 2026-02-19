# VAT 已修复问题归档

从 `docs/known_issues.md` 迁移的已修复问题记录。

---

## ASR

### ASR-7: Whisper 分块重叠导致内容重复

- **现象**：ChunkedASR 分块处理时，相邻 chunk 的重叠窗口（默认 每600s重叠10s）使 Whisper 对同一音频区域产生两次识别结果。表现为：
  1. 内容重复：如原文 "スタンプ" 在 split 后变成 "スタンプスタンプ"
  2. ASS 字幕 crash：重叠时间段导致 ASS 渲染器卡死
  3. 1ms 残渣段：split 阶段尝试拆分重复内容产生极短碎片
- **根因**：`ChunkedASR._merge_segments` 仅按时间排序拼接，对重叠窗口产生的重复段没有去重
- **修复**：
  1. **`ChunkedASR._merge_segments` 新增 dedup 逻辑**：
     - Case 1: 时间完全包含（A⊃B 或 B⊃A）→ 保留较长段
     - Case 2: 时间重叠 ≥50%（相似内容）→ 合并为一段
     - Case 3: 时间重叠 <50%（不同内容，时序微偏）→ 保留两段，调整边界
  2. **`ASRData.__init__` 退化段过滤**：过滤 duration < 50ms 的 whisper 幽灵段
- **验证**：367 个视频全量扫描，修复后残余重叠 = 0，ASS crash 风险段 = 0
- **状态**：已修复。已有视频需从 split 阶段重跑才能消除内容重复（ASS crash 已自动修复）

---

## Split

### Split-1: 断句后偶发时间错位

- **现象**：经 split 后字幕偶尔出现轻微时间错位（延后），但无法稳定复现
- **状态**：已修复主要问题（字符级时间插值 + 重叠消除），不排除边缘情况仍有微小偏差

### Split-2: Whisper 分块重叠导致 split 输入含重复段

- **现象**：当 whisper 原始输出包含分块边界重叠段时，LLM split 收到重复内容作为输入，导致断句结果中出现内容重复和 1ms 残渣段
- **状态**：已修复。根因在 ASR 层 dedup 不充分，详见 ASR-7

---

## B 站上传

### Upload-1: 添加合集不稳定

- **现象**：视频上传成功，但添加到合集始终失败（API 返回 code=0 但实际未添加）
- **根因**：B站创作中心 `episodes/add` 接口的参数格式与常规 REST API 不同——需要 `sectionId`（驼峰）+ `episodes`（含 aid/cid/title 的对象数组）+ JSON body + csrf 同时在 query string 和 body 中。之前使用的 `{id, aids}` 格式返回"空成功"（code=0 但不生效）
- **修复**：通过逆向 B站创作中心前端 JS 找到正确格式，重写 `add_to_season`、`remove_from_season`。详见 `docs/bilibili_season_api.md`
- **排序功能**：`sort_season_episodes` 已修复。关键是 `section` 对象必须包含 `id/type/seasonId/title` 四个字段，且 body 中不能有 `csrf`

### Upload-2: Playlist 同步不获取新视频 + 视频/直播混合问题

- **现象**：fubuki 频道的 playlist 同步后视频数量不增长（始终 2948），且视频和直播混在同一个 playlist 中
- **根因**：
  1. `source_url` 指向 `/@xxx/videos` tab，YouTube 该 tab 只返回 ~172 个热门视频（非全量），导致 sync 无法发现新视频
  2. 单一 playlist 混合了 2668 个直播和 280 个普通视频，语义不清晰，且 yt-dlp 对 `/videos` 和 `/streams` tab 返回相同的 channel ID，无法区分
- **修复**：
  1. **Playlist 拆分**：将原 `UCdn5BQ06XqgXoAxIhbqw5Rg` 拆为 `-videos`（280 个）和 `-streams`（2668 个）
  2. **Videos playlist**：source_url 改为 UU uploads playlist（全量上传列表），metadata 中 `sync_live_filter: "not_live"` 自动过滤 `live_status=was_live` 的直播条目
  3. **Streams playlist**：source_url 使用 `/@xxx/streams`（只返回直播）
  4. **`sync_playlist` 新增 `target_playlist_id` 参数**：显式指定 DB 中的 playlist ID，不再依赖 yt-dlp 返回的（可能不唯一的）playlist ID
  5. **`get_playlist_info` 新增 `live_status` 字段**：从 yt-dlp flat 模式中提取，用于 sync 时自动区分视频和直播
  6. B 站合集同步更新：拆分后重算 `upload_order_index`，批量更新 39 个已上传直播的 B 站标题中的 `#N` 编号

### Upload-3: 视频编号（#N）与时间顺序不一致

- **现象**：上传标题中的 `#N` 编号与视频的实际发布时间顺序不符。例如倒数第 2 新的视频显示 `#3` 而非 `#29`
- **根因**：4 个 bug 叠加导致
  1. `database.update_video_playlist_info` 使用 `INSERT OR REPLACE`，每次 sync 都**删除旧行再插入新行**，丢失已分配的 `upload_order_index`
  2. `_assign_upload_order_index` 只处理本次 sync 新增的视频，已有视频即使索引丢失也不会重新分配
  3. `executor` 上传时 `upload_order_index` 缺失会回退到 `playlist_index`（YouTube 的 1=最新逆序），语义与 `upload_order_index`（1=最旧正序）**完全相反**
  4. `backfill_upload_order_index` 保留已有的错误索引，不做全量修正
- **修复**：
  1. `update_video_playlist_info` 改用 `ON CONFLICT UPDATE`，只更新 `playlist_index`，保留 `upload_order_index`
  2. 重写为 `_reassign_upload_order_indices`：每次 sync 全量按 `upload_date` 排序分配 `1~N`
  3. 移除 executor 中对 `playlist_index` 的错误回退
  4. `backfill` 改为全量重分配
- **验证**：12 个新增测试 + 8 个 playlist 全量数据验证通过

### Upload-4: 创作中心 API 截断 desc 导致编辑操作覆盖完整简介

- **现象**：通过 `edit_video_info` 修改标题/标签，或 `replace_video` 替换视频后，B站上的视频简介被截断为 250 字符
- **根因**：创作中心 API (`/x/client/archive/view`) 返回的 `desc` 字段**固定截断到 250 字符**，但编辑接口 (`/x/vu/web/edit`) 会用 payload 中的 `desc` 覆盖完整简介。公共 API (`/x/web-interface/view`) 对已发布视频返回完整 desc（1000+ 字符）
- **影响范围**：
  - `edit_video_info`：每次调用都会将 desc 从创作中心 250 字截断值写回
  - `replace_video`：替换视频时同样使用截断的 desc
  - 实际影响：38 个视频 desc 被截断（通过批量重渲染模板恢复），19 个视频标题 `#N` 未同步（已批量修复）
- **修复**：
  1. `edit_video_info`：改为从公共 API (`get_video_detail`) 获取完整 desc，仅在公共 API 不可用时回退到创作中心的截断值
  2. `replace_video`：新增 `_get_full_desc` 方法，优先从公共 API 获取完整 desc
- **验证**：73 个 pipeline 视频全量验证，标题 #N 和 desc 完整性全部通过
