# VAT 模块文档：Upload（上传阶段）

> **阶段定义**：`TaskStep.UPLOAD` 是一个单一阶段（非阶段组）
>
> 职责：将嵌入字幕的视频上传到目标平台（当前仅支持 B 站），并管理合集、元信息同步等运营操作

---

## 1. 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          UPLOAD 阶段                                        │
│                          (TaskStep.UPLOAD)                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. 加载上传配置                                                     │   │
│  │    UploadConfigManager.load() → config/upload.yaml                  │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. 构建模板上下文                                                   │   │
│  │    build_upload_context(video, playlist_info)                        │   │
│  │    → 从 DB 记录中提取频道、翻译、模型等变量                          │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. 渲染上传元数据                                                   │   │
│  │    render_upload_metadata(video, templates, playlist_info)           │   │
│  │    → title, description（${变量} 已替换为实际值）                    │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 4. 上传视频                                                         │   │
│  │    BilibiliUploader.upload(video_path, title, desc, ...)            │   │
│  │    → biliup 库 Web 端 API → 返回 bvid/aid                          │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 5. 后续操作（可选）                                                  │   │
│  │    ├─ 添加到合集: add_to_season(aid, season_id)                     │   │
│  │    ├─ 合集排序: auto_sort_season(season_id)                         │   │
│  │    └─ 更新 DB: bilibili_aid, bilibili_bvid 等                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块组成

| 文件 | 职责 |
|------|------|
| `base.py` | 上传器抽象基类 `BaseUploader`，定义 upload/validate_credentials/get_upload_limit 接口 |
| `bilibili.py` | B 站上传器 `BilibiliUploader` 实现，含合集管理、违规修复、元信息同步等完整功能 |
| `template.py` | 模板渲染引擎 `TemplateRenderer` + 上下文构建 `build_upload_context` |
| `upload_config.py` | 上传配置管理 `UploadConfigManager`，独立于主配置，支持 WebUI 在线编辑 |

---

## 3. 模板系统

### 3.1 工作原理

模板使用 `${变量名}` 语法进行变量替换。渲染流程：

```
config/upload.yaml 中的模板字符串
    │
    ▼
build_upload_context(video, playlist_info)
    │  从 video.metadata 提取所有可用变量
    ▼
TemplateRenderer.render(template, context)
    │  正则替换 ${变量名} → 实际值
    │  未定义的变量保留原样并输出 WARNING
    ▼
渲染后的 title / description
```

### 3.2 可用变量完整列表

变量值来源于 `build_upload_context()`（`vat/uploaders/template.py`）。

#### 基础信息

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `video_id` | YouTube 视频 ID | `dQw4w9WgXcQ` |
| `source_url` | 原视频链接 | `https://www.youtube.com/watch?v=...` |
| `today` | 渲染当天日期 | `2026-03-02` |

#### 频道信息

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `channel_name` | 频道/主播名称（可被 playlist 的 uploader_name 覆盖） | `白上フブキ` |
| `channel_id` | YouTube 频道 ID | `UCdn5BQ06XqgXoAxIhbqw5Rg` |
| `channel_url` | 频道链接 | `https://www.youtube.com/channel/...` |
| `uploader_name` | 上传者名（优先使用 playlist 自定义名） | `白上フブキ` |

#### 原始内容

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `original_title` | 原视频标题 | `【雑談】朝の雑談...` |
| `original_desc` | 原视频简介 | |
| `original_date` | 原发布日期（格式化） | `2026-01-15` |
| `original_date_raw` | 原发布日期（原始格式） | `20260115` |

#### 翻译内容

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `translated_title` | 翻译后标题 | `【杂谈】早上的闲聊...` |
| `translated_desc` | 翻译后简介 | |
| `tldr` | 简介摘要（LLM 生成） | `白上讨论了最近的...` |

#### 视频信息

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `duration` | 时长（HH:MM:SS 或 MM:SS） | `1:23:45` |
| `duration_seconds` | 时长秒数 | `5025` |
| `duration_minutes` | 时长分钟数（取整） | `84` |
| `thumbnail` | 缩略图 URL | |

#### 播放列表

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `playlist_name` | 播放列表名称 | `白上フブキ切片` |
| `playlist_index` | 在列表中的上传序号 | `42` |
| `playlist_id` | 播放列表 ID | |

#### 模型信息

从 `metadata['stage_models']` 提取，记录各处理阶段实际使用的模型。

| 变量名 | 说明 | 示例 |
|--------|------|------|
| `whisper_model` | Whisper ASR 模型 | `large-v3` |
| `split_model` | 断句 LLM 模型 | `gpt-4o-mini` |
| `optimize_model` | 优化 LLM 模型 | `kimi-k2.5` |
| `translate_model` | 翻译 LLM 模型 | `gemini-3-flash-preview` |
| `models_summary` | LLM 模型汇总（split + optimize + translate） | `gpt-4o-mini、kimi-k2.5、gemini-3-flash-preview` |

#### 自定义变量

在 `config/upload.yaml` 的 `custom_vars` 中定义，可在模板中直接使用：

```yaml
bilibili:
  templates:
    custom_vars:
      brand: 全熟
      project_url: https://github.com/112292454/VAT
```

模板中使用 `${brand}`、`${project_url}` 即可引用。

### 3.3 变量优先级

当上下文变量与自定义变量同名时，上下文变量（从视频数据提取的）优先级更高。

---

## 4. 配置

上传配置独立存储在 `config/upload.yaml`，与主配置 `config/config.yaml` 分离，支持 WebUI 在线编辑。

### 4.1 配置结构

```yaml
bilibili:
  copyright: 1              # 1=自制, 2=转载
  default_tid: 21            # 默认分区 ID
  default_tags:              # 默认标签
    - VTuber
    - hololive
  auto_cover: true           # 自动设置封面
  cover_source: thumbnail    # 封面来源（thumbnail=YouTube 缩略图）
  season_id: null            # 默认合集 ID（可选）
  templates:
    title: '【${channel_name}】${translated_title} | #${playlist_index}'
    description: '...'       # 支持 ${变量名} 的多行模板
    custom_vars:             # 自定义变量
      brand: 全熟
```

### 4.2 配置管理

`UploadConfigManager` 提供加载/保存/更新接口：

| 方法 | 说明 |
|------|------|
| `load()` | 从 YAML 加载配置，文件不存在时使用默认值 |
| `save()` | 保存当前配置到 YAML |
| `update_bilibili(updates)` | 部分更新 B 站配置 |
| `get_bilibili_dict()` | 获取 B 站配置字典 |

便捷函数：`load_upload_config()`、`save_upload_config()` 通过全局单例操作。

---

## 5. BilibiliUploader 功能一览

### 5.1 核心上传

| 方法 | 说明 |
|------|------|
| `upload(video_path, title, desc, ...)` | 上传视频到 B 站，返回 `UploadResult(bvid, aid)` |
| `upload_with_metadata(video_path, metadata)` | 兼容旧接口的 metadata 字典上传 |
| `validate_credentials()` | 验证 cookies 是否有效 |

底层使用 `biliup` 库的 Web 端 API。Cookie 通过 `scripts/bilibili_login.py` 获取。

### 5.2 合集管理

| 方法 | 说明 |
|------|------|
| `list_seasons()` | 获取用户的合集列表 |
| `create_season(title, description)` | 创建新合集 |
| `add_to_season(aid, season_id)` | 将视频添加到合集 |
| `remove_from_season(aids, season_id)` | 从合集移除视频 |
| `get_season_episodes(season_id)` | 获取合集内视频列表 |
| `sort_season_episodes(season_id, aids_in_order)` | 手动指定合集排序 |
| `auto_sort_season(season_id)` | 按标题中 `#数字` 自动排序 |

### 5.3 视频信息管理

| 方法 | 说明 |
|------|------|
| `edit_video_info(aid, title, desc, tags, tid)` | 编辑已上传视频的元信息 |
| `get_video_detail(aid)` | 获取视频详情（公共 API） |
| `get_archive_detail(aid)` | 获取稿件详情（创作中心 API，含 filename） |
| `get_my_videos(page, page_size)` | 获取稿件列表 |
| `bvid_to_aid(bvid)` | BV 号转 AV 号 |

### 5.4 违规修复

| 方法 | 说明 |
|------|------|
| `get_rejected_videos()` | 获取被退回稿件列表及违规详情 |
| `fix_violation(aid, video_path, ...)` | 自动修复违规视频：获取违规段→遮罩→替换上传 |
| `replace_video(aid, new_video_path)` | 替换稿件视频文件并重新提交审核 |
| `download_video(aid, output_path)` | 从 B 站下载视频（降级路径，质量低于原始文件） |

### 5.5 模块级函数

| 函数 | 说明 |
|------|------|
| `season_sync(db, uploader, playlist_id)` | 批量同步未入集视频到合集，含诊断检查 |
| `resync_video_info(db, uploader, config, aid)` | 从 DB 和模板重新渲染元信息并同步到 B 站 |
| `create_bilibili_uploader(config)` | 从配置创建上传器实例 |

---

## 6. CLI 命令

所有上传命令在 `vat upload` 子命令组下：

```bash
# 上传单个视频
vat upload video <video_id> -p <playlist_id> [--season <id>] [--dry-run]

# 批量上传 playlist 中的视频
vat upload playlist <playlist_id> [--season <id>] [--limit <n>] [--dry-run]

# 合集同步：将已上传但未入集的视频批量添加到合集
vat upload sync -p <playlist_id>

# 批量更新已上传视频的标题和简介（重新渲染模板）
vat upload update-info -p <playlist_id> [--dry-run] [--yes]

# 将 B 站合集信息同步回数据库
vat upload sync-db --season <season_id> -p <playlist_id> [--dry-run]
```

---

## 7. 关键代码索引

| 组件 | 文件位置 | 类/函数 |
|------|----------|---------|
| CLI 入口 | `vat/cli/commands.py` | `upload()` 命令组 |
| 阶段执行 | `vat/pipeline/executor.py` | `_run_upload()` |
| 上传器基类 | `vat/uploaders/base.py` | `BaseUploader` |
| B 站上传器 | `vat/uploaders/bilibili.py` | `BilibiliUploader` |
| 模板渲染 | `vat/uploaders/template.py` | `TemplateRenderer`, `build_upload_context` |
| 配置管理 | `vat/uploaders/upload_config.py` | `UploadConfigManager` |
| 配置文件 | `config/upload.yaml` | 模板定义 + 自定义变量 |
| WebUI 上传页 | `vat/web/routes/` | 上传相关路由 |

---

## 8. 注意事项

- **Cookie 获取**：B 站上传需要有效的 cookie，通过 `scripts/bilibili_login.py` 扫码获取，保存为 JSON 文件
- **biliup monkey-patch**：模块启动时对 `biliup` 库的 `upos` 方法打了补丁，修复 B 站 API 不再返回 `chunk_size` 字段导致的 KeyError
- **API 频率限制**：合集操作 API 有频率限制（code=20111），批量操作间需间隔 3 秒
- **desc 截断**：创作中心 API 返回的 desc 会被截断到 250 字符，编辑时需从公共 API 获取完整 desc
- **filename 来源**：编辑视频时 videos 数组中的 filename 必须使用创作中心 API 返回的真实值，否则报 code=21036
