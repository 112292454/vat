# VAT 模块文档：Download（下载阶段）

> **阶段定义**：`TaskStep.DOWNLOAD` 是一个单一阶段（非阶段组）
> 
> 职责：从外部来源拉取视频文件，为后续 ASR/翻译/嵌入阶段准备输入

---

## 1. 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOWNLOAD 阶段                                        │
│                         (TaskStep.DOWNLOAD)                                  │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CLI / Pipeline 入口                               │    │
│  │  vat download -u <url>  或  vat pipeline -u <url>                   │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1. create_video_from_url()                                          │    │
│  │    - 解析 URL，识别来源类型 (YouTube/Local/...)                      │    │
│  │    - 生成内部 video_id = md5(url)[:16]                              │    │
│  │    - 创建 DB 记录 (Video)                                           │    │
│  │    - 创建输出目录 <output_dir>/<video_id>/                          │    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 2. _download()                                                      │    │
│  │    │                                                                │    │
│  │    ├─ 根据 source_type 选择下载器                                   │    │
│  │    │      YouTube → YouTubeDownloader                               │    │
│  │    │      Local   → 跳过下载                                        │    │
│  │    │                                                                │    │
│  │    ├─ downloader.download(url, output_dir)                          │    │
│  │    │      ├─ yt-dlp 下载视频                                        │    │
│  │    │      ├─ 返回 video_path, title, metadata                       │    │
│  │    │      └─ 保存为 <youtube_id>.<ext>                              │    │
│  │    │                                                                │    │
│  │    ├─ 更新 DB: Video.title, Video.metadata                          │    │
│  │    │                                                                │    │
│  │    └─ Metadata 增强 (LLM)                                           │    │
│  │           ├─ SceneIdentifier.detect_scene() → metadata['scene']     │    │
│  │           └─ VideoInfoTranslator.translate() → metadata['translated']│    │
│  └───────────────────────────────┬─────────────────────────────────────┘    │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 输出文件                                                            │    │
│  │   <output_dir>/<video_id>/                                          │    │
│  │       └─ <youtube_id>.mp4  (或其他格式)                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段定义与依赖

| 阶段名 | TaskStep 枚举 | 依赖 | 输出 |
|--------|---------------|------|------|
| **DOWNLOAD** | `TaskStep.DOWNLOAD` | 无 | 视频文件 + DB 记录 |

```python
# vat/models.py
STAGE_DEPENDENCIES = {
    TaskStep.DOWNLOAD: [],  # 无依赖，是 pipeline 起点
}
```

---

## 3. 模块职责与边界

Download 阶段负责：
- 从外部来源（当前主要是 **YouTube**）拉取视频文件到该 `video_id` 的输出目录。
- 为后续阶段准备 **可被 `_find_video_file()` 找到的视频文件**。
- 在下载完成后，补全/增强数据库中的 `Video.metadata`（场景识别、视频信息翻译等）。

明确边界：
- Download 阶段**不做字幕相关处理**。
- Download 阶段的“是否跳过/是否重跑”主要由 **数据库的 step 完成状态**控制。

---

## 2. 入口与调用链（从 CLI 到核心函数）

### 2.1 CLI 入口
- `python -m vat download -u <url>`
- `python -m vat download -p <playlist_url>`
- `python -m vat download -f <url_list_file>`

实现位置：@`vat/cli/commands.py` `download()`

流程：
- 收集 URL 列表。
- 调用 `create_video_from_url()` 为每个 URL 建立 DB 记录，并生成 **内部 video_id**。
- 调用 `schedule_videos(..., steps=['download'], use_multi_gpu=False)` 执行下载步骤。

### 2.2 Pipeline 入口
- `python -m vat pipeline -u <url> [--force]` 会走完整流水线，但 download 仍由同一 `_download()` 执行。

### 2.3 核心执行入口
- `VideoProcessor.process()`
  - `_execute_step(TaskStep.DOWNLOAD)`
    - `VideoProcessor._download()`

实现位置：@`vat/pipeline/executor.py` `VideoProcessor._download`

---

## 3. 输入与输出

### 3.1 输入（运行时依赖）
- **数据库 Video 记录**：
  - `video.source_type`：`youtube` / `local` / 其他
  - `video.source_url`：YouTube URL 或本地路径
- **配置**：
  - `proxy.http_proxy`（全局代理配置）
  - `downloader.youtube.format`

### 3.2 输出目录结构（关键概念：内部 video_id）
- `create_video_from_url()` 会用 `md5(url)[:16]` 生成 **内部 video_id**。
- `VideoProcessor.output_dir` 默认是：
  - `Path(config.storage.output_dir) / <内部video_id>`

注意：
- 这个 `<内部video_id>` **不是** YouTube 的 11 位 video id。

### 3.3 输出文件（视频文件名的易混淆点）
- YouTube 下载器默认 `outtmpl`：`%(id)s.%(ext)s`
  - 即下载到输出目录内的文件名通常是：`<YouTube视频ID>.<ext>`
- `_find_video_file()` 会优先查 `original.*`，找不到再兜底扫描输出目录内任意视频文件（排除 `final.*`）。
  - 因此即使下载得到的是 `<youtube_id>.mp4`，后续 ASR/Embed 仍可找到。

---

## 4. 下载器实现细节（YouTubeDownloader）

实现位置：@`vat/downloaders/youtube.py` `YouTubeDownloader`

### 4.1 yt-dlp 关键参数
- **format**：来自 `downloader.youtube.format`
- **proxy**：来自全局配置 `proxy.http_proxy`（空字符串表示不使用）
- **outtmpl**：`output_dir / '%(id)s.%(ext)s'`
- **日志适配**：`YtDlpLogger`
  - 会把大量 yt-dlp 的 `info` 降级为 `debug`，避免刷屏

### 4.2 返回结构
`YouTubeDownloader.download()` 返回 dict（供 `_download()` 校验/落库）：
- `video_path`: 实际下载的视频文件路径
- `title`: 标题
- `subtitles`: YouTube 字幕文件路径字典 `{lang: Path}`（如 `{'ja': Path('xxx.ja.vtt')}`）
- `metadata`: 简介、时长、上传日期、uploader、url、youtube video_id、available_subtitles、available_auto_subtitles 等

### 4.3 YouTube 字幕下载
- **默认启用**：`download_subs=True`
- **默认语言**：`['ja', 'ja-orig', 'en']`（日语原始、日语、英语）
- **字幕格式**：VTT
- **文件命名**：`{youtube_id}.{lang}.vtt`
- **存储位置**：与视频文件同目录

字幕来源优先级：
1. 手动上传字幕（`subtitles`）
2. YouTube 自动生成字幕（`automatic_captions`）

字幕信息存储到 `video.metadata`：
```python
metadata['youtube_subtitles'] = {'ja': '/path/to/video_id.ja.vtt', ...}
metadata['available_subtitles'] = ['live_chat']  # 手动上传
metadata['available_auto_subtitles'] = ['ja', 'en', ...]  # 自动生成
```

---

## 5. Download 阶段的 metadata 增强（会触发 LLM）

`VideoProcessor._download()` 在下载成功后会尝试补充 `Video.metadata`：

### 5.1 场景识别（SceneIdentifier）
- 触发条件：下载器返回了 `title`。
- 调用：@`vat/llm/scene_identifier.py` `SceneIdentifier.detect_scene()`
- 配置：`config.downloader.scene_identify`（model/api_key/base_url，留空继承全局 `llm` 配置）
- model fallback 链：`downloader.scene_identify.model` → `llm.model`
- 结果写入：
  - `metadata['scene']`
  - `metadata['scene_name']`
  - `metadata['scene_auto_detected']`

缓存特性：
- `SceneIdentifier` 内部使用 `call_llm()`，该函数带 `diskcache` memoize（默认 1 小时 TTL）。
- Download 阶段没有禁用缓存；因此场景识别可能出现“秒出结果”。

### 5.2 视频信息翻译（VideoInfoTranslator，用于上传阶段）
- 触发条件：`config.llm.is_available()` 为真 **且** 没有已有翻译结果。
- 调用：@`vat/llm/video_info_translator.py` `VideoInfoTranslator.translate()`
- 配置：`config.downloader.video_info_translate`（model/api_key/base_url，留空继承全局 `llm` 配置）
- model fallback 链：`downloader.video_info_translate.model` → `llm.model`
- 结果写入：`metadata['translated'] = translated_info.to_dict()`

**翻译复用机制**：
- Download 阶段会检查 `video.metadata.get('translated')`
- 如果已有翻译结果（可能由 Playlist sync 阶段异步翻译完成），则跳过翻译
- 避免重复调用 LLM

缓存特性：
- `VideoInfoTranslator` **不使用** `call_llm()` 的 memoize，而是直接 `client.chat.completions.create`。
- 因此它默认不走 `diskcache` 缓存（会真实发请求）。

---

## 6. 缓存与重复执行语义

### 6.1 步骤级（数据库控制）
- 非 `--force`：若 DB 记录显示 `download` 已完成，则跳过。
- `--force`：会强制重新执行 `download` 步骤。

### 6.2 下载器级（yt-dlp 行为）
- 即使步骤被强制重跑，yt-dlp 是否重新下载取决于其内部策略与现有文件状态。
- 若你确实要“重新下载”，通常需要手动清理输出目录内已有视频文件（或后续给 yt-dlp 加强制覆盖参数——当前代码未做）。

---

## 7. 常见问题与 Debug Checklist

### 7.1 “下载完成但后续找不到视频文件”
排查顺序：
- 检查输出目录：`<output_dir>/<内部video_id>/` 是否存在视频文件。
- 了解 `_find_video_file()` 的查找规则：优先 `original.*`，否则扫描所有视频后缀且排除 `final.*`。

### 7.2 “内部 video_id 和 YouTube video_id 对不上”
- 内部 `video_id`：`md5(url)[:16]`（用于 DB 主键与输出目录名）。
- YouTube `video_id`：11 位（用于下载文件名与 metadata['video_id']）。

### 7.3 代理/网络/地区限制
- 检查全局代理配置 `proxy.http_proxy`。
- 直接运行 yt-dlp 可能更容易得到完整错误信息；VAT 内部会对部分日志降级。

### 7.4 YouTube 风控注意事项

- **获取 playlist 视频列表**：基本无风控。
- **逐个获取视频 info**：并发约 10 时偶尔触发验证，但 yt-dlp 最新版可自动处理，默认并发 10 即可。
- **下载视频内容**：风控较明显，容易遇到 429/401 错误。经过测试，至少本人所用梯子，并发=2时，会在下几个视频之后迅速被限制，要求传递cookie（并非yt-dlp说的“日均1L"）。所以建议下载阶段保持并发=1，并且挂载后台。将并发处理留给后续的 ASR/翻译等阶段。
- **失败重试**：Pipeline 层面已实现"失败放队尾"重试机制（最多 2 轮），详见 `vat/pipeline/readme.md`。

### 7.5 Debug Checklist（建议按顺序）
1. **确认 DB 记录**：该 `video_id` 是否存在、`source_url/source_type` 是否正确。
2. **确认输出目录**：`storage.output_dir/<内部video_id>/` 是否存在。
3. **确认下载文件**：目录内是否有 `<youtube_id>.(mp4/webm/mkv)`。
4. **确认后续可见性**：`_find_video_file()` 会优先找 `original.*`，否则扫描目录内除 `final.*` 外的视频后缀。
5. **确认 LLM 相关副作用**（可选）：场景识别/视频信息翻译是否触发、是否因缓存导致“秒出结果”。

---

## 8. 关键代码索引

| 组件 | 文件位置 | 函数/类 |
|------|----------|---------|
| CLI 入口 | `vat/cli/commands.py` | `download()` |
| 创建 video 记录 | `vat/pipeline/executor.py` | `create_video_from_url()` |
| 执行下载 | `vat/pipeline/executor.py` | `VideoProcessor._download()` |
| YouTube 下载器 | `vat/downloaders/youtube.py` | `YouTubeDownloader` |
| 场景识别 | `vat/llm/scene_identifier.py` | `SceneIdentifier.detect_scene()` |
| 视频信息翻译 | `vat/llm/video_info_translator.py` | `VideoInfoTranslator.translate()` |
| 查找视频文件 | `vat/pipeline/executor.py` | `_find_video_file()` |

---

## 9. 修改指南：如果你想改某个功能...

| 如果你想... | 应该看/改哪里 |
|-------------|---------------|
| 添加新的下载源 | 继承 `BaseDownloader`，参考 `YouTubeDownloader` |
| 改下载文件名格式 | `YouTubeDownloader._get_ydl_opts()` 中的 `outtmpl` |
| 改场景识别逻辑 | `vat/llm/scene_identifier.py` + `vat/llm/scenes.yaml` |
| 改视频信息翻译格式 | `vat/llm/video_info_translator.py` |
| 改 video_id 生成规则 | `create_video_from_url()` 中的 `md5(url)[:16]` |
| 改 yt-dlp 参数 | `config.downloader.youtube.*` |

---

## 10. 相关配置片段（config/default.yaml）

```yaml
# 全局代理配置（用于下载、HuggingFace 模型加载、LLM API 调用）
proxy:
  http_proxy: "http://localhost:7890"

downloader:
  youtube:
    # 代理：引用全局配置 proxy.http_proxy
    format: "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"
    max_workers: 1

storage:
  output_dir: "/local/gzy/4090-48/vat/data/videos"
  database_path: "/local/gzy/4090-48/vat/data/database.db"

asr:
  split:
    model: "gpt-4o-mini"   # 注意：场景识别默认复用这个 model

translator:
  llm:
    model: "gpt-5-nano"    # 注意：视频信息翻译默认用这个 model

llm:
  api_key: "${VAT_LLM_APIKEY}"
  base_url: "https://api.videocaptioner.cn"
```
