# Phase E Technical Subdomains Implementation Plan

> **For agentic workers:** Execute this plan using the current harness capabilities. Use checkboxes (`- [ ]`) for tracking. If subagents are explicitly requested or clearly beneficial, delegate bounded subtasks; otherwise execute directly.

**Goal:** 完成第一轮技术子域收口，统一 LLM 调用 facade、统一媒体基础操作入口，并把字幕编解码从 `ASRData` 中抽到独立模块，由 `ASRData` 委托使用。

**Architecture:** 本阶段不追求一次性拆完所有技术大文件，而是优先收口最核心的三个技术边界：1) 所有通用 LLM 文本调用统一走一套 facade；2) ffprobe / extract_audio 这类媒体基础操作不再多处重复实现；3) 字幕文件编解码和文件 I/O 从 `vat/asr/asr_data.py` 中抽离，形成更稳定的字幕子域入口。目标是“先收边界，再为后续更深拆分打地基”。 

**Tech Stack:** Python, pytest, OpenAI-compatible API client, ffmpeg/ffprobe

---

## 范围与原则

- 本阶段优先处理：
  - LLM facade
  - 媒体基础操作
  - 字幕编解码委托层
- 本阶段暂不处理：
  - `LLMTranslator` 的更细策略拆分
  - `FFmpegWrapper` 的全面拆分
  - 字幕样式/排版/渲染的全部迁移
- 保守原则：
  - 先做“边界稳定化”
  - 不为技术整洁感一次性移动太多代码
  - 旧调用点允许通过薄 wrapper/委托过渡

## 当前拟定方案

1. 新增 `vat/llm/facade.py`
   - 提供统一的文本调用入口
   - `SceneIdentifier` 与 `VideoInfoTranslator` 改为走 facade
2. 新增 `vat/media/`
   - `probe.py`：统一 ffprobe 解析
   - `audio.py`：统一 ffmpeg 音频提取
   - `BaseDownloader` / `FFmpegWrapper` / `WhisperASR` / `VideoProcessor` 复用
3. 新增 `vat/subtitle_utils/codecs.py`
   - 收字幕文件编解码与文件保存
   - `ASRData` 保留领域对象，但将 `save/from_*` 等 codec 职责委托出去

## 仍需二次核查的点

- `VideoInfoTranslator` 是否最终完全不再需要 `_get_client()`，本阶段更倾向于保留兼容 helper，但不再作为主调用路径
- 字幕子域最终是否扩成更明确的 `vat/subtitles/`，本阶段先不定目录结构
- `FFmpegWrapper` 与 `ASRData.to_ass()` 的渲染边界，本阶段只做编解码和媒体基础操作，不做全面样式迁移

## 目标文件结构

**主要文件：**
- Create: `vat/llm/facade.py`
- Create: `vat/media/__init__.py`
- Create: `vat/media/probe.py`
- Create: `vat/media/audio.py`
- Create: `vat/subtitle_utils/codecs.py`
- Modify: `vat/llm/scene_identifier.py`
- Modify: `vat/llm/video_info_translator.py`
- Modify: `vat/downloaders/base.py`
- Modify: `vat/embedder/ffmpeg_wrapper.py`
- Modify: `vat/asr/whisper_wrapper.py`
- Modify: `vat/pipeline/executor.py`
- Modify: `vat/asr/asr_data.py`
- Test: `tests/test_scene_identifier.py`
- Test: `tests/test_video_info_translator.py`
- Test: `tests/test_embedder_runtime.py`
- Test: `tests/test_downloader_helpers.py`
- Test: `tests/test_media_helpers.py`

---

## Chunk 1: 统一 LLM facade

### Task 1: 为 LLM facade 写失败测试

**Files:**
- Modify: `tests/test_scene_identifier.py`
- Modify: `tests/test_video_info_translator.py`

- [x] **Step 1: 将 `SceneIdentifier` 的测试改为围绕 facade 调用**
- [x] **Step 2: 将 `VideoInfoTranslator` 的测试改为围绕 facade 调用**
- [x] **Step 3: 确认 `translate()` 和 `detect_scene()` 不再依赖各自独立的 client 语义**

### Task 2: 实现 `vat/llm/facade.py`

**Files:**
- Create: `vat/llm/facade.py`
- Modify: `vat/llm/scene_identifier.py`
- Modify: `vat/llm/video_info_translator.py`

- [x] **Step 1: 新增统一文本调用入口**
- [x] **Step 2: `SceneIdentifier` 改用 facade**
- [x] **Step 3: `VideoInfoTranslator` 改用 facade**
- [x] **Step 4: 保留最小兼容 helper，不强行删除所有旧私有方法**

---

## Chunk 2: 收口媒体基础操作

### Task 3: 为媒体基础操作写失败测试

**Files:**
- Modify: `tests/test_embedder_runtime.py`
- Modify: `tests/test_downloader_helpers.py`
- Create: `tests/test_media_helpers.py`

- [x] **Step 1: 为统一 ffprobe helper 补直接测试**
- [x] **Step 2: 为统一音频提取 helper 补直接测试**
- [x] **Step 3: 保证原有 `BaseDownloader` / `FFmpegWrapper` 契约测试仍成立**

### Task 4: 实现 `vat/media/`

**Files:**
- Create: `vat/media/__init__.py`
- Create: `vat/media/probe.py`
- Create: `vat/media/audio.py`
- Modify: `vat/downloaders/base.py`
- Modify: `vat/embedder/ffmpeg_wrapper.py`
- Modify: `vat/asr/whisper_wrapper.py`
- Modify: `vat/pipeline/executor.py`

- [x] **Step 1: 新增统一 ffprobe 解析 helper**
- [x] **Step 2: 新增统一 extract_audio helper**
- [x] **Step 3: `BaseDownloader.probe_video_metadata()` 改为委托**
- [x] **Step 4: `FFmpegWrapper.get_video_info()/extract_audio()` 改为委托**
- [x] **Step 5: `WhisperASR._extract_audio()` / `VideoProcessor._extract_audio()` 改为委托**

---

## Chunk 3: 抽离字幕编解码

### Task 5: 为 codec 委托写失败测试

**Files:**
- Create: `tests/test_subtitle_codecs.py`

- [x] **Step 1: 为 SRT / ASS / JSON 编解码补最小契约测试**
- [x] **Step 2: 为 `ASRData.save()` / `from_subtitle_file()` 仍保持兼容补测试**

### Task 6: 新增 `vat/subtitle_utils/codecs.py`

**Files:**
- Create: `vat/subtitle_utils/codecs.py`
- Modify: `vat/asr/asr_data.py`

- [x] **Step 1: 将 `save()/to_srt()/to_txt()/to_json()/from_*` 的 codec 职责迁到新模块**
- [x] **Step 2: `ASRData` 保留领域对象，对外 API 不变，只做委托**
- [x] **Step 3: 不在本阶段迁移 `to_ass()` 的渲染样式细节**

---

## 验证矩阵

至少运行：

```bash
pytest tests/test_scene_identifier.py tests/test_video_info_translator.py -q
pytest tests/test_embedder_runtime.py tests/test_downloader_helpers.py tests/test_media_helpers.py -q
pytest tests/test_subtitle_codecs.py -q
```

收尾前建议补跑：

```bash
pytest tests/test_database.py tests/test_pipeline.py tests/test_cli_process.py tests/test_services.py tests/test_watch_service.py tests/test_web_jobs.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_watch_api.py tests/test_bilibili_web_api.py tests/test_bilibili_batch_resync.py tests/test_tools_job.py tests/test_videos_api.py tests/test_scene_identifier.py tests/test_video_info_translator.py tests/test_embedder_runtime.py tests/test_downloader_helpers.py tests/test_media_helpers.py tests/test_subtitle_codecs.py -q
```

---

## 当前不做

- 不在 Phase E 里拆 `LLMTranslator`
- 不在 Phase E 里重做 `FFmpegWrapper` 的全部结构
- 不在 Phase E 里全面迁移字幕样式/排版

---

## 当前结论

`Phase E` 已完成第一轮技术子域收口，当前已落地的点：

- 新增 `vat/llm/facade.py`，并让 `SceneIdentifier` / `VideoInfoTranslator` 走统一文本调用入口
- 新增 `vat/media/`，统一 ffprobe 与音频提取基础操作
- `BaseDownloader`、`FFmpegWrapper`、`WhisperASR`、`VideoProcessor` 已复用媒体基础 helper
- 新增 `vat/subtitle_utils/codecs.py`
- `ASRData` 的文件编解码与文件保存职责已迁到 codec 模块，并通过委托保持对外 API 不变

这意味着：

- LLM 基础设施的主入口已经明显收口
- 媒体基础操作不再多处重复实现
- `ASRData` 开始从“领域对象 + codec + 文件 I/O + 渲染入口”的混合体中抽离出更明确的 codec 边界

本阶段仍明确留给后续的内容：

- `LLMTranslator` 的更细策略拆分
- `FFmpegWrapper` 的进一步有限拆分
- 字幕样式/排版/ASS 渲染的更深迁移

## 验证结果

已运行：

```bash
pytest tests/test_scene_identifier.py tests/test_video_info_translator.py tests/test_embedder_runtime.py tests/test_downloader_helpers.py tests/test_media_helpers.py tests/test_subtitle_codecs.py -q
pytest tests/test_database.py tests/test_pipeline.py tests/test_cli_process.py tests/test_services.py tests/test_watch_service.py tests/test_web_jobs.py tests/test_tasks_api.py tests/test_playlists_api.py tests/test_watch_api.py tests/test_bilibili_web_api.py tests/test_bilibili_batch_resync.py tests/test_tools_job.py tests/test_videos_api.py tests/test_scene_identifier.py tests/test_video_info_translator.py tests/test_embedder_runtime.py tests/test_downloader_helpers.py tests/test_media_helpers.py tests/test_subtitle_codecs.py -q
```

结果：

- `68 passed in 17.31s`
- `402 passed in 19.96s`

Plan saved to `docs/superpowers/plans/2026-03-24-phase-e-technical-subdomains-implementation-plan.md` and executed.
