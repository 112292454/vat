# VAT 项目结构审查报告

## 概述

本文档对 VAT（Video Auto Translator）项目进行全面审查，涵盖三个层面：
1. **功能性 Bug**（已修复）
2. **Pipeline 与 WebUI 集成问题**（已修复 / 已确认）
3. **设计与架构问题**（建议方案）

审查范围：pipeline 核心流程（executor, scheduler, commands）、WebUI 层（routes, jobs, app）、各功能模块间的衔接。

---

## 一、已修复的功能性 Bug

### Bug-1: CLI 旧命令引用不存在的 `TaskStep.ASR`

**文件**: `vat/cli/commands.py`

**问题**: 细粒度阶段重构后，`TaskStep` 枚举中 `ASR` 已被拆分为 `WHISPER` + `SPLIT`，但以下旧命令仍引用了不存在的 `TaskStep.ASR`：
- `asr` 命令 (line ~151): `db.is_step_completed(vid, TaskStep.ASR)` → `AttributeError`
- `translate` 命令 (line ~192): `db.is_step_completed(vid, TaskStep.ASR)` → `AttributeError`

同时，传递给 `schedule_videos` 的 step 名称也已过时：
- `steps=['asr']` — `'asr'` 不是合法的 `TaskStep` 值，会在 `process()` 中 `TaskStep('asr')` 抛出 `ValueError`
- `steps=['download', 'asr', 'translate', 'embed']` — 同上

**修复**:
- `TaskStep.ASR` → `TaskStep.SPLIT`（检查 ASR 组是否完成时以最后一步 SPLIT 为准）
- 在 `VideoProcessor.process()` 中增加阶段组名展开逻辑：接收到 `'asr'` 时自动展开为 `['whisper', 'split']`，`'translate'` 展开为 `['optimize', 'translate']`，基于 `models.py` 中已有的 `STAGE_GROUPS` 和 `expand_stage_group()` 函数
- CLI 命令保持使用语义化组名：`steps=['asr']`、`steps=['translate']`、`steps=['download', 'asr', 'translate', 'embed']`

**影响**: 所有旧的快捷 CLI 命令（`vat asr`, `vat translate`, `vat pipeline`）在修复前完全不可用。

---

### Bug-2: `db.update_video_metadata()` 方法不存在

**文件**: `vat/cli/commands.py` (upload 命令，line ~1063)

**问题**: `upload` 命令在上传成功后调用 `db.update_video_metadata(video_id, {...})`，但 `Database` 类没有此方法。运行时会抛出 `AttributeError`。

**修复**: 改为先获取现有 metadata，合并新字段后调用 `db.update_video(video_id, metadata=updated_metadata)`。

**影响**: B站上传成功后无法将 bvid 等信息写回数据库。

---

### Bug-3: `_set_passthrough_config` 直接修改共享 Config 对象

**文件**: `vat/pipeline/executor.py`

**问题**: 当用户指定不连续的阶段（如 `whisper,embed`）时，`_set_passthrough_config` 会直接修改 `self.config` 上的字段：
```python
self.config.asr.split.enable = False
self.config.translator.llm.optimize.enable = False
self.config.translator.skip_translate = True
```
由于 `Config` 对象在 Scheduler 层是所有 `VideoProcessor` 共享的，第一个视频的 passthrough 修改会"泄漏"到后续视频，导致后续视频的 split/optimize/translate 被错误地禁用。

**修复**:
1. 在修改前保存原始值到 `_config_backup`
2. 新增 `_restore_passthrough_config()` 方法恢复原值
3. `process()` 主循环用 `try/finally` 包裹，确保无论正常退出还是异常退出都会恢复 config

**影响**: 多视频批量处理时，如果第一个视频使用了不连续阶段，后续所有视频的 split/optimize/translate 配置会被污染。

---

### Bug-4: `scene_prompt` 获取后未传递给 Translator

**文件**: `vat/pipeline/executor.py` (`_run_optimize`, `_run_translate`)

**问题**: 两个新的细粒度方法都获取了 `scene_prompt`（通过 `SceneIdentifier.get_scene_prompts()`），但获取后赋给了一个局部变量，从未合并到传给 `LLMTranslator` 构造函数的 `custom_optimize_prompt` / `custom_translate_prompt` 参数中。

对比旧的 `_translate()` 方法（line ~1484-1491）正确地做了合并：
```python
optimize_prompt = f"{scene_prompts['optimize']}\n\n{optimize_prompt}" if ...
```

**修复**: 在两个方法中，将 scene_prompt 获取移到 translator 创建之前，合并到对应的 prompt 参数中再传入构造函数。

**影响**: 场景特定的提示词（如 gaming、chatting 等场景的优化/翻译提示）完全失效，LLM 只使用用户自定义 prompt 而无场景增强。

---

## 二、Pipeline 与 WebUI 集成问题

### 集成-1: `task_service.py` 是废弃死代码 ✅ 已清理

**文件**: `vat/web/services/task_service.py`（已删除）

**问题**: `TaskService` 类使用完全错误的 `VideoProcessor` 构造参数，且完全未被使用。

**修复**: 已删除 `task_service.py`，清理 `web/services/__init__.py` 中的导出。

---

### 集成-2: WebUI 进度计算方式不一致

**问题**: 不同位置使用不同方式计算视频处理进度：

| 位置 | 计算方式 |
|------|----------|
| `app.py` 首页 | `db.get_pending_steps()` → `(7 - pending) / 7 * 100` |
| `routes/videos.py` API | `sum(t.status == COMPLETED for t in tasks) / 7` |
| `app.py` task_new 页 | `completed = 7 - len(pending)` → `completed/7*100%` |

`get_pending_steps` 返回的是"未完成的步骤"（排除 COMPLETED 和 SKIPPED），而 `routes/videos.py` 直接统计 COMPLETED 的 task 数量（不含 SKIPPED）。当某些阶段被标记为 SKIPPED 时，两种计算结果不同。

**建议**: 统一为一个 helper 函数，基于 `get_pending_steps` 计算（因为它已处理了 COMPLETED/SKIPPED 的逻辑）。

---

### 集成-3: 每个 Web 路由模块独立创建 Database 实例

**问题**: `routes/videos.py`、`routes/playlists.py`、`routes/files.py`、`app.py` 各自有 `get_db()` 函数，每次请求都创建新的 `Database` 实例（含新的 SQLite 连接）。`app.py` 的全局 `_db` 与路由模块的 `get_db()` 也不共享。

**建议**: 使用 FastAPI 的依赖注入统一提供单例 Database，或在 app 启动时创建一次并通过 `app.state` 共享。

---

## 三、设计与架构问题

### 设计-1: `executor.py` God Class（2000+ 行）

**严重程度**: 高

**问题**: `VideoProcessor` 承担了所有阶段的执行逻辑，包括 download、whisper、split、optimize、translate、embed、upload，加上辅助方法（缓存、进度、音频提取、时间戳重对齐等），总计 2000+ 行。

**具体职责**:
- 视频下载（含 YouTube 字幕、场景识别、视频信息翻译）
- ASR 转录（含人声分离、后处理、幻觉检测）
- 智能断句（含分块、说话人感知、时间戳重对齐）
- 字幕优化（LLM）
- 翻译（LLM，含说话人分组）
- 字幕嵌入（软/硬字幕，ASS 生成）
- B站上传（含封面、合集、模板渲染）
- 缓存判断
- 进度追踪

**建议方案**:
```
vat/pipeline/
├── executor.py          # VideoProcessor: 流程编排 + 状态管理（~300行）
├── stages/
│   ├── __init__.py
│   ├── base.py          # StageHandler 基类
│   ├── download.py      # DownloadHandler
│   ├── whisper.py       # WhisperHandler
│   ├── split.py         # SplitHandler
│   ├── optimize.py      # OptimizeHandler
│   ├── translate.py     # TranslateHandler
│   ├── embed.py         # EmbedHandler
│   └── upload.py        # UploadHandler
├── scheduler.py
├── progress.py
└── exceptions.py
```

每个 Handler 继承 `StageHandler`，实现 `execute(context) -> bool`，其中 `context` 包含 config、video、output_dir、db、progress_callback 等。VideoProcessor 只负责编排调用顺序和状态管理。

---

### 设计-2: 大量死代码（旧方法未清理） ✅ 已清理

**已删除的死代码**:
- `executor.py` 中旧的 `_run_asr()` 方法（约 135 行）：已被 `_run_whisper` + `_run_split` 替代
- `executor.py` 中旧的 `_translate()` 方法（约 130 行）：已被 `_run_optimize` + `_run_translate` 替代
- `web/services/task_service.py`（221 行）：废弃的 TaskService 类

**注意**: 删除旧 `_translate()` 前，已将其中的说话人感知翻译逻辑移植到新的 `_run_translate()` 中。

---

### 设计-3: Scheduler 全部使用 `print()` 而非 Logger ✅ 已修复

**文件**: `vat/pipeline/scheduler.py`

**修复**: 所有 `print()` 已替换为 `logger = setup_logger("pipeline.scheduler")`，使用 `logger.info()` / `logger.warning()` / `logger.error()`。子进程中使用 `worker_logger` 避免跨进程日志冲突。同时将 `executor.py` 中所有 `traceback.print_exc()` 替换为 `self.logger.debug(traceback.format_exc())`。

---

### 设计-4: `force` 参数双传

**严重程度**: 低

**文件**: `vat/pipeline/executor.py`, `vat/pipeline/scheduler.py`

**问题**: `force` 同时通过构造函数和 `process()` 方法参数传入：
```python
# scheduler.py
processor = VideoProcessor(video_id=..., config=..., force=force)
success = processor.process(steps, force=force)

# executor.py process()
self.force = force or self.force  # 合并两处的 force
```

两个入口点传同一参数，语义不清，容易导致"构造函数传了 True，process 传了 False，结果仍然是 True"的困惑。

**建议**: 只在 `process()` 接受 `force` 参数（运行时决定），构造函数不接受 `force`。或者反过来，只在构造函数设置，`process()` 不再接受。

---

### 设计-5: `create_video_from_url` 重复添加任务记录

**严重程度**: 中

**文件**: `vat/pipeline/executor.py` (line ~1986-2021)

**问题**: 每次调用 `create_video_from_url()` 都会为该视频创建 7 个新的 PENDING 任务记录（`db.add_task()`）。但 `db.add_video()` 使用 `INSERT OR REPLACE`，如果同一 URL 被多次添加，视频记录会被替换，但旧的任务记录不会被删除，导致数据库中出现重复的任务记录。

**建议**: 在 `add_video` 时先检查是否已存在；或在创建任务前清理该 video_id 的旧任务记录。

---

### 设计-6: 异常处理策略不一致

**严重程度**: 中

**问题**: 各阶段方法的错误处理方式不统一：

| 方法 | 失败时行为 |
|------|-----------|
| `_run_whisper` | raise `ASRError` |
| `_run_split` | raise `ASRError` |
| `_run_optimize` | raise `TranslateError` |
| `_run_translate` | raise `TranslateError` |
| `_embed` | 部分情况 return `False`，部分 raise `EmbedError` |
| `_upload` | return `False`，不 raise |
| `_download` | 部分 raise `DownloadError`，部分 return `False` |

`process()` 循环既处理异常（`except PipelineError`）又检查返回值（`if not success`），两种模式并存。

**建议**: 统一为"失败一律 raise 对应的 PipelineError 子类"，`process()` 只用 `try/except` 处理。移除 return False 路径。

---

### 设计-7: LLMTranslator 重复创建

**严重程度**: 低

**问题**: `_run_optimize`、`_run_translate`、旧 `_translate` 方法中，创建 `LLMTranslator` 的代码几乎完全相同（~15 行参数构造）。

**建议**: 提取 `_create_translator(optimize_prompt, translate_prompt, enable_optimize, is_reflect)` 工厂方法，集中管理 translator 创建逻辑。

---

### 设计-8: 命名与约定

**严重程度**: 低

以下命名可以改善：

| 当前 | 建议 | 原因 |
|------|------|------|
| `_run_whisper`, `_run_split`, `_run_optimize`, `_run_translate` | 统一前缀，如 `_stage_whisper` | 与 `_download`, `_embed`, `_upload` 不一致（后者没有 `_run_` 前缀） |
| `batch_num` (LLMTranslator) | `batch_size` | 与 config 中的 `batch_size` 字段名对齐 |
| `_fill_passthrough_stages` | `_resolve_stage_gaps` | 更准确描述"填充阶段间隙"的语义 |
| `_set_passthrough_config` / `_restore_passthrough_config` | 考虑用 context manager | `with self._passthrough_config(steps):` 更安全 |

---

### 设计-9: WebUI 路由中重复的 Database/Config 获取模式

**严重程度**: 低

**问题**: 几乎每个路由文件都有自己的 `get_db()` 函数，重复如下模式：
```python
def get_db():
    from vat.config import load_config
    config = load_config()
    return Database(config.storage.database_path)
```

`routes/files.py` 甚至在每个端点函数内部都手动创建 Database 实例（未使用依赖注入）。

**建议**: 在 `app.py` 中定义统一的 `get_db` 依赖，所有路由通过 `Depends(get_db)` 获取。`files.py` 也改为依赖注入模式。

---

### 设计-10: `_execute_step` handler 映射可扩展性

**严重程度**: 低

**当前**: `_execute_step` 使用硬编码字典映射阶段到处理方法：
```python
handlers = {
    TaskStep.DOWNLOAD: self._download,
    TaskStep.WHISPER: self._run_whisper,
    ...
}
```

**建议**: 如果实施了设计-1的阶段拆分，可改为注册式：
```python
# 在 stages/__init__.py 中注册
STAGE_HANDLERS = {
    TaskStep.DOWNLOAD: DownloadHandler,
    TaskStep.WHISPER: WhisperHandler,
    ...
}
```

---

## 四、总结

### 已修复（4 个功能性 Bug）

| # | Bug | 严重程度 | 影响范围 |
|---|-----|---------|---------|
| 1 | CLI 旧命令引用 `TaskStep.ASR` | **致命** | 所有旧 CLI 命令不可用 |
| 2 | `db.update_video_metadata` 不存在 | **高** | 上传后元数据无法保存 |
| 3 | Passthrough 污染共享 Config | **高** | 多视频批处理配置错乱 |
| 4 | Scene prompt 未传递给 Translator | **中** | 场景提示词失效 |

### 已修复的集成问题

| # | 问题 | 状态 |
|---|------|------|
| 1 | `task_service.py` 死代码 | ✅ 已删除 |
| 2 | WebUI 任务进度只显示单视频进度 | ✅ 已增加批次总进度 `[TOTAL:N%]` |

### 待处理的集成问题

| # | 问题 | 建议 |
|---|------|------|
| 1 | 进度计算不一致（app.py vs routes/videos.py） | 统一为 helper 函数 |
| 2 | Database 实例重复创建 | 统一依赖注入 |

### 测试套件状态

**62 passed, 0 failed**（修复前：28 passed, 4 failed）

修复的测试（4 个）：
- 3 个 Playlist 测试：改为使用 `add_video_to_playlist()` 写入关联表，匹配 `list_videos(playlist_id=...)` 的 JOIN 查询
- 1 个 Config 测试：移除过时的 `proxy` 字段（已迁移为顶层 `ProxyConfig`），更新 `from_dict` 字典结构

新增的测试类（34 个测试）：
| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|----------|
| `TestStageGroupExpansion` | 5 | 阶段组展开逻辑（asr→whisper+split 等） |
| `TestBatchProgressCalculation` | 5 | 批次总进度公式验证 |
| `TestProgressLogParsing` | 4 | WebUI 日志解析 `[TOTAL:N%]` 格式 |
| `TestSpeakerGrouping` | 2 | 说话人分组逻辑 |
| `TestCreateVideoFromUrl` | 4 | 重复创建视频时清理旧任务（设计-5） |
| `TestDeadCodeRemoval` | 5 | 死代码清理回归验证 |
| `TestSchedulerLogging` | 5 | 日志统一 + force 单入口 + 工厂方法验证（设计-4/7） |
| `TestProgressTracker` | 3 | ProgressTracker 阶段进度推进 |
| `TestPipelineExceptions` | 2 | UploadError + 统一异常类型验证（设计-6） |

### 架构改进建议（按优先级）

| 优先级 | 问题 | 工作量 | 状态 |
|--------|------|--------|------|
| 高 | executor.py God Class 拆分 | 大（~2天） | 待实施 |
| ~~高~~ | ~~清理死代码~~ | ~~小~~ | ✅ 已完成 |
| ~~中~~ | ~~统一异常处理策略~~ | ~~中~~ | ✅ 已完成（设计-6：return False→raise，新增 UploadError） |
| ~~中~~ | ~~修复 create_video_from_url 重复任务~~ | ~~小~~ | ✅ 已完成（设计-5：清理旧任务记录） |
| ~~低~~ | ~~scheduler.py print → logger~~ | ~~小~~ | ✅ 已完成 |
| ~~低~~ | ~~force 参数单一入口~~ | ~~小~~ | ✅ 已完成（设计-4：仅构造函数接受 force） |
| ~~低~~ | ~~LLMTranslator 工厂方法~~ | ~~小~~ | ✅ 已完成（设计-7：_create_translator + _get_scene_prompt） |
| ~~低~~ | ~~命名规范化~~ | ~~小~~ | ✅ 已完成（设计-8：_run_ 前缀统一 + _resolve_stage_gaps） |
| ~~低~~ | ~~WebUI 依赖注入统一~~ | ~~小~~ | ✅ 已完成（设计-9：deps.py 共享单例 + N+1 查询消除） |
