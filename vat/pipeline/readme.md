# VAT Pipeline 架构文档

> 本文档说明 VAT 的整体 Pipeline 架构，包括阶段定义、依赖关系、执行流程、任务调度、以及各组件的职责边界。

---

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              VAT Pipeline 架构                                   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                           CLI 入口层                                      │   │
│  │   vat download | asr | translate | embed | upload | pipeline             │   │
│  │                              (commands.py)                                │   │
│  └──────────────────────────────────┬───────────────────────────────────────┘   │
│                                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                           调度层 (Scheduler)                              │   │
│  │   - 收集待处理视频                                                        │   │
│  │   - 展开阶段组为细粒度阶段                                                │   │
│  │   - 多 GPU 任务分配                                                       │   │
│  │                              (scheduler.py)                               │   │
│  └──────────────────────────────────┬───────────────────────────────────────┘   │
│                                     ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                           执行层 (VideoProcessor)                         │   │
│  │                                                                           │   │
│  │   process(steps=[...])                                                    │   │
│  │       │                                                                   │   │
│  │       ├──▶ _execute_step(DOWNLOAD)  ──▶ _download()                      │   │
│  │       │                                                                   │   │
│  │       ├──▶ _execute_step(WHISPER)   ──▶ _run_whisper()                   │   │
│  │       ├──▶ _execute_step(SPLIT)     ──▶ _run_split()                     │   │
│  │       │                                                                   │   │
│  │       ├──▶ _execute_step(OPTIMIZE)  ──▶ _run_optimize()                  │   │
│  │       ├──▶ _execute_step(TRANSLATE) ──▶ _run_translate()                 │   │
│  │       │                                                                   │   │
│  │       ├──▶ _execute_step(EMBED)     ──▶ _embed()                         │   │
│  │       │                                                                   │   │
│  │       └──▶ _execute_step(UPLOAD)    ──▶ _upload()                        │   │
│  │                                                                           │   │
│  │                              (executor.py)                                │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                           数据层                                          │   │
│  │   - Database: 视频/任务记录                                               │   │
│  │   - Config: 全局配置                                                      │   │
│  │   - CacheMetadata: 子步骤缓存元数据                                       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段定义（重构后）

### 2.1 细粒度阶段列表

| 阶段 | TaskStep 枚举 | 所属阶段组 | 职责 |
|------|---------------|-----------|------|
| **DOWNLOAD** | `TaskStep.DOWNLOAD` | download | 下载视频 + 元数据增强 |
| **WHISPER** | `TaskStep.WHISPER` | asr | Whisper 语音识别 |
| **SPLIT** | `TaskStep.SPLIT` | asr | LLM 智能断句 |
| **OPTIMIZE** | `TaskStep.OPTIMIZE` | translate | 字幕优化 |
| **TRANSLATE** | `TaskStep.TRANSLATE` | translate | LLM 翻译 |
| **EMBED** | `TaskStep.EMBED` | embed | 字幕嵌入 |
| **UPLOAD** | `TaskStep.UPLOAD` | upload | 上传到平台 |

### 2.2 阶段组定义

```python
# vat/models.py
STAGE_GROUPS = {
    "download": [TaskStep.DOWNLOAD],
    "asr": [TaskStep.WHISPER, TaskStep.SPLIT],
    "translate": [TaskStep.OPTIMIZE, TaskStep.TRANSLATE],
    "embed": [TaskStep.EMBED],
    "upload": [TaskStep.UPLOAD],
}
```

### 2.3 阶段依赖关系

```python
STAGE_DEPENDENCIES = {
    TaskStep.DOWNLOAD: [],
    TaskStep.WHISPER: [TaskStep.DOWNLOAD],
    TaskStep.SPLIT: [TaskStep.WHISPER],
    TaskStep.OPTIMIZE: [TaskStep.SPLIT],
    TaskStep.TRANSLATE: [TaskStep.OPTIMIZE],
    TaskStep.EMBED: [TaskStep.TRANSLATE],
    TaskStep.UPLOAD: [TaskStep.EMBED],
}
```

### 2.4 依赖关系图

```
DOWNLOAD
    │
    ▼
WHISPER ─────▶ SPLIT
                 │
                 ▼
            OPTIMIZE ─────▶ TRANSLATE
                               │
                               ▼
                            EMBED ─────▶ UPLOAD
```

---

## 3. 默认执行顺序

```python
DEFAULT_STAGE_SEQUENCE = [
    TaskStep.DOWNLOAD,
    TaskStep.WHISPER,
    TaskStep.SPLIT,
    TaskStep.OPTIMIZE,
    TaskStep.TRANSLATE,
    TaskStep.EMBED,
    TaskStep.UPLOAD,
]
```

完整 pipeline 执行时，按此顺序依次执行每个阶段。

---

## 4. 文件流转图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         文件产出流程                                     │
│                                                                          │
│  DOWNLOAD                                                                │
│      │                                                                   │
│      └──▶ <youtube_id>.mp4          (下载的视频)                         │
│                 │                                                        │
│  WHISPER        ▼                                                        │
│      │     <video>.wav              (提取的音频)                         │
│      │                                                                   │
│      └──▶ original_raw.srt          (原始 Whisper 转录)                  │
│                 │                                                        │
│  SPLIT          ▼                                                        │
│      │     original_split.srt       (断句后，可选)                       │
│      │                                                                   │
│      ├──▶ original.srt              (最终原文字幕)                       │
│      └──▶ original.json             (调试用 JSON)                        │
│                 │                                                        │
│  OPTIMIZE       ▼                                                        │
│      │                                                                   │
│      └──▶ optimized.srt             (优化后的字幕)                       │
│                 │                                                        │
│  TRANSLATE      ▼                                                        │
│      │                                                                   │
│      ├──▶ translated.srt            (翻译后的字幕)                       │
│      └──▶ translated.ass            (带样式的字幕)                       │
│                 │                                                        │
│  EMBED          ▼                                                        │
│      │                                                                   │
│      └──▶ final.mp4                 (嵌入字幕的最终视频)                 │
│                 │                                                        │
│  UPLOAD         ▼                                                        │
│      │                                                                   │
│      └──▶ (上传到 B 站等平台)                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 核心组件职责

### 5.1 VideoProcessor（执行器）

**位置**：`vat/pipeline/executor.py`

**职责**：
- 执行单个视频的处理流程
- 管理阶段间的依赖和跳过逻辑
- 提供统一的进度回调和错误处理

**核心方法**：

| 方法 | 职责 |
|------|------|
| `process(steps, force)` | 主入口，按顺序执行指定阶段 |
| `_execute_step(step)` | 执行单个阶段（含 DB 检查、依赖检查） |
| `_download()` | DOWNLOAD 阶段实现 |
| `_run_whisper()` | WHISPER 阶段实现 |
| `_run_split()` | SPLIT 阶段实现 |
| `_run_optimize(force)` | OPTIMIZE 阶段实现 |
| `_run_translate(force)` | TRANSLATE 阶段实现 |
| `_embed(force)` | EMBED 阶段实现 |
| `_upload()` | UPLOAD 阶段实现 |
| `_progress_with_tracker(msg, component_name=None)` | 带进度追踪的回调，可指定组件名使日志来源显示为该组件 |
| `_make_component_progress_callback(name)` | 创建组件专用进度回调，日志来源显示为组件名而非 pipeline.executor |

### 5.2 Scheduler（调度器）

**位置**：`vat/pipeline/scheduler.py`

**职责**：
- 收集待处理的视频列表
- 展开阶段组为细粒度阶段
- 多 GPU 任务分配和并发控制

**核心函数**：

```python
def schedule_videos(
    video_ids: List[str],
    steps: List[TaskStep],
    force: bool = False,
    use_multi_gpu: bool = False,
    gpu_ids: List[int] = None
) -> Dict[str, Any]:
    """
    调度多个视频的处理任务
    
    Args:
        video_ids: 待处理的视频 ID 列表
        steps: 要执行的阶段列表
        force: 是否强制重跑
        use_multi_gpu: 是否使用多 GPU 并行
        gpu_ids: 指定的 GPU ID 列表
    """
```

### 5.3 Database（数据库）

**位置**：`vat/database.py`

**职责**：
- 管理 Video 和 Task 记录
- 跟踪阶段完成状态
- 支持断点续传

**核心方法**：

| 方法 | 职责 |
|------|------|
| `add_video(video)` | 添加视频记录 |
| `get_video(video_id)` | 获取视频信息 |
| `add_task(task)` | 添加任务记录 |
| `update_task_status(...)` | 更新任务状态 |
| `is_step_completed(video_id, step)` | 检查阶段是否已完成 |
| `get_pending_steps(video_id)` | 获取待处理阶段 |

---

## 6. 阶段执行逻辑

### 6.1 `_execute_step` 流程

```python
def _execute_step(self, step: TaskStep) -> bool:
    # 1. 检查是否已完成（非 force 模式）
    if not self.force and self.db.is_step_completed(self.video_id, step):
        self.progress_callback(f"跳过已完成的阶段: {step.value}")
        return True
    
    # 2. 检查依赖是否满足
    for dep in STAGE_DEPENDENCIES.get(step, []):
        if not self.db.is_step_completed(self.video_id, dep):
            raise PipelineError(f"依赖未满足: {dep.value}")
    
    # 3. 创建/更新任务记录
    self.db.add_task(Task(
        video_id=self.video_id,
        step=step,
        status=TaskStatus.RUNNING
    ))
    
    # 4. 执行阶段
    try:
        result = self._dispatch_step(step)
        self.db.update_task_status(self.video_id, step, TaskStatus.COMPLETED)
        return result
    except Exception as e:
        self.db.update_task_status(self.video_id, step, TaskStatus.FAILED, str(e))
        raise
```

### 6.2 阶段分发

```python
def _dispatch_step(self, step: TaskStep) -> bool:
    dispatch_map = {
        TaskStep.DOWNLOAD: self._download,
        TaskStep.WHISPER: self._run_whisper,
        TaskStep.SPLIT: self._run_split,
        TaskStep.OPTIMIZE: lambda: self._run_optimize(self.force),
        TaskStep.TRANSLATE: lambda: self._run_translate(self.force),
        TaskStep.EMBED: lambda: self._embed(self.force),
        TaskStep.UPLOAD: self._upload,
    }
    return dispatch_map[step]()
```

---

## 7. 缓存机制

### 7.1 三层缓存架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        缓存层次                                  │
├─────────────────────────────────────────────────────────────────┤
│ 层次 1: 步骤级缓存 (DB tasks 表)                                 │
│         - 记录阶段完成状态                                       │
│         - force=False 时跳过已完成阶段                           │
├─────────────────────────────────────────────────────────────────┤
│ 层次 2: 文件级缓存 (.cache_metadata.json)                        │
│         - 记录子步骤的配置快照                                   │
│         - 配置变化时自动失效                                     │
│         - 用于 WHISPER/SPLIT 等阶段                             │
├─────────────────────────────────────────────────────────────────┤
│ 层次 3: LLM 调用缓存 (diskcache)                                 │
│         - call_llm() 的 memoize                                 │
│         - Chunk 级翻译缓存                                       │
│         - force 时通过 disable_cache() 禁用                     │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 force 语义

| 标志 | 层次 1 (DB) | 层次 2 (文件) | 层次 3 (LLM) |
|------|-------------|---------------|--------------|
| 不传 `-f` | 跳过已完成 | 复用缓存 | 复用缓存 |
| 传 `-f` | 强制执行 | 跳过检查 | 禁用缓存 |

---

## 8. GPU 调度

### 8.1 GPU 选择策略

```python
# vat/utils/gpu.py
def resolve_gpu_device(
    device: str = "auto",
    allow_cpu_fallback: bool = False,
    min_free_memory_mb: int = 2000
) -> Tuple[str, int]:
    """
    解析 GPU 设备
    
    Args:
        device: "auto" | "cuda:N" | "cpu"
        allow_cpu_fallback: 是否允许 CPU 回退
        min_free_memory_mb: 自动选择时的最小空闲显存
    
    Returns:
        (device_str, gpu_id)
    """
```

### 8.2 多 GPU 并行

```python
# scheduler.py
if use_multi_gpu:
    # 按 GPU 分配任务
    gpu_count = len(gpu_ids) if gpu_ids else torch.cuda.device_count()
    for i, video_id in enumerate(video_ids):
        gpu_id = gpu_ids[i % gpu_count] if gpu_ids else i % gpu_count
        # 在指定 GPU 上执行
```

### 8.3 GPU 原则

- **项目默认运行在 GPU 服务器**
- **禁止静默回退 CPU**
- 需要 GPU 的阶段：WHISPER, EMBED
- GPU 不可用时应 fail-fast

---

## 9. 错误处理

### 9.1 异常类型

```python
# vat/pipeline/exceptions.py
class PipelineError(Exception):
    """Pipeline 基础异常"""
    pass

class DownloadError(PipelineError):
    """下载阶段异常"""
    pass

class ASRError(PipelineError):
    """ASR 阶段异常"""
    def __init__(self, message, original_error=None):
        self.original_error = original_error
        ...

class TranslateError(PipelineError):
    """翻译阶段异常"""
    def __init__(self, message, original_error=None):
        self.original_error = original_error
        ...

class EmbedError(PipelineError):
    """嵌入阶段异常"""
    pass
```

### 9.2 错误传播

```
阶段方法抛出异常
    │
    ▼
_execute_step 捕获
    ├── 更新 DB 状态为 FAILED
    ├── 记录错误信息
    └── 向上传播
           │
           ▼
process() 捕获
    ├── 记录到日志
    └── 返回失败状态
```

---

## 10. CLI 命令映射

| 命令 | 执行的阶段 |
|------|-----------|
| `vat download` | DOWNLOAD |
| `vat asr` | WHISPER + SPLIT |
| `vat translate` | OPTIMIZE + TRANSLATE |
| `vat embed` | EMBED |
| `vat upload` | UPLOAD |
| `vat pipeline` | DOWNLOAD → ... → UPLOAD（完整流程） |
| `vat pipeline --steps whisper,translate` | WHISPER + TRANSLATE（自定义阶段） |

---

## 11. 关键代码索引

| 组件 | 文件位置 |
|------|----------|
| 阶段定义 | `vat/models.py` |
| 执行器 | `vat/pipeline/executor.py` |
| 调度器 | `vat/pipeline/scheduler.py` |
| 异常定义 | `vat/pipeline/exceptions.py` |
| CLI 入口 | `vat/cli/commands.py` |
| 数据库 | `vat/database.py` |
| 配置 | `vat/config.py` |
| 缓存元数据 | `vat/utils/cache_metadata.py` |
| GPU 工具 | `vat/utils/gpu.py` |

---

## 12. 修改指南

| 如果你想... | 应该看/改哪里 |
|-------------|---------------|
| 添加新阶段 | `vat/models.py` TaskStep + STAGE_GROUPS + STAGE_DEPENDENCIES |
| 改阶段执行逻辑 | `vat/pipeline/executor.py` 对应的 `_run_*` 方法 |
| 改调度策略 | `vat/pipeline/scheduler.py` |
| 改缓存逻辑 | `_should_use_cache()` + `CacheMetadata` |
| 改依赖检查 | `_execute_step()` 中的依赖检查逻辑 |
| 改 CLI 参数 | `vat/cli/commands.py` |
| 添加新数据库字段 | `vat/database.py` + `vat/models.py` |

---

## 13. 已修复的历史问题

### 13.1 下载失败重试机制

CLI `process` 命令采用"失败放队尾"策略：首轮处理全部视频后，收集失败的视频在队尾重试（最多 2 轮）。利用 `get_pending_steps` 返回 FAILED 步骤的特性，重新调用 `process()` 会自然从失败步骤重试。相比内联重试（阻塞当前线程等待），该策略让其他视频先继续处理，减少整体等待时间。实现位置：`vat/cli/commands.py` `_run_batch()`。

### 13.2 ASR 空输出处理（no_speech）

PV/纯音乐等无人声视频，Whisper 输出为空时不再报错，而是在 `video.metadata` 中标记 `no_speech=True`，后续阶段（split/optimize/translate/embed）检测到该标记后自动跳过。embed 阶段会直接复制原视频作为 `final.mp4`。实现位置：`vat/pipeline/executor.py` `_is_no_speech()` 及各阶段方法开头的检查。
