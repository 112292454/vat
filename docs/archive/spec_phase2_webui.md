# VAT Web UI 设计规划文档

> **文档版本**: v1.0  
> **创建日期**: 2026-02-02  
> **状态**: ✅ 已实施完成（归档）  
> **前置依赖**: [spec_phase1_pipeline.md](./spec_phase1_pipeline.md)

---

## 目录

1. [概述](#一概述)
2. [Web 架构设计](#二web-架构设计)
3. [任务执行机制](#三任务执行机制)
4. [实时输出与日志](#四实时输出与日志)
5. [页面功能设计](#五页面功能设计)
6. [文件管理功能](#六文件管理功能)
7. [CLI 等价性保证](#七cli-等价性保证)
8. [实施计划](#八实施计划)

---

## 一、概述

### 1.1 本阶段目标

Web UI 阶段的目标是提供一个**功能完备、数据准确**的 Web 管理界面：
1. 支持视频/Playlist 的管理和处理
2. 任务异步执行，支持实时查看进度和日志
3. 与 CLI 完全等价的执行效果
4. 文件查看（及可能的编辑）功能

### 1.2 核心原则

1. **功能为先**：确保数据正确、功能可用，外观其次
2. **CLI 等价**：Web 操作与命令行执行产生完全相同的结果
3. **异步非阻塞**：长时间任务后台执行，不阻塞用户界面
4. **实时反馈**：处理进度和日志实时推送到前端

---

## 二、Web 架构设计

### 2.1 技术栈（已实现）

```
后端: FastAPI (异步框架)
前端: Jinja2 模板 + TailwindCSS + Alpine.js
通信: REST API + SSE (日志流)
任务执行: 子进程 + CLI (与 Web 服务器生命周期解耦)
```

### 2.2 目录结构（实际）

```
vat/web/
├── __init__.py
├── app.py              # FastAPI 应用入口 + 页面路由
├── jobs.py             # 任务持久化（子进程执行 CLI）
├── routes/
│   ├── __init__.py
│   ├── videos.py       # 视频管理 API
│   ├── playlists.py    # Playlist 管理 + 同步 API
│   ├── tasks.py        # 任务执行 API
│   ├── files.py        # 文件浏览 API
│   └── prompts.py      # Custom Prompt 管理 API
├── services/           # 业务服务（当前为空，逻辑在 vat/services/）
└── templates/          # Jinja2 HTML 模板
    ├── base.html           # 基础布局
    ├── index.html          # 视频列表
    ├── video_detail.html   # 视频详情
    ├── playlists.html      # Playlist 列表
    ├── playlist_detail.html # Playlist 详情
    ├── prompts.html        # Custom Prompt 管理
    ├── tasks.html          # 任务列表
    ├── task_new.html       # 新建任务
    └── task_detail.html    # 任务详情
```

### 2.3 API 设计概览

#### 2.3.1 视频管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/videos` | 列出所有视频（支持分页、过滤） |
| GET | `/api/videos/{id}` | 获取视频详情 |
| POST | `/api/videos` | 添加视频（URL/本地路径） |
| DELETE | `/api/videos/{id}` | 删除视频记录 |
| GET | `/api/videos/{id}/files` | 获取视频相关文件列表 |

#### 2.3.2 Playlist 管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/playlists` | 列出所有 Playlist |
| GET | `/api/playlists/{id}` | 获取 Playlist 详情及视频列表 |
| POST | `/api/playlists` | 添加 Playlist（URL） |
| POST | `/api/playlists/{id}/sync` | 同步 Playlist（增量更新，后台执行） |
| DELETE | `/api/playlists/{id}` | 删除 Playlist |
| GET | `/api/playlists/{id}/prompt` | 获取 Playlist 的 Custom Prompt 配置 |
| PUT | `/api/playlists/{id}/prompt` | 设置 Playlist 的 Custom Prompt 配置 |

#### 2.3.2.1 Custom Prompt 管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/prompts` | 列出所有 Custom Prompts |
| GET | `/api/prompts/{type}/{name}` | 获取指定 Prompt 内容 |
| POST | `/api/prompts` | 创建新 Prompt |
| PUT | `/api/prompts/{type}/{name}` | 更新 Prompt |
| DELETE | `/api/prompts/{type}/{name}` | 删除 Prompt |

**Prompt 类型**：`translate`（翻译）、`optimize`（优化）

#### 2.3.3 任务执行

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/tasks/execute` | 执行处理任务 |
| GET | `/api/tasks` | 列出任务历史 |
| GET | `/api/tasks/{id}` | 获取任务详情 |
| POST | `/api/tasks/{id}/cancel` | 取消任务 |
| DELETE | `/api/tasks/{id}` | 删除任务记录（仅已完成/失败/取消的任务） |
| POST | `/api/tasks/{id}/retry` | 重新运行任务（基于原参数创建新任务） |
| GET | `/api/tasks/{id}/logs` | SSE 实时日志流 |

#### 2.3.4 文件管理

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/files/list` | 列出指定目录下文件 |
| GET | `/api/files/view` | 查看文件内容 |
| PUT | `/api/files/save` | 保存文件（可选功能） |

---

## 三、任务执行机制

### 3.1 子进程执行架构（已实现）

**设计决定**：任务通过子进程执行 CLI 命令，与 Web 服务器生命周期完全解耦。

```
┌────────────────────────────────────────────────────────────┐
│                      FastAPI 主进程                         │
│  ┌─────────────┐    ┌─────────────┐                        │
│  │  HTTP 请求  │ -> │  JobManager │                        │
│  │   /execute  │    │  (任务管理)  │                        │
│  └─────────────┘    └──────┬──────┘                        │
│         │                  │                                │
│         v                  v                                │
│  ┌─────────────┐    ┌─────────────────────────────────┐   │
│  │  立即返回   │    │  subprocess.Popen(              │   │
│  │  task_id   │    │    "python -m vat process ..."  │   │
│  └─────────────┘    │  )                              │   │
│                     └──────┬──────────────────────────┘   │
└────────────────────────────│──────────────────────────────┘
                             │
                             v
┌────────────────────────────────────────────────────────────┐
│                      独立子进程                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  vat process -v xxx -s download,whisper,...        │   │
│  │                                                     │   │
│  │  stdout/stderr -> job_logs/job_xxx.log             │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                             │
                             v
┌────────────────────────────────────────────────────────────┐
│  日志查看: SSE 实时读取 log 文件 -> 前端显示                  │
└────────────────────────────────────────────────────────────┘
```

**优点**：
- 任务与 Web 服务器生命周期解耦，Web 重启不影响任务
- 复用 CLI 逻辑，确保 CLI 等价性
- 任务状态持久化在 SQLite（`web_jobs` 表）

### 3.2 JobManager 实现（实际）

```python
# vat/web/jobs.py

@dataclass
class WebJob:
    """Web 任务记录"""
    job_id: str
    video_ids: List[str]
    steps: List[str]
    gpu_device: str
    force: bool
    status: JobStatus  # pending/running/completed/failed/cancelled
    pid: Optional[int]
    log_file: Optional[str]
    progress: float
    error: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class JobManager:
    """任务管理器 - 通过子进程执行 CLI"""
    
    def submit_job(self, video_ids, steps, gpu_device, force) -> str:
        # 1. 写入数据库 (web_jobs 表)
        # 2. 启动子进程: python -m vat process -v xxx -s yyy
        # 3. 返回 job_id
    
    def get_job(self, job_id) -> WebJob:
        # 从数据库读取任务信息
    
    def update_job_status(self, job_id):
        # 检查子进程是否还在运行，更新状态
    
    def cancel_job(self, job_id):
        # 发送 SIGTERM 终止子进程
```

### 3.3 任务执行 API（实际）

```python
# vat/web/routes/tasks.py

class ExecuteRequest(BaseModel):
    video_ids: List[str]
    steps: List[str]       # 阶段名或阶段组
    gpu_device: str = "auto"
    force: bool = False
    generate_cli: bool = False

@router.post("/execute")
async def execute_task(request: ExecuteRequest):
    # 解析步骤（支持阶段组展开）
    steps = parse_steps(request.steps)
    
    # 通过 JobManager 提交任务
    job_id = job_manager.submit_job(
        video_ids=request.video_ids,
        steps=steps,
        gpu_device=request.gpu_device,
        force=request.force
    )
    
    return {"task_id": job_id, "status": "submitted"}
```

### 3.4 生成的 CLI 命令示例

```bash
python -m vat process -v VIDEO_ID1 -v VIDEO_ID2 -s download,whisper,split -g cuda:0
```

---

## 四、实时输出与日志

### 4.1 日志实现（实际）

**方案**：子进程输出到日志文件，前端通过 SSE 实时读取文件。

```
子进程 stdout/stderr -> job_logs/job_xxx.log -> SSE 读取 -> 前端显示
```

### 4.2 SSE 日志流实现（实际）

```python
# vat/web/routes/tasks.py

@router.get("/{task_id}/logs")
async def stream_logs(task_id: str):
    """SSE 实时日志流 - 从日志文件读取"""
    job = job_manager.get_job(task_id)
    if not job or not job.log_file:
        raise HTTPException(404, "Task not found")
    
    async def event_generator():
        # 实时读取日志文件并推送
        with open(job.log_file, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.strip()}\n\n"
                else:
                    # 检查任务是否完成
                    job_manager.update_job_status(task_id)
                    job = job_manager.get_job(task_id)
                    if job.status in ('completed', 'failed', 'cancelled'):
                        yield f"event: complete\ndata: {job.status}\n\n"
                        break
                    await asyncio.sleep(0.3)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### 4.3 前端接收示例

```javascript
// task_detail.html 中的 SSE 连接
const eventSource = new EventSource(`/api/tasks/${taskId}/logs`);

eventSource.onmessage = (event) => {
    appendLog(event.data);
};

eventSource.addEventListener('complete', (event) => {
    updateTaskStatus(event.data);
    eventSource.close();
});
```

---

## 五、页面功能设计

### 5.1 整体布局

```
┌──────────────────────────────────────────────────────────────┐
│  VAT Video Handler & Translator                    [设置]   │
├──────────────────────────────────────────────────────────────┤
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐            │
│  │ Videos │  │Playlist│  │ Tasks  │  │ Files  │            │
│  └────────┘  └────────┘  └────────┘  └────────┘            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                     [主内容区域]                             │
│                                                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Videos 页面

#### 5.2.1 视频列表

| 字段 | 说明 |
|------|------|
| 缩略图 | 视频缩略图（如有） |
| 标题 | 视频标题 |
| 来源 | YouTube / 本地 / Playlist名 |
| 状态 | 各阶段完成情况（进度条或图标） |
| 操作 | 处理 / 查看文件 / 删除 |

#### 5.2.2 视频详情

```
┌───────────────────────────────────────────────────────────┐
│  视频标题                                                  │
├───────────────────────────────────────────────────────────┤
│  ┌─────────┐                                              │
│  │  缩略图  │   URL: https://youtube.com/watch?v=xxx      │
│  │         │   来源: Playlist - My Channel                │
│  │         │   添加时间: 2026-02-01 10:30                 │
│  └─────────┘                                              │
├───────────────────────────────────────────────────────────┤
│  处理状态                                                  │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │
│  │Download│→│  ASR   │→│Translate│→│ Embed  │→│ Upload │  │
│  │   ✓    │ │   ✓    │ │  进行中 │ │  待定  │ │  待定  │  │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │
│                                                           │
│  子阶段详情:                                              │
│  - WHISPER: ✓ 完成                                        │
│  - SPLIT: ✓ 完成                                          │
│  - OPTIMIZE: 进行中...                                    │
│  - TRANSLATE: 待定                                        │
├───────────────────────────────────────────────────────────┤
│  相关文件                                                  │
│  📁 video.mp4                           [查看]            │
│  📄 subtitle_raw.srt                    [查看] [编辑]     │
│  📄 subtitle_translated.srt             [查看] [编辑]     │
│  📁 video_embedded.mp4                  [查看]            │
├───────────────────────────────────────────────────────────┤
│  操作                                                      │
│  [执行下一步] [选择阶段执行...] [生成CLI命令]              │
└───────────────────────────────────────────────────────────┘
```

**视频详情页功能**：
- **阶段状态显示**：每个阶段显示完成/进行中/待处理/失败状态，带颜色区分
- **单阶段执行**：鼠标悬停显示执行按钮，可单独执行或强制重新执行某阶段
- **执行下一步**：自动执行第一个未完成的阶段
- **选择阶段执行**：跳转到新建任务页面，可选择多个阶段
- **生成 CLI 命令**：生成等价的命令行命令（仅显示，不执行）

#### 5.2.3 新建任务页面

```
┌─────────────────────────────────────────┐
│  新建处理任务                            │
├─────────────────────────────────────────┤
│  选择视频:                               │
│  [搜索视频标题...]                       │
│  ┌─────────────────────────────────────┐│
│  │ ☑ Video 1          0/7 (0%)        ││
│  │ ☐ Video 2          3/7 (42%)       ││
│  │ ...                                 ││
│  └─────────────────────────────────────┘│
│  [全选] [取消全选] [选择待处理] [选择可见]│
├─────────────────────────────────────────┤
│  执行阶段:                               │
│  ☑ 下载                                  │
│  ☑ ASR (语音识别)                        │
│    ├ ☑ Whisper (语音转文字)              │
│    └ ☑ Split (智能断句)                  │
│  ☑ 翻译                                  │
│    ├ ☑ Optimize (提示词优化)             │
│    └ ☑ Translate (翻译)                  │
│  ☑ 嵌入字幕                              │
├─────────────────────────────────────────┤
│  GPU 设置: [自动选择 ▼]                  │
│  ☐ 强制重新处理                          │
├─────────────────────────────────────────┤
│  [开始执行]  [生成 CLI 命令]             │
└─────────────────────────────────────────┘
```

**新建任务功能**：
- **视频搜索**：支持按标题搜索过滤视频列表
- **子阶段可选**：ASR 可单独选择 Whisper/Split，翻译可单独选择 Optimize/Translate
- **选择可见**：搜索过滤后可一键选中所有可见视频
- **生成 CLI 命令**：仅生成命令文本，不执行任务

### 5.3 Playlist 页面

#### 5.3.1 Playlist 列表

| 字段 | 说明 |
|------|------|
| 名称 | Playlist 名称 |
| 来源 | YouTube 频道名 |
| 视频数 | 总视频数 / 已处理数 |
| 最后同步 | 上次同步时间 |
| 操作 | 同步 / 处理全部 / 查看 / 删除 |

#### 5.3.2 Playlist 详情

```
┌───────────────────────────────────────────────────────────┐
│  Playlist: My Favorite Videos                              │
│  来源: https://youtube.com/playlist?list=xxx              │
│  总视频数: 50 | 已处理: 30 | 待处理: 20                   │
│  最后同步: 2026-02-01 15:00                               │
├───────────────────────────────────────────────────────────┤
│  操作:  [同步] [处理全部] [处理范围...]                    │
├───────────────────────────────────────────────────────────┤
│  视频列表 (按 playlist_index 排序)                        │
│  ┌─────┬────────────────────────┬────────┬───────┐       │
│  │ #   │ 标题                    │ 状态   │ 操作  │       │
│  ├─────┼────────────────────────┼────────┼───────┤       │
│  │ 1   │ First Video            │ ✓ 完成 │ 查看  │       │
│  │ 2   │ Second Video           │ ✓ 完成 │ 查看  │       │
│  │ ... │ ...                    │ ...    │ ...   │       │
│  │ 49  │ Newest Video           │ 待处理 │ 处理  │       │
│  │ 50  │ Just Added             │ 待处理 │ 处理  │       │
│  └─────┴────────────────────────┴────────┴───────┘       │
└───────────────────────────────────────────────────────────┘
```

#### 5.3.3 Playlist 同步机制

**两层异步优化**：
1. **第一层**：后台获取视频 `upload_date`（用于按发布日期排序）
2. **第二层**：获取到视频信息后，异步发起 LLM 翻译（线程池执行）

**日期补全机制**：
- 同步时会检查已存在视频是否缺少 `upload_date`
- 如果缺少，会补充获取（解决删除 Playlist 后重建导致的日期丢失问题）

**翻译复用**：
- 同步时的异步翻译结果存储到 `video.metadata['translated']`
- Download 阶段检查到已有翻译则跳过，避免重复调用 LLM

**不可用视频统计**：
- Playlist 进度统计区分：总视频、已完成、待处理、失败、不可用
- 不可用视频（`metadata.unavailable=True`）不计入待处理数
- 处理任务时自动跳过不可用视频

#### 5.3.4 批量处理交互

**从 Playlist 选中视频处理**：
- 选中视频后点击"处理选中"会跳转到新建任务页面
- 用户可选择执行阶段、GPU 设置等（而非直接执行全流程）
- 返回链接正确指向来源页面（Playlist 详情）

#### 5.3.5 范围处理对话框

```
┌─────────────────────────────────────────┐
│  范围处理                                │
├─────────────────────────────────────────┤
│  选择方式:                               │
│  ◉ 按索引范围                            │
│  ○ 仅未处理的视频                        │
│  ○ 最新 N 个                            │
│                                          │
│  起始索引: [1]    结束索引: [10]         │
│                                          │
├─────────────────────────────────────────┤
│           [取消]  [应用并处理]           │
└─────────────────────────────────────────┘
```

**范围处理说明**：
- 点击"范围处理"按钮打开对话框
- 支持三种选择方式：按索引范围、仅未处理、最新N个
- 应用后跳转到新建任务页面，预选符合条件的视频
- 自动排除不可用视频

### 5.4 Tasks 页面

#### 5.4.1 任务列表

| 字段 | 说明 |
|------|------|
| ID | 任务 ID |
| 类型 | 单视频 / Playlist / 批量 |
| 状态 | 运行中 / 完成 / 失败 |
| 进度 | 进度条 |
| 时间 | 创建时间 / 耗时 |
| 操作 | 查看日志 / 取消 |

#### 5.4.2 任务详情/日志

```
┌───────────────────────────────────────────────────────────┐
│  任务 #abc123                                    [X 关闭] │
├───────────────────────────────────────────────────────────┤
│  状态: 运行中                                              │
│  进度: ████████░░░░░░░░ 50%                               │
│  当前: 处理视频 "Some Video Title" - TRANSLATE 阶段       │
├───────────────────────────────────────────────────────────┤
│  实时日志:                                                │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ [10:30:01] 开始处理视频: xxx                        │ │
│  │ [10:30:02] 下载完成                                 │ │
│  │ [10:30:05] ASR 开始，使用 GPU cuda:1                │ │
│  │ [10:31:20] ASR 完成，识别到 150 个片段              │ │
│  │ [10:31:21] SPLIT 开始...                            │ │
│  │ [10:31:25] SPLIT 完成，生成 45 个句子              │ │
│  │ [10:31:26] OPTIMIZE 开始...                         │ │
│  │ [10:31:30] OPTIMIZE 完成                            │ │
│  │ [10:31:31] TRANSLATE 开始...                        │ │
│  │ [10:32:00] 翻译进度: 20/45                          │ │
│  │ █                                                    │ │
│  └─────────────────────────────────────────────────────┘ │
│                                           [导出日志]      │
├───────────────────────────────────────────────────────────┤
│  [重新运行] [删除任务] (任务完成/失败时显示)               │
│  [取消任务] (任务运行中时显示)                            │
└───────────────────────────────────────────────────────────┘
```

**任务操作按钮**：
- **重新运行**：基于原任务参数创建新任务（仅已完成/失败/取消任务可用）
- **删除任务**：删除任务记录和日志文件（仅已完成/失败/取消任务可用）
- **取消任务**：发送 SIGTERM 终止运行中的任务

---

## 六、文件管理功能

### 6.1 设计原则

1. **仅可见 output_dir 内的文件**：安全限制，不暴露系统其他目录
2. **支持常见格式查看**：文本（srt, txt, json）、视频（mp4）、图片（jpg, png）
3. **编辑功能可选**：如实现简单，支持文本文件编辑

### 6.2 API 实现

```python
# vat/web/routes/files.py

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
import os

from vat.config import Config

router = APIRouter(prefix="/api/files", tags=["files"])


@router.get("/list")
async def list_files(
    path: str = Query("", description="相对于 output_dir 的路径"),
    config: Config = None
):
    """列出目录内容"""
    output_dir = Path(config.output_dir)
    target_dir = output_dir / path
    
    # 安全检查：确保在 output_dir 内
    try:
        target_dir.resolve().relative_to(output_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    
    if not target_dir.exists():
        raise HTTPException(404, "Directory not found")
    
    if not target_dir.is_dir():
        raise HTTPException(400, "Not a directory")
    
    items = []
    for item in target_dir.iterdir():
        items.append({
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None,
            "modified": item.stat().st_mtime
        })
    
    return {
        "path": path,
        "items": sorted(items, key=lambda x: (x["type"] != "directory", x["name"]))
    }


@router.get("/view")
async def view_file(
    path: str = Query(..., description="相对于 output_dir 的文件路径"),
    config: Config = None
):
    """查看文件内容"""
    output_dir = Path(config.output_dir)
    target_file = output_dir / path
    
    # 安全检查
    try:
        target_file.resolve().relative_to(output_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    
    if not target_file.exists():
        raise HTTPException(404, "File not found")
    
    if not target_file.is_file():
        raise HTTPException(400, "Not a file")
    
    # 根据文件类型返回
    suffix = target_file.suffix.lower()
    
    if suffix in [".srt", ".txt", ".json", ".yaml", ".md"]:
        # 文本文件：直接返回内容
        content = target_file.read_text(encoding="utf-8")
        return {"type": "text", "content": content}
    
    elif suffix in [".mp4", ".webm", ".mkv"]:
        # 视频文件：返回流式响应
        return FileResponse(target_file, media_type="video/mp4")
    
    elif suffix in [".jpg", ".jpeg", ".png", ".gif"]:
        # 图片文件
        media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", 
                       ".png": "image/png", ".gif": "image/gif"}
        return FileResponse(target_file, media_type=media_types[suffix])
    
    else:
        raise HTTPException(400, f"Unsupported file type: {suffix}")


@router.put("/save")
async def save_file(
    path: str = Query(...),
    content: str = None,
    config: Config = None
):
    """
    保存文件（可选功能）
    
    仅支持文本文件
    """
    output_dir = Path(config.output_dir)
    target_file = output_dir / path
    
    # 安全检查
    try:
        target_file.resolve().relative_to(output_dir.resolve())
    except ValueError:
        raise HTTPException(403, "Access denied")
    
    if not target_file.exists():
        raise HTTPException(404, "File not found")
    
    suffix = target_file.suffix.lower()
    if suffix not in [".srt", ".txt", ".json"]:
        raise HTTPException(400, "Only text files can be edited")
    
    # 备份原文件
    backup_path = target_file.with_suffix(target_file.suffix + ".bak")
    import shutil
    shutil.copy(target_file, backup_path)
    
    # 写入新内容
    target_file.write_text(content, encoding="utf-8")
    
    return {"status": "saved", "backup": str(backup_path)}
```

### 6.3 前端文件浏览器

```
┌───────────────────────────────────────────────────────────┐
│  文件浏览器                                     [X 关闭] │
├───────────────────────────────────────────────────────────┤
│  路径: /output/videos/abc123/                [上级目录]  │
├───────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐ │
│  │ 📁 ..                                               │ │
│  │ 📁 subtitles/                                       │ │
│  │ 📄 video.mp4                        150 MB   [▶]   │ │
│  │ 📄 subtitle_raw.srt                 12 KB   [👁]   │ │
│  │ 📄 subtitle_translated.srt          15 KB   [👁✏]  │ │
│  │ 📄 metadata.json                    2 KB    [👁]   │ │
│  └─────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘

[▶] = 播放  [👁] = 查看  [✏] = 编辑
```

---

## 七、CLI 等价性保证

### 7.1 设计原则

Web 界面的任何操作都必须能够映射到等价的 CLI 命令，且执行结果完全相同。

### 7.2 实现方式

#### 7.2.1 共享核心接口

```
┌─────────────────────────────────────────────────────────┐
│                     调用层                               │
│   ┌─────────┐                         ┌─────────┐      │
│   │   CLI   │                         │   Web   │      │
│   └────┬────┘                         └────┬────┘      │
│        │                                   │            │
│        v                                   v            │
│   ┌─────────────────────────────────────────────────┐  │
│   │              统一服务层                          │  │
│   │   - PlaylistService                             │  │
│   │   - VideoProcessor                              │  │
│   │   - Database                                    │  │
│   └─────────────────────────────────────────────────┘  │
│                          │                              │
│                          v                              │
│   ┌─────────────────────────────────────────────────┐  │
│   │              核心 Pipeline                       │  │
│   │   - Downloader                                  │  │
│   │   - ASR (WhisperASR)                            │  │
│   │   - Translator                                  │  │
│   │   - Embedder (FFmpegWrapper)                    │  │
│   └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### 7.2.2 CLI 命令生成

每个 Web 操作都应能生成对应的 CLI 命令：

```python
# vat/web/utils/cli_generator.py

class CLIGenerator:
    """生成等价的 CLI 命令"""
    
    @staticmethod
    def process_videos(
        video_ids: List[str],
        steps: List[str],
        gpu_id: Optional[str] = None
    ) -> str:
        """生成 process 命令"""
        cmd = ["vat", "process"]
        
        for vid in video_ids:
            cmd.append(f"--video {vid}")
        
        cmd.append(f"--steps {','.join(steps)}")
        
        if gpu_id:
            if gpu_id == "auto":
                cmd.append("--gpu auto")
            elif gpu_id.startswith("cuda:"):
                cmd.append(f"--gpu {gpu_id[5:]}")
            elif gpu_id == "cpu":
                cmd.append("--cpu")
        
        return " ".join(cmd)
    
    @staticmethod
    def add_playlist(url: str) -> str:
        return f"vat playlist add {url}"
    
    @staticmethod
    def sync_playlist(playlist_id: str) -> str:
        return f"vat playlist sync {playlist_id}"
    
    @staticmethod
    def process_playlist_range(
        playlist_id: str,
        start: int,
        end: int,
        steps: List[str]
    ) -> str:
        return f"vat playlist process {playlist_id} --range {start}:{end} --steps {','.join(steps)}"
```

#### 7.2.3 前端显示

执行对话框中的 "生成 CLI 命令" 选项：

```
执行成功！

Task ID: abc123
状态: 已提交

等价 CLI 命令:
┌─────────────────────────────────────────────────────┐
│ vat process --video xxx --steps asr,translate      │
│              --gpu auto                             │
└─────────────────────────────────────────────────────┘
                                        [复制命令]
```

---

## 八、实施计划

### 8.1 依赖关系

```
Phase 1 (Pipeline 改动) ──────────────────────────────┐
  ├── GPU 调度机制                                    │
  ├── 子阶段独立化                                    │
  └── Playlist 管理                                   │
                                                      │
Phase 2 (Web UI) ─────────────────────────────────────┤
  ├── 后端 API (依赖 Phase 1)                         │
  ├── 任务执行机制                                    │
  ├── 实时日志 SSE                                    │
  └── 前端页面                                        │
```

### 8.2 开发顺序

#### 第一周：后端 API 框架

1. **Day 1-2**: 搭建 FastAPI 应用结构
   - 创建 `vat/web/` 目录结构
   - 配置路由、依赖注入
   - 基础 CRUD API（videos, playlists）

2. **Day 3-4**: 任务执行机制
   - 实现 `TaskService`
   - 后台线程执行
   - 任务状态管理

3. **Day 5**: SSE 日志流
   - 实现日志队列
   - SSE endpoint
   - `progress_callback` 集成

#### 第二周：前端开发

4. **Day 6-7**: 前端框架搭建
   - React + Vite 初始化
   - TailwindCSS + shadcn/ui 配置
   - 路由和布局

5. **Day 8-9**: 核心页面实现
   - Videos 页面（列表、详情、执行对话框）
   - Playlists 页面
   - Tasks 页面（含实时日志）

6. **Day 10**: 文件管理
   - 文件浏览 API
   - 前端文件浏览器
   - 文本编辑功能（可选）

#### 第三周：集成与测试

7. **Day 11-12**: 功能集成
   - CLI 等价性验证
   - 端到端流程测试

8. **Day 13-14**: 优化与文档
   - 性能优化
   - 错误处理完善
   - 用户文档

### 8.3 验证点

| 验证项 | 验证方法 |
|--------|----------|
| Web 执行 = CLI 执行 | 相同输入，比较输出文件 MD5 |
| 实时日志推送 | 观察日志延迟 < 1s |
| 任务取消 | 取消后确认进程停止 |
| 文件安全 | 尝试访问 output_dir 外路径，应返回 403 |
| GPU 选择 | 验证任务使用指定/自动选择的 GPU |
| Playlist 同步 | 新增视频正确导入，playlist_index 正确 |

### 8.4 技术风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| SSE 连接断开 | 日志丢失 | 服务端缓存最近 N 条日志，支持重连恢复 |
| 长任务线程阻塞 | 资源占用 | 限制并发任务数；实现任务队列 |
| 大文件查看 | 内存溢出 | 视频采用流式响应；大文本分页 |
| 前后端状态不一致 | 显示错误 | 定期轮询刷新；错误边界处理 |

---

## 附录

### A. API 完整列表

```
GET    /api/videos                    # 视频列表
GET    /api/videos/{id}               # 视频详情
POST   /api/videos                    # 添加视频
DELETE /api/videos/{id}               # 删除视频
GET    /api/videos/{id}/files         # 视频文件列表

GET    /api/playlists                 # Playlist 列表
GET    /api/playlists/{id}            # Playlist 详情
POST   /api/playlists                 # 添加 Playlist
POST   /api/playlists/{id}/sync       # 同步 Playlist
DELETE /api/playlists/{id}            # 删除 Playlist

POST   /api/tasks/execute             # 执行任务
GET    /api/tasks                     # 任务列表
GET    /api/tasks/{id}                # 任务详情
POST   /api/tasks/{id}/cancel         # 取消任务
DELETE /api/tasks/{id}                # 删除任务
POST   /api/tasks/{id}/retry          # 重新运行任务
GET    /api/tasks/{id}/logs           # SSE 日志流

GET    /api/files/list                # 文件列表
GET    /api/files/view                # 查看文件
PUT    /api/files/save                # 保存文件
```

### B. 数据模型（Pydantic）

```python
# vat/web/schemas/video.py

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from enum import Enum


class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoCreate(BaseModel):
    url: Optional[str] = None
    local_path: Optional[str] = None


class VideoResponse(BaseModel):
    id: str
    title: str
    url: Optional[str]
    source_type: str
    status: VideoStatus
    playlist_id: Optional[str]
    playlist_index: Optional[int]
    created_at: datetime
    steps_completed: List[str]
    steps_pending: List[str]


class VideoListResponse(BaseModel):
    items: List[VideoResponse]
    total: int
    page: int
    page_size: int
```

### C. 前端技术栈详情

```json
{
  "framework": "React 18",
  "build": "Vite",
  "styling": "TailwindCSS 3.x",
  "components": "shadcn/ui",
  "icons": "Lucide React",
  "state": "React Query (TanStack Query)",
  "routing": "React Router v6",
  "forms": "React Hook Form + Zod"
}
```

---

> **文档结束**
> 
> 本文档与 [spec_phase1_pipeline.md](./spec_phase1_pipeline.md) 共同构成 VAT 增强计划的完整规划。
> 实施时应先完成 Phase 1（Pipeline），再进行 Phase 2（Web UI）。
