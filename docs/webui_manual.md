# VAT Web UI 使用手册

> **版本**: v1.2  
> **更新日期**: 2026-02-14  
> **状态**: 已实现

---

## 目录

1. [概述](#一概述)
2. [快速开始](#二快速开始)
3. [系统架构](#三系统架构)
4. [页面功能详解](#四页面功能详解)
5. [API 参考](#五api-参考)
6. [常见操作流程](#六常见操作流程)
7. [故障排除](#七故障排除)

---

## 一、概述

### 1.1 功能简介

VAT Web UI 是一个基于 FastAPI 的视频处理管理界面，提供以下核心功能：

- **视频管理**：查看、搜索、处理视频，查看处理状态和相关文件
- **Playlist 管理**：添加 YouTube Playlist，自动同步视频列表，批量处理
- **任务管理**：创建处理任务，实时查看进度和日志，取消/重试任务
- **Custom Prompts**：管理翻译和优化阶段使用的自定义提示词
- **文件浏览**：查看、编辑、下载视频处理生成的各类文件

### 1.2 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI (异步) |
| 前端模板 | Jinja2 + TailwindCSS + Alpine.js |
| 实时通信 | SSE (Server-Sent Events) |
| 任务执行 | 子进程 + CLI (生命周期解耦) |
| 数据存储 | SQLite (web_jobs 表) |

---

## 二、快速开始

### 2.1 启动服务

```bash
# 方式1：直接运行模块
python -m vat.web.app

# 方式2：使用 uvicorn
uvicorn vat.web.app:app --host 0.0.0.0 --port 8080

# 服务默认运行在 http://localhost:8080
```

### 2.2 首页导航

启动后访问 `http://localhost:8080`，顶部导航栏包含：

| 入口 | 说明 |
|------|------|
| 📹 视频 | 视频列表，查看所有已添加的视频 |
| 📂 Playlist | Playlist 管理，添加和同步 YouTube 播放列表 |
| ⚙️ 任务 | 任务管理，查看任务历史和运行状态 |
| 📝 Prompts | Custom Prompt 管理 |

---

## 三、系统架构

### 3.1 目录结构

```
vat/web/
├── app.py              # FastAPI 应用入口 + 页面路由
├── jobs.py             # 任务管理器 (JobManager)
├── routes/
│   ├── videos.py       # 视频 API
│   ├── playlists.py    # Playlist API
│   ├── tasks.py        # 任务执行 API
│   ├── files.py        # 文件浏览 API
│   └── prompts.py      # Prompt 管理 API
└── templates/          # Jinja2 模板
    ├── base.html           # 基础布局
    ├── index.html          # 视频列表
    ├── video_detail.html   # 视频详情
    ├── playlists.html      # Playlist 列表
    ├── playlist_detail.html # Playlist 详情
    ├── tasks.html          # 任务列表
    ├── task_new.html       # 新建任务
    ├── task_detail.html    # 任务详情
    └── prompts.html        # Prompt 管理
```

### 3.2 任务执行机制

**核心设计**：任务通过子进程执行 CLI 命令，与 Web 服务器生命周期完全解耦。

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI 主进程                        │
│  POST /api/tasks/execute                                │
│         ↓                                               │
│  JobManager.submit_job()                                │
│         ↓                                               │
│  subprocess.Popen("python -m vat process ...")          │
│         ↓                                               │
│  返回 task_id（立即响应）                                │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    独立子进程                            │
│  vat process -v VIDEO_ID -s download,whisper,...        │
│  stdout/stderr → job_logs/job_xxx.log                   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  前端通过 SSE 实时读取日志文件显示                        │
└─────────────────────────────────────────────────────────┘
```

**优点**：
- Web 服务重启不影响正在运行的任务
- 复用 CLI 逻辑，确保 Web 操作与命令行完全等价
- 任务状态持久化在 SQLite

**重要行为**：

1. **配置热更新**：每次任务执行都会重新加载 `config/default.yaml`。修改配置后，重新运行任务即可应用新配置，无需重启 Web 服务。

2. **阶段跳跃处理**（直通模式）：
   - **连续阶段**（如 split,optimize,translate）：正常执行，缺少前置输出则报错
   - **不连续阶段**（如只选 whisper + embed）：中间被跳过的阶段自动以"直通模式"执行
   - **直通模式实现**：系统会临时修改 config 中对应阶段的开关，让各阶段内部自行处理直通逻辑
   - **状态标记**：直通阶段完成后标记为 `SKIPPED` 而非 `COMPLETED`
   
   例如：选择 `whisper,embed` 时，系统会：
   1. 自动填充中间阶段：`whisper, split, optimize, translate, embed`
   2. 设置直通阶段的 config 开关：
      - `asr.split.enable = False`
      - `translator.llm.optimize.enable = False`
      - `translator.skip_translate = True`
   3. 各阶段正常执行，内部根据开关自动复制输入到输出
   4. 直通阶段标记为 `SKIPPED`，用户指定阶段标记为 `COMPLETED`

### 3.3 进度追踪系统

每个处理阶段的进度通过 `ProgressTracker` 追踪，日志格式：

```
[0%] 开始执行步骤: download
[12%] 视频下载完成
[20%] 步骤完成: download
[40%] 步骤完成: whisper
[60%] 步骤完成: split
[80%] 步骤完成: translate
[100%] 步骤完成: embed
```

前端每 2 秒轮询获取进度并更新进度条。

---

## 四、页面功能详解

### 4.1 视频列表页 (`/`)

![视频列表页](assets/webui_index.png)

**功能**：
- 显示所有已添加的视频
- 每行显示：标题、来源、时长、任务状态、进度、发布日期、创建时间
- 任务状态以图标形式显示 7 个阶段的完成情况（✓ 完成 / · 待处理）
- 支持按状态筛选：全部 / 已完成 / 进行中 / 失败（SQL 层面过滤，不加载全量数据）
- 支持搜索：按标题/频道/ID 搜索（SQL LIKE）
- 分页显示，默认每页 50 条

**操作**：
- 点击行进入视频详情页
- 统计卡片显示：视频总数、已完成、进行中、失败数量
- 支持按 Playlist 过滤
- 支持按阶段状态过滤（DL/WH/SP/OP/TR/EM/UP）

### 4.2 视频详情页 (`/video/{id}`)

![视频详情页](assets/webui_video_detail.png)

**信息展示**：
- 视频标题（原文 + 翻译）
- 来源类型、时长、源地址链接
- 处理进度时间线：7 个阶段的状态和完成时间
- 翻译信息：翻译标题、优化标题、简介摘要、推荐标签、推荐分区
- 相关文件列表

**阶段时间线**：
| 阶段 | 说明 |
|------|------|
| 下载 | 视频下载 + 信息翻译 |
| 语音识别 | Whisper ASR |
| 句子分割 | LLM 智能断句 |
| 提示词优化 | 字幕优化 |
| 翻译 | LLM 翻译 |
| 嵌入字幕 | FFmpeg 硬字幕 |
| 上传 | (待实现) |

**操作按钮**：
- **▶ (执行)**：执行该阶段
- **↺ (强制重做)**：强制重新执行该阶段（忽略缓存）
- **执行下一步**：自动执行第一个未完成的阶段
- **选择阶段执行...**：跳转到新建任务页面
- **生成 CLI 命令**：显示等价的命令行命令

**文件操作**：
- **查看**：在模态框中查看文件内容
- **编辑**：编辑文本文件（srt, txt, json, ass）
- **播放**：播放视频/音频文件
- **下载**：下载文件

### 4.3 新建任务页 (`/tasks/new`)

![新建任务页](assets/webui_task_new.png)

**视频选择**：
- 搜索框：按标题过滤视频
- 复选框列表：选择要处理的视频
- 每个视频显示当前进度（如 "6/7 (85%)"）
- 快捷按钮：全选 / 取消全选 / 选择待处理 / 选择可见

**阶段选择**：
```
☑ 下载
☑ ASR (语音识别)
  ├ ☑ Whisper (语音转文字)
  └ ☑ Split (智能断句)
☑ 翻译
  ├ ☑ Optimize (提示词优化)
  └ ☑ Translate (翻译)
☑ 嵌入字幕
```

子阶段可单独选择，支持精细控制处理流程。

**其他选项**：
- **GPU 设置**：自动选择 / GPU 0 / GPU 1 / 仅 CPU
- **并发数量**：设置同时并行处理的视频数（1-5，默认1=串行）。包含 GPU 步骤（whisper/embed）时建议不超过 2
- **强制重新处理**：勾选后忽略已完成状态，强制重新执行

**操作按钮**：
- **开始执行**：提交任务并跳转到任务详情页
- **生成 CLI 命令**：仅显示命令，不执行

### 4.4 任务列表页 (`/tasks`)

![任务列表页](assets/webui_tasks.png)

**显示内容**：
- 任务 ID、视频数、执行阶段、状态、进度、创建时间
- 状态：completed / running / failed / cancelled

**操作**：
- **+ 新建任务**：跳转到新建任务页
- 点击任务 ID 进入任务详情页

### 4.5 任务详情页 (`/tasks/{id}`)

**信息展示**：
- 任务 ID、创建时间、状态徽章
- 进度条（实时更新）
- 视频数量、执行阶段
- 处理视频：从数据库获取视频标题并显示为可点击链接
  - 单视频：直接显示标题
  - 2-3 个视频：逗号分隔显示
  - 3 个以上：默认折叠，显示前 2 个 + "等 N 个视频..."，点击可展开
- 当前阶段：从日志实时解析正在执行的步骤
- 错误信息（失败时显示）

**实时日志**：
- 绿色终端风格显示
- 通过 SSE 实时推送日志
- 自动滚动开关：勾选时日志自动滚动到底部
- 支持清空日志显示

**操作按钮**：
- **取消任务**：运行中任务可取消（发送 SIGTERM）
- **重新运行**：已完成/失败/取消任务可重新运行
- **删除任务**：删除任务记录

### 4.6 Playlist 列表页 (`/playlists`)

![Playlist 列表页](assets/webui_playlists.png)

**显示内容**：
- Playlist 标题、频道名、视频数、进度、最后同步时间

**操作**：
- **+ 添加 Playlist**：输入 YouTube Playlist URL 添加
- **同步**：增量同步 Playlist（获取新视频）
- **处理**：处理全部待处理视频
- **删除**：删除 Playlist（视频记录保留）

### 4.7 Playlist 详情页 (`/playlists/{id}`)

![Playlist 详情页](assets/webui_playlist_detail.png)

**信息展示**：
- Playlist 标题、进度统计（已完成/待处理/失败/不可用）
- 统计卡片可点击，跳转到首页对应 Playlist + 状态筛选
- 进度条
- Custom Prompt 配置（可为 Playlist 单独指定翻译/优化 Prompt）
- B 站上传配置（主播简称、合集、标题模板、版权类型，覆盖全局设置）

**视频列表**：
- 复选框选择视频
- 显示：标题、上传日期、时长、状态（已完成/失败/进行中/待处理）
- 按上传日期排序（旧 → 新）
- 分页显示，默认每页 100 条（仅对当前页视频查询进度，避免全量加载）

**操作按钮**：
- **同步 Playlist**：增量同步
- **处理选中**：处理选中的视频
- **处理范围**：打开范围选择对话框
- **强制重做选中**：强制重新处理选中视频

**范围处理对话框**：
- 按索引范围：指定起始和结束索引
- 仅未处理：选择所有待处理视频
- 最新 N 个：选择最新的 N 个视频

### 4.8 Prompts 管理页 (`/prompts`)

**功能**：
- 管理翻译和优化阶段使用的 Custom Prompts
- 支持创建、编辑、删除 Prompt
- 设置默认使用的 Prompt
- Playlist 可单独配置专属 Prompt

**Prompt 类型**：
- **翻译 Prompts**：用于 LLM 翻译阶段
- **优化 Prompts**：用于字幕优化阶段

---

## 五、API 参考

### 5.1 视频管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/videos` | 列出视频（支持分页、状态过滤） |
| GET | `/api/videos/{id}` | 获取视频详情 |
| POST | `/api/videos` | 添加视频 |
| DELETE | `/api/videos/{id}` | 删除视频 |

### 5.2 Playlist API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/playlists` | 列出 Playlist |
| GET | `/api/playlists/{id}` | 获取 Playlist 详情 |
| POST | `/api/playlists` | 添加 Playlist |
| POST | `/api/playlists/{id}/sync` | 同步 Playlist |
| DELETE | `/api/playlists/{id}` | 删除 Playlist |
| GET | `/api/playlists/{id}/prompt` | 获取 Prompt 配置 |
| PUT | `/api/playlists/{id}/prompt` | 设置 Prompt 配置 |

### 5.3 任务执行 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/tasks/execute` | 执行处理任务 |
| GET | `/api/tasks` | 列出任务历史 |
| GET | `/api/tasks/{id}` | 获取任务详情 |
| POST | `/api/tasks/{id}/cancel` | 取消任务 |
| DELETE | `/api/tasks/{id}` | 删除任务记录 |
| POST | `/api/tasks/{id}/retry` | 重新运行任务 |
| GET | `/api/tasks/{id}/logs` | SSE 实时日志流 |
| GET | `/api/tasks/{id}/log-content` | 获取完整日志内容 |

**执行任务请求体**：
```json
{
  "video_ids": ["video_id_1", "video_id_2"],
  "steps": ["download", "whisper", "split", "optimize", "translate", "embed"],
  "gpu_device": "auto",
  "force": false,
  "concurrency": 1,
  "generate_cli": false
}
```

`concurrency`: 并发处理的视频数量，默认 1 表示串行。设置 > 1 时多个视频将并行处理。

### 5.4 文件管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/files/view/{video_id}/{filename}` | 查看文件内容 |
| PUT | `/api/files/save/{video_id}/{filename}` | 保存文件 |
| GET | `/api/files/download/{video_id}/{filename}` | 下载文件 |

**支持的文件类型**：
- 文本：`.srt`, `.vtt`, `.txt`, `.json`, `.yaml`, `.md`, `.ass`, `.log`
- 视频：`.mp4`, `.webm`, `.mkv`（支持 Range 请求）
- 音频：`.mp3`, `.wav`, `.m4a`
- 图片：`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`

### 5.5 Prompt 管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/prompts` | 列出所有 Prompts |
| GET | `/api/prompts/{type}/{name}` | 获取 Prompt 内容 |
| POST | `/api/prompts` | 创建 Prompt |
| PUT | `/api/prompts/{type}/{name}` | 更新 Prompt |
| DELETE | `/api/prompts/{type}/{name}` | 删除 Prompt |

---

## 六、常见操作流程

### 6.1 处理单个 YouTube 视频

1. 访问 `/tasks/new`
2. 在搜索框输入视频标题或 ID 查找
3. 勾选目标视频
4. 选择要执行的阶段（默认全选）
5. 点击"开始执行"
6. 在任务详情页查看实时进度和日志

### 6.2 批量处理 Playlist 视频

1. 访问 `/playlists`，点击"+ 添加 Playlist"
2. 输入 YouTube Playlist URL，点击添加
3. 等待同步完成（自动获取视频列表和信息翻译）
4. 进入 Playlist 详情页
5. 选择视频（或使用"处理范围"功能）
6. 点击"处理选中"
7. 在新建任务页确认配置后执行

### 6.3 重新处理某个阶段

1. 进入视频详情页
2. 找到需要重新处理的阶段
3. 点击 ↺ 按钮强制重新执行
4. 确认后自动创建任务并跳转

### 6.4 编辑字幕文件

1. 进入视频详情页
2. 在"相关文件"区域找到字幕文件（如 `translated.srt`）
3. 点击"编辑"按钮
4. 在模态框中修改内容
5. 点击"保存"（原文件自动备份为 `.bak`）

### 6.5 使用 Custom Prompt

1. 访问 `/prompts`
2. 点击"+ 新建 Prompt"
3. 选择类型（翻译/优化），输入名称和内容
4. 保存后在"当前使用的 Prompt"处选择
5. 或在 Playlist 详情页为特定 Playlist 配置专属 Prompt

---

## 七、故障排除

### 7.1 任务失败

**查看错误信息**：
1. 进入任务详情页查看错误信息
2. 查看实时日志定位具体错误

**常见原因**：
- 网络问题导致下载失败
- GPU 显存不足
- 视频不可用（已删除/私有）

**处理方法**：
- 点击"重新运行"重试
- 或进入视频详情页单独执行失败的阶段

### 7.2 进度条不更新

**可能原因**：
- 任务刚启动，还未输出进度信息
- 浏览器 SSE 连接断开

**处理方法**：
- 刷新页面重新建立连接
- 检查任务是否仍在运行（查看进程状态）

### 7.3 Playlist 同步失败

**可能原因**：
- Playlist URL 无效
- 网络/代理问题
- YouTube 限流（429 错误）

**处理方法**：
- 检查 URL 是否正确
- 检查代理配置
- 稍后重试

### 7.4 文件无法查看

**可能原因**：
- 文件类型不支持
- 文件不存在（处理中断）

**处理方法**：
- 使用下载功能获取文件
- 重新执行相关阶段生成文件

---

## 附录

### A. 生成的 CLI 命令示例

```bash
# 处理单个视频全流程
python -m vat process -v VIDEO_ID -s download,whisper,split,optimize,translate,embed

# 处理多个视频，指定 GPU
python -m vat process -v VIDEO1 -v VIDEO2 -s download,whisper,split -g cuda:0

# 强制重新处理翻译阶段
python -m vat process -v VIDEO_ID -s translate -f

# 并行处理 3 个视频
python -m vat process -v VIDEO1 -v VIDEO2 -v VIDEO3 -c 3
```

### B. 任务状态说明

| 状态 | 说明 |
|------|------|
| pending | 任务已创建，等待执行 |
| running | 任务正在执行 |
| completed | 任务成功完成（所有视频均成功） |
| partial_completed | 部分完成（批量任务中部分视频成功、部分失败） |
| failed | 任务执行失败（所有视频均失败） |
| cancelled | 任务被用户取消 |

### C. 处理阶段对应关系

| Web UI 显示 | CLI 阶段名 | 说明 |
|------------|-----------|------|
| 下载 | download | 视频下载 + 信息翻译 |
| 语音识别 | whisper | Whisper ASR |
| 句子分割 | split | LLM 智能断句 |
| 提示词优化 | optimize | 字幕优化 |
| 翻译 | translate | LLM 翻译 |
| 嵌入字幕 | embed | FFmpeg 硬字幕嵌入 |
| 上传 | upload | 上传到 B 站 |
