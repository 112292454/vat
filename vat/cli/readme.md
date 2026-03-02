# VAT 模块文档：CLI（命令行接口）

> 基于 Click 框架的命令行入口，是 VAT 所有处理能力的主要交互方式。
>
> WebUI 通过调用 CLI 子进程执行任务，CLI 不依赖 WebUI。

---

## 1. 模块组成

| 文件 | 职责 |
|------|------|
| `commands.py` | 主命令定义：所有用户面向的 CLI 命令（`vat` 命令组） |
| `tools.py` | `vat tools` 子命令组：标准化输出格式，供 WebUI JobManager 作为子进程调用 |
| `embed_service.py` | `vat embed-service` 子命令组：异步嵌字服务管理 |

---

## 2. 命令结构

```
vat
├── init                    # 初始化配置文件
├── download                # 下载 YouTube 视频
├── asr                     # 语音识别（Whisper + Split）
├── translate               # 翻译字幕
├── embed                   # 嵌入字幕到视频
├── pipeline                # 完整流水线（下载→转录→翻译→嵌入）
├── process                 # 细粒度阶段控制（核心命令）
├── status                  # 查看处理状态
├── clean                   # 清理视频数据
│
├── watch                   # 自动监控 Playlist 新视频
│
├── playlist                # Playlist 管理组
│   ├── add                 # 添加 Playlist
│   ├── sync                # 同步 Playlist
│   ├── list                # 列出所有 Playlist
│   ├── show                # 查看 Playlist 详情
│   ├── refresh             # 刷新视频元信息
│   ├── retranslate         # 重新翻译视频信息
│   └── delete              # 删除 Playlist
│
├── upload                  # 上传管理组
│   ├── video               # 上传单个视频
│   ├── playlist            # 批量上传 Playlist
│   ├── sync                # 合集同步（未入集视频→合集）
│   ├── update-info         # 批量更新已上传视频元信息
│   └── sync-db             # B站合集信息同步回数据库
│
├── tools                   # WebUI 子进程调用组
│   ├── fix-violation       # 修复违规视频
│   ├── sync-playlist       # 同步 Playlist
│   ├── refresh-playlist    # 刷新 Playlist
│   ├── retranslate-playlist # 重新翻译
│   ├── upload-sync         # 合集同步
│   ├── update-info         # 更新视频信息
│   ├── sync-db             # 同步数据库
│   ├── season-sync         # 合集同步
│   └── watch               # Watch 监控任务
│
└── embed-service           # 异步嵌字服务
    └── start               # 启动服务
```

---

## 3. 核心命令：process

`vat process` 是最常用的命令，支持细粒度阶段控制：

```bash
# 处理指定视频的所有阶段
vat process -v <video_id>

# 处理指定阶段
vat process -v <video_id> --stages whisper,split,optimize,translate

# 处理 Playlist 中的所有视频
vat process -p <playlist_id> --stages translate --force

# 多 GPU 并发
vat process -p <playlist_id> --gpu 0,1,2 --concurrency 3

# 定时上传模式
vat process -p <playlist_id> --stages upload --upload-cron "0 18 * * *" --upload-batch-size 3

# B站定时发布模式
vat process -p <playlist_id> --stages upload --upload-mode dtime --upload-cron "0 18 * * *"

# dry-run 预览
vat process -p <playlist_id> --stages upload --dry-run
```

### 阶段名称

| 阶段 | 说明 | 阶段组展开 |
|------|------|-----------|
| `download` | 下载视频 | — |
| `whisper` | Whisper ASR | — |
| `split` | LLM 断句 | — |
| `asr` | ASR 阶段组 | → `whisper` + `split` |
| `optimize` | LLM 字幕优化 | — |
| `translate` | LLM 翻译 | — |
| `embed` | 字幕嵌入 | — |
| `upload` | 上传 B 站 | — |

---

## 4. tools 子命令组

`vat tools` 为 WebUI 的 JobManager 提供标准化输出格式：

- `[N%]` — 进度百分比
- `[SUCCESS]` — 任务成功
- `[FAILED]` — 任务失败

WebUI 启动子进程执行 `vat tools xxx`，通过 stdout 解析进度和结果。这使 WebUI 不直接依赖任何处理模块，仅作为任务调度和展示层。

---

## 5. 关键代码索引

| 组件 | 文件位置 | 说明 |
|------|----------|------|
| CLI 根入口 | `vat/__main__.py` | `python -m vat` 入口 |
| 命令组定义 | `vat/cli/commands.py` | `@click.group()` 根命令 |
| 阶段解析 | `vat/cli/commands.py` | `parse_stages()` 字符串→TaskStep 列表 |
| 调度入口 | `vat/pipeline/scheduler.py` | `schedule_videos()` 被 process 命令调用 |
| 子进程工具 | `vat/cli/tools.py` | `_emit()` / `_success()` / `_failed()` |

---

## 6. 配置加载

CLI 通过 `get_config()` 延迟加载配置（全局单例）。LLM 环境变量（`OPENAI_API_KEY` / `OPENAI_BASE_URL`）由 `config.py` 的 `LLMConfig.__post_init__` 统一设置，CLI 层不重复设置。

支持 `--config / -c` 指定配置文件路径，默认使用 `config/config.yaml`。
