# Watch Mode — 自动监控 Playlist 并处理新视频

> 需求规格文档 v1.1 — **已实现**（Phase 1-4 完成）

---

## 1. 需求背景

当前 VAT 的视频处理流程已经完善（download → whisper → split → optimize → translate → embed → upload），但所有操作都需要人工触发。对于关注时效性的场景（如 VTuber 直播结束后尽快发布翻译版），需要一种**自动化机制**：持续监控 YouTube Playlist，发现新视频后自动完成全流程处理并上传到 B站。

### 核心用户场景

1. 用户启动 `vat watch -p PLxxxFubuki`
2. Watch 进程每小时检查一次该 Playlist
3. 发现新视频后，自动提交全流程处理任务（download → ... → upload）
4. 处理完成后自动添加到 B站合集
5. 与用户手动发起的其他处理任务**安全共存**，不产生资源冲突

---

## 2. 功能概述

### 2.1 Watch 模式

一个**持续运行的后台进程**，定期同步指定的 Playlist，发现新视频后自动提交处理任务。

- **默认模式**: 持续运行，按配置间隔轮询
- **单次模式**: `--once` 参数，检查一次后退出（可搭配系统 cron）
- **多 Playlist 支持**: 可同时监控多个 Playlist

### 2.2 资源协调锁（新增基础设施）

跨进程的下载/上传速率控制机制，确保无论有多少个 VAT 进程实例同时运行，YouTube 下载和 B站上传都不会超出安全速率。

### 2.3 WebUI 集成

在 WebUI 中新增独立的 **Watch Tab**（而非仅在 playlist 详情页附加），用于统一管理所有 watch 会话。可启动、停止、查看 Watch 任务的实时状态（基于数据库，非日志解析）。Playlist 详情页也提供快捷入口。

---

## 3. 详细设计

### 3.1 Watch 主循环

```
┌─────────────────────────────────────────────────┐
│                  vat watch 启动                   │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │  for each playlist in target_playlists:  │    │
│  │    1. sync_playlist() → 获取新视频列表    │    │
│  │    2. 过滤: 排除已有 pending/running task │    │
│  │    3. 过滤: 排除 unavailable 视频         │    │
│  │    4. 提交 process job (全流程+上传)      │    │
│  │    5. 记录 watch 状态到数据库             │    │
│  └──────────────────────────────────────────┘    │
│                      │                            │
│              sleep(interval)                      │
│                      │                            │
│              ↑ 循环 (除非 --once) ↑               │
└─────────────────────────────────────────────────┘
```

**关键行为**:

- Watch 进程**只负责发现新视频并提交任务**，不直接执行处理
- 实际处理由 JobManager 提交的子进程完成（复用现有 `vat process` 基础设施）
- 每轮检查开始前更新数据库中的 watch 状态（last_check_at、next_check_at）
- 每轮结束后记录发现的新视频数和提交的任务数

### 3.2 CLI 接口

```bash
# 持续监控（默认间隔 1 小时）
vat watch -p PLxxxFubuki

# 同时监控多个 Playlist
vat watch -p PLxxxFubuki -p PLxxxMarine

# 自定义间隔（分钟）
vat watch -p PLxxxFubuki --interval 30

# 单次检查后退出
vat watch -p PLxxxFubuki --once

# 指定 GPU
vat watch -p PLxxxFubuki --gpu cuda:0

# 自定义处理阶段（默认 all = 全流程含上传）
vat watch -p PLxxxFubuki --stages download,asr,translate,embed

# 并发数
vat watch -p PLxxxFubuki --concurrency 2
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-p, --playlist` | 必填 | Playlist ID（可多次指定） |
| `--interval` | 60 | 轮询间隔（分钟） |
| `--once` | false | 单次模式 |
| `--stages` | all | 处理阶段（逗号分隔） |
| `--gpu` | auto | GPU 设备 |
| `--concurrency` | 1 | 并发处理数 |
| `--force` | false | 强制重新处理 |
| `--fail-fast` | false | 失败时停止 |

### 3.3 配置文件 (`default.yaml`)

```yaml
watch:
  # 默认轮询间隔（分钟）
  default_interval: 60
  
  # 默认处理阶段（CLI 未指定 --stages 时使用）
  default_stages: "all"
  
  # 每轮最多提交的新视频数（0 = 使用代码安全上限 50）
  max_new_videos_per_round: 20
  
  # 提交任务时的默认并发数
  default_concurrency: 1
```

### 3.4 资源协调锁

#### 3.4.1 问题

多个 VAT 进程（watch 自动提交 + 用户手动执行）同时运行时：
- **YouTube 下载**: 并发下载触发 bot 检测 → 封 IP/cookie
- **B站上传**: 并发上传触发风控 → 限流/封号

#### 3.4.2 设计

基于 SQLite 的**跨进程资源锁**（利用已有的 SQLite WAL 模式数据库）。

**新增数据库表 `resource_locks`**:

```sql
CREATE TABLE resource_locks (
    resource_type TEXT PRIMARY KEY,  -- 'youtube_download' | 'bilibili_upload'
    holder_pid INTEGER,              -- 持有锁的进程 PID
    acquired_at TIMESTAMP,           -- 获取时间
    last_activity_at TIMESTAMP,      -- 最后活动时间（心跳）
    expires_at TIMESTAMP             -- 过期时间（防止死锁）
);
```

**锁行为**:

| 资源类型 | 最大并发 | 最小间隔 | 锁超时 |
|----------|----------|----------|--------|
| `youtube_download` | 1 | 配置项 `downloader.youtube.download_delay`（当前默认 10s） | 30 分钟 |
| `bilibili_upload` | 1 | 配置项 `uploader.bilibili.upload_interval`（当前默认值） | 60 分钟 |

**获取锁流程**:

```python
def acquire_lock(resource_type, timeout_seconds=300):
    """
    尝试获取资源锁
    
    1. 检查当前锁持有者
    2. 如果无锁 → 直接获取
    3. 如果有锁：
       a. 检查持有者 PID 是否存活（os.kill(pid, 0)）
       b. 检查锁是否已过期（expires_at < now）
       c. 检查心跳是否超时（last_activity_at 距今 > heartbeat_timeout）
       d. 满足 a/b/c 任一 → 清理死锁并抢占
       e. 否则 → 等待重试（指数退避，最多 timeout_seconds）
    4. 获取成功后启动心跳守护线程（每 30s 更新 last_activity_at 和 expires_at）
    """
```

**释放锁流程**:

```python
def release_lock(resource_type):
    """释放锁 + 记录完成时间（用于间隔控制）+ 停止心跳线程"""
```

**上下文管理器支持**:

```python
# 推荐使用方式：确保异常/kill 场景下锁的安全释放
with resource_lock('youtube_download', timeout=300) as lock:
    do_download()
# __exit__ 中自动 release_lock
# 进程被 kill 时，心跳停止 → 其他进程通过心跳超时检测到死锁 → 自动清理
```

**健壮性设计（应对进程被 kill 的场景）**:

| 场景 | 检测方式 | 恢复策略 |
|------|----------|----------|
| 进程正常退出 | `release_lock()` 在 `__exit__` / `finally` 中调用 | 锁立即释放 |
| 进程被 SIGTERM | Python 的 atexit / signal handler 触发清理 | 注册 atexit 回调释放锁 |
| 进程被 SIGKILL（kill -9） | PID 存活检测失败（`os.kill(pid, 0)` 抛 `ProcessLookupError`） | 下一个请求锁的进程检测到并清理 |
| 进程僵死（卡住） | 心跳超时（`last_activity_at` 距今 > 2 * heartbeat_interval） | 下一个请求锁的进程检测到并清理 |
| 数据库损坏 | SQLite WAL 模式的自动恢复 | 锁表可重建（幂等 CREATE IF NOT EXISTS） |

**间隔控制**:

锁释放时不立即允许下一个进程获取，而是检查上一次操作的完成时间，确保两次操作之间满足最小间隔要求。这通过一个额外的 `resource_cooldowns` 表实现：

```sql
CREATE TABLE resource_cooldowns (
    resource_type TEXT PRIMARY KEY,
    last_completed_at TIMESTAMP,     -- 上一次操作完成时间
    min_interval_seconds INTEGER     -- 最小间隔（秒）
);
```

#### 3.4.3 配置

```yaml
# 现有配置的复用（无需新增配置项）
downloader:
  youtube:
    download_delay: 10  # 已有，作为下载间隔的最小值

uploader:
  bilibili:
    upload_interval: 60  # 已有，作为上传间隔的最小值
```

#### 3.4.4 集成点

资源锁需要集成到现有的下载和上传执行路径中：

- **下载**: `VideoProcessor` 的 download 阶段执行前获取 `youtube_download` 锁
- **上传**: `VideoProcessor` 的 upload 阶段执行前获取 `bilibili_upload` 锁
- **所有调用者**自动受益（watch 模式、手动 process、cron upload 等）

### 3.5 错误处理与自动重试

Watch 提交的 process job 可能失败（网络问题、GPU OOM、YouTube 限流等）。需要合理的重试策略：

#### 3.5.1 Job 内部重试（已有）

`vat process` 命令已内建失败重试逻辑（最多 2 轮），这在单个 job 内部覆盖了大多数临时性错误。

#### 3.5.2 Watch 轮次间重试（新增）

Watch 每轮检查时，除了发现新视频，还应**检查本 session 之前提交过且失败的视频**，将它们纳入本轮的处理列表：

```python
def _get_retry_candidates(self, video_ids: List[str]) -> List[str]:
    """
    从本 session 之前提交过的视频中，找出失败的重试候选
    
    关键设计：只重试本 session 提交过的视频（_session_submitted_ids），
    不扫描全量 playlist 历史，避免首轮/误触时重处理大量历史视频。
    
    逻辑：
    1. 遍历 session 已提交的视频 ID 列表
    2. 排除 unavailable、running、全部完成的视频
    3. 如果视频有 failed task → 加入重试列表
    
    重试限制：
    - 同一视频最多重试 max_retries 次（默认 3，避免无限循环）
    - 通过 _retry_counts 字典跨轮次累计
    """
```

#### 3.5.3 Watch 进程自身的错误恢复

- **sync_playlist 失败**（网络/YouTube API）: 记录错误到 `watch_rounds`，跳过本轮，下一轮正常继续
- **JobManager 提交失败**: 同上，记录错误，下一轮重试
- **Watch 进程崩溃**: WebUI 可通过 PID 检测感知，提示用户重启
- **数据库锁定（SQLite busy）**: 现有的 `_retry_on_locked` 机制覆盖

### 3.6 防止重复处理

Watch 每轮同步后，在提交处理任务前需过滤掉：

1. **已有 pending/running task 的视频** — 避免重复提交
2. **已完成全流程的视频** — 除非 `--force`
3. **标记为 unavailable 的视频** — YouTube 已删除/私有化的视频

过滤逻辑：

```python
def get_processable_new_videos(db, video_ids, target_steps):
    """
    从新发现的视频中筛选出可以提交处理的视频
    
    排除条件:
    1. 视频有任何一个 target_step 处于 running 状态
    2. 视频已有 running 的 web_job（通过 web_jobs 表检查）
    3. 视频 metadata 中标记为 unavailable
    4. 视频所有 target_steps 都已 completed（除非 force=True）
    """
```

### 3.7 多 Watch 进程共存

用户可能同时运行多个 watch 进程（监控不同 playlist，或相同 playlist 意外重复启动）。

#### 3.7.1 不同 Playlist — 天然隔离

- 各自同步各自的 playlist，互不干扰
- 资源锁保证下载/上传的速率安全
- 各自在 `watch_sessions` 中有独立记录

#### 3.7.2 相同 Playlist — 需要检测和处理

- **启动时检测**: 新 watch 进程启动前，检查 `watch_sessions` 中是否已有 `status='running'` 且 PID 存活的同 playlist session
- **如果已有**: 打印警告并拒绝启动（除非 `--force` 覆盖，这会先停止旧 session）
- **PID 已死**: 自动接管（更新旧 session 为 `stopped`，启动新 session）

#### 3.7.3 JobManager 视角

多个 watch 进程可能同时调用 `JobManager.submit_job()`。SQLite WAL 模式支持并发写入（等待重试），加上现有的 `_retry_on_locked` 机制，这在实际中不会有问题。

### 3.8 Watch 状态追踪（数据库）

#### 3.8.1 新增表 `watch_sessions`

```sql
CREATE TABLE watch_sessions (
    session_id TEXT PRIMARY KEY,
    playlist_ids TEXT NOT NULL,       -- JSON array of playlist IDs
    status TEXT NOT NULL,             -- 'running' | 'stopped' | 'error'
    pid INTEGER,                      -- Watch 进程 PID
    config TEXT,                      -- JSON: {interval, stages, gpu, concurrency, ...}
    started_at TIMESTAMP NOT NULL,
    last_check_at TIMESTAMP,          -- 上一次检查时间
    next_check_at TIMESTAMP,          -- 下一次检查时间
    total_rounds INTEGER DEFAULT 0,   -- 已完成的轮次数
    total_new_found INTEGER DEFAULT 0,-- 累计发现的新视频数
    total_jobs_submitted INTEGER DEFAULT 0, -- 累计提交的任务数
    error TEXT,                       -- 最近一次错误信息
    stopped_at TIMESTAMP
);
```

#### 3.8.2 新增表 `watch_rounds`

```sql
CREATE TABLE watch_rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    round_number INTEGER NOT NULL,
    playlist_id TEXT NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    new_videos_found INTEGER DEFAULT 0,
    jobs_submitted INTEGER DEFAULT 0,
    submitted_video_ids TEXT,          -- JSON array
    submitted_job_ids TEXT,            -- JSON array
    error TEXT,
    FOREIGN KEY (session_id) REFERENCES watch_sessions(session_id)
);
```

Watch 进程每轮每个 playlist 写入一条 `watch_rounds` 记录，WebUI 通过查询这两个表展示实时状态。

### 3.9 WebUI 集成

#### 3.9.1 独立 Watch Tab（新增页面）

在 WebUI 导航栏新增 **"Watch"** Tab，作为所有 watch 会话的统一管理界面：

**Watch 列表页**（`/watch`）:
- 显示所有 watch sessions（运行中/已停止/异常）
- 每个 session 显示：监控的 playlist 列表、状态、运行时长、累计统计
- 提供"新建 Watch"按钮（选择 playlist + 配置参数 → 启动）
- 提供"停止"按钮（发送 SIGTERM）

**Watch 详情页**（`/watch/{session_id}`）:
- Session 基本信息（配置参数、启动时间等）
- 实时状态卡片：
  - 当前状态指示灯（运行中/已停止/异常）
  - 上次检查时间 / 下次检查时间（倒计时）
  - 累计：发现新视频数 / 提交任务数 / 成功数 / 失败数
- 轮次历史列表：
  - 每轮的检查时间、发现视频数、提交的 job 链接
  - 错误信息（如果有）
- 关联的 Job 列表（watch 提交的所有 process job）

#### 3.9.2 Playlist 详情页（快捷入口）

在 playlist 详情页添加：
- **"开始监控" 按钮** — 快捷创建该 playlist 的 watch session
- **Watch 状态摘要** — 如果该 playlist 正在被某个 watch session 监控，显示简要状态和跳转链接

#### 3.9.3 API 端点

```
# Watch 会话管理
POST   /api/watch/start                           # 启动 watch session（通过 JobManager 提交）
GET    /api/watch/sessions                        # 列出所有 sessions
GET    /api/watch/sessions/{session_id}           # 获取 session 详情
POST   /api/watch/sessions/{session_id}/stop      # 停止 running session（发送 SIGTERM）
DELETE /api/watch/sessions/{session_id}           # 删除已停止的 session 记录

# Watch 轮次
GET    /api/watch/sessions/{session_id}/rounds    # 获取轮次历史
```

#### 3.9.4 Watch 状态展示

WebUI 从数据库读取 `watch_sessions` 和 `watch_rounds` 表，而非解析日志。这确保：
- 状态信息结构化、可查询
- 即使 watch 进程意外终止，历史状态仍可查看
- 与 JobManager 的日志追踪互补（日志仍可用于调试）

### 3.10 任务提交策略

Watch 发现新视频后的任务提交方式：

```
新视频 [v1, v2, v3] → 提交一个 process job，包含所有新视频
                       video_ids=[v1, v2, v3]
                       steps=all（或配置的 stages）
                       playlist_id=当前 playlist
```

- **一个 playlist 的新视频合并为一个 job**，而非每个视频一个 job
- 复用现有 `vat process` 的批量处理能力（含重试、并发、download_delay）
- Upload 后自动触发 season sync（现有行为，无需修改）

### 3.11 与现有系统的交互

| 现有组件 | 交互方式 | 改动 |
|----------|----------|------|
| `PlaylistService.sync_playlist()` | Watch 直接调用 | 无需改动 |
| `JobManager.submit_job()` | Watch 通过它提交 process job | 无需改动 |
| `vat process` | 被 JobManager 作为子进程执行 | 集成资源锁 |
| `VideoProcessor` | 在 download/upload 阶段使用资源锁 | 需改动 |
| `web_jobs` 表 | Watch 提交的 job 正常记录在此 | 无需改动 |
| GPU 自动选择 | 现有 `select_best_gpu()` 已足够 | 无需改动 |
| Season sync | 现有 `_auto_season_sync()` 已在 upload 后触发 | 无需改动（已验证） |

**Season Sync 验证**:

`commands.py` 第 864-868 行：`process` 命令在 stages 包含 upload 且有 playlist 上下文时自动调用 `_auto_season_sync()`。Watch 提交的 job 格式为 `vat process -p PLAYLIST_ID -s all`，满足两个条件，因此 season sync **会自动触发**，无需额外代码。

`_auto_season_sync()` 内部已含重试逻辑（首次失败后等 30 分钟再试一次），覆盖了 B站索引延迟的场景。

---

## 4. 实施计划

### Phase 1: 资源协调锁（基础设施）

1. 在 `vat/utils/` 下新增 `resource_lock.py`
2. 新增 `resource_locks` + `resource_cooldowns` 数据库表
3. 在 `VideoProcessor` 的 download/upload 阶段集成锁
4. 测试：两个 `vat process` 进程并发执行时锁的行为

### Phase 2: Watch 核心逻辑

1. 新增 `vat/services/watch_service.py` — Watch 主循环逻辑
2. 新增 `vat/cli/` 下的 `watch` 命令（或作为 tools 子命令）
3. 新增 `watch_sessions` + `watch_rounds` 数据库表
4. `default.yaml` 新增 `watch` 配置节
5. 测试：CLI 模式下 watch 的基本工作流

### Phase 3: WebUI 集成

1. 新增 WebUI API 端点（watch start/stop/status）
2. 修改 `playlist_detail.html` — 添加 Watch 控制和状态展示
3. 将 `watch` 注册到 `TOOLS_TASK_TYPES` 和 JobManager
4. 测试：WebUI 启动/停止 watch，状态展示

### Phase 4: 测试与文档

1. 集成测试：watch 发现新视频 → 自动处理 → 上传 → season sync
2. 并发测试：watch + 手动 process 同时运行
3. 更新 README 和相关文档

---

## 5. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| YouTube 频繁 API 调用被封 | 无法同步新视频 | 间隔控制（默认 1h）+ cookies + remote_components |
| Watch 进程崩溃 | 停止监控 | 数据库记录状态，WebUI 可感知异常并提示 |
| 资源锁死锁 | 下载/上传卡死 | PID 存活检测 + 锁超时自动释放 |
| 大量新视频涌入 | 系统过载 | `max_new_videos_per_round` 配置限制（默认 20）+ 代码安全上限（50） |
| 与手动任务的 GPU 争抢 | OOM | 现有 `select_best_gpu()` + 显存检查 |

---

## 6. 不在 v1 范围内

以下功能可在后续迭代中添加：

- Webhook/通知（发现新视频时通知用户）
- Playlist 级别的 stages 覆写（不同 playlist 不同处理流程）
- 多机分布式 watch（当前仅支持单机多进程）
- 智能调度（根据系统负载动态调整间隔）
- Watch session 的暂停/恢复（区别于停止/重启）

---

## 7. 实现记录

### 已完成（v1.0 实现）

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | 资源协调锁（`vat/utils/resource_lock.py`）+ VideoProcessor 集成 | ✅ 20/20 测试通过 |
| Phase 2 | Watch 核心逻辑（`vat/services/watch_service.py`）+ CLI + DB 迁移 v7 | ✅ 33/33 测试通过 |
| Phase 3 | WebUI Watch Tab + API + Playlist 快捷入口 | ✅ 15/15 API 测试通过 |
| Phase 4 | 文档更新 | ✅ 完成 |

### 实现与设计的差异

1. **资源锁使用独立 SQLite 文件**：`resource_lock.py` 使用独立的 `resource_locks.db`（与主数据库同目录），而非共用主 DB。这样即使不通过 `Database` 类的场景也能使用锁。
2. **Watch CLI 注册为顶级命令**：`vat watch` 作为顶级命令，同时也注册为 `vat tools watch` 子命令（供 WebUI JobManager 调用）。
3. **watch_rounds 表增加 `retry_video_ids` 列**：记录每轮中哪些视频是重试候选，方便 WebUI 展示和调试。
4. **Watch 通过 JobManager 提交 process job**：`WatchService` 通过 `JobManager.submit_job(task_type='process')` 提交处理任务，而非直接 `subprocess.Popen`。这样 watch 提交的处理任务在 `web_jobs` 表有正式记录，WebUI 可见、可追踪、可取消。Watch 只做轻量编排（sync → filter → submit），不干涉处理流程内部。WebUI 通过 `JobManager` 提交 `watch` tools 任务来启动 watch 进程本身。
5. **视频候选范围修复**：`_get_processable_videos` 只接收 sync 报告的新增视频 ID，不再扫描全量 playlist（旧实现导致首次 watch 提交数千视频）。retry 范围限定为本 session 提交过的视频，防止历史失败被大规模重试。额外增加安全上限（代码硬上限 50 + 配置默认 20）。

### 已知限制与未来规划

- **直播视频处理**：当 watch 检测到正在直播的视频时，yt-dlp 会通过 HLS 实时录制直播流，download 阶段会阻塞直到直播结束。当前行为可接受（直播结束后自动完成下载并进入后续处理），但会长时间占用 process job。
- **（未来）流式 pipeline**：当前 pipeline 各阶段（download → whisper → translate → ...）是严格顺序执行的。未来可改为流式处理：边下载边做 ASR，边 ASR 边翻译，适用于直播实时处理场景。改动范围较大，仅作规划记录。

### 涉及的文件

- `vat/utils/resource_lock.py` — 跨进程资源锁
- `vat/services/watch_service.py` — Watch 主循环逻辑
- `vat/pipeline/executor.py` — 集成资源锁到 download/upload
- `vat/config.py` — WatchConfig 数据类
- `vat/database.py` — DB 迁移 v6→v7
- `vat/cli/commands.py` — `vat watch` 命令
- `vat/cli/tools.py` — `vat tools watch` 子命令
- `vat/web/routes/watch.py` — Watch API 路由
- `vat/web/app.py` — Watch 页面路由 + 路由注册
- `vat/web/templates/watch.html` — Watch 管理页面
- `vat/web/templates/base.html` — 导航栏添加 Watch 链接
- `vat/web/templates/playlist_detail.html` — 快捷 Watch 按钮
- `vat/web/jobs.py` — `watch` 注册到 TOOLS_TASK_TYPES
- `config/default.yaml` — watch 配置节
- `tests/test_resource_lock.py` — 资源锁测试（20 个）
- `tests/test_watch_service.py` — Watch 服务测试（33 个：session 管理 5 + 视频筛选 6 + round 记录 2 + 集成 8 + 边界场景 4 + 全流程 4 + 命令构建 3 + 安全防护 1）
- `tests/test_watch_api.py` — Watch API 路由测试（15 个：列表 4 + 详情 2 + 轮次 2 + 删除 4 + 停止 3）
